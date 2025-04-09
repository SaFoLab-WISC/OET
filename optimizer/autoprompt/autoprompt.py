import gc
import json
import os

import numpy as np 
import torch
from torch.nn import CrossEntropyLoss
# https://huggingface.co/docs/accelerate/v0.11.0/en/memory#accelerate.find_executable_batch_size
from accelerate.utils import find_executable_batch_size
from ..baseline import SingleBehaviorRedTeamingMethod
from ..model_utils import get_template
from tqdm import tqdm
from ..check_refusal_utils import check_refusal_completions
from .autoprompt_utils import sample_control_autoprompt, get_nonascii_toks


# ============================== GCG CLASS DEFINITION ============================== #
class AutoPrompt(SingleBehaviorRedTeamingMethod):
    def __init__(self,
                 target_model,
                 num_steps=50,
                 adv_string_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
                 adv_string_emplifier=1,
                 allow_non_ascii=False,
                 search_width=512,
                 use_prefix_cache=True,
                 eval_steps=10,
                 eval_with_check_refusal=False,
                 check_refusal_min_loss=0.1,
                 early_stopping=False,
                 early_stopping_min_loss=0.1, # early stopping min loss criteria
                 **model_kwargs):
        """
        :param target_model: a dictionary specifying the target model (kwargs to load_model_and_tokenizer)
        :param num_steps: the number of optimization steps to use
        :param adv_string_init: the initial adversarial string to start with
        :param allow_non_ascii: whether to allow non-ascii characters when sampling candidate updates
        :param search_width: the number of candidates to sample at each step
        :param use_prefix_cache: whether to use KV cache for static prefix tokens (WARNING: Not all models support this)
        :param targets_path: the path to the targets JSON file
        :param eval_steps: the number of steps between each evaluation
        :param eval_with_check_refusal: whether to generate and check for refusal every eval_steps; used for early stopping
        :param check_refusal_min_loss: a minimum loss before checking for refusal
        :param early_stopping: an alternate form of early stopping, based solely on early_stopping_min_loss
        :param early_stopping_min_loss: the loss at which to stop optimization if early_stopping is True
        """
        super().__init__(target_model=target_model, **model_kwargs)  # assigns self.model, self.tokenizer, and self.model_name_or_path
        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.adv_string_emplifier = adv_string_emplifier
        self.allow_non_ascii = allow_non_ascii
        self.search_width = search_width
        self.use_prefix_cache = use_prefix_cache
        if use_prefix_cache and self.model.config.use_cache != use_prefix_cache:
            print(f"WARNING: setting model.config.use_cache={use_prefix_cache}")
            self.model.config.use_cache = use_prefix_cache
        
        ### Eval Vars ###
        self.eval_steps = eval_steps
        self.eval_with_check_refusal = eval_with_check_refusal
        self.check_refusal_min_loss = check_refusal_min_loss
        self.early_stopping = early_stopping
        self.early_stopping_min_loss = early_stopping_min_loss

        ## Instruction template
        template = get_template(**target_model)['prompt']
        self.template = template
        self.before_tc, self.after_tc = template.split("{instruction}")
        
    def generate_test_cases_single_behavior(self, behavior_dict, target_sentence=None, verbose=False, **kwargs):
        """
        Generates test cases for a single behavior

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param verbose: whether to print progress
        :return: a test case and logs
        """

        # starting search_batch_size, will auto reduce batch_size later if go OOM (resets for every new behavior)
        self.search_batch_size = self.search_width

        # ========== Behavior and Target str ==========
        after_adv = behavior_dict['after_adv']
        before_adv = behavior_dict['before_adv']

        
        if target_sentence:
            target = target_sentence
        elif 'target' in behavior_dict:
            target = behavior_dict['target']
        # behavior += ' '
        # if context_str:
        #     behavior = f"{context_str}\n\n---\n\n{behavior}"

        # print(f'Behavior: {behavior_id} || Target: {target}')
        ### Targeted Model and Tokenier ###
        model = self.model
        device = model.device
        tokenizer = self.tokenizer

        ### GCG hyperparams ###
        num_steps = self.num_steps
        adv_string_init = self.adv_string_init * self.adv_string_emplifier
        allow_non_ascii = self.allow_non_ascii
        search_width = self.search_width
        before_tc = self.before_tc
        after_tc = self.after_tc

        print(f'Before Adv: {before_adv}\n')
        print(f'Init Adv: {adv_string_init}\n')
        print(f'After Adv: {after_adv}\n')
        print(f'target: {target}\n')
        print('\n')

        ### Eval vars ###
        # eval_steps = self.eval_steps
        # eval_with_check_refusal = self.eval_with_check_refusal
        # check_refusal_min_loss = self.check_refusal_min_loss
        # early_stopping = self.early_stopping
        # early_stopping_min_loss = self.early_stopping_min_loss

        embed_layer = model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[0]  # can be larger than tokenizer.vocab_size for some models
        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(device))

        not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
        # remove redundant tokens
        not_allowed_tokens = torch.unique(not_allowed_tokens)

        # ========== setup optim_ids and cached components ========== #
        optim_ids = tokenizer(adv_string_init, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
        num_optim_tokens = len(optim_ids[0])

        cache_input_ids = tokenizer([before_tc], padding=False)['input_ids'] # some tokenizer have <s> for before_tc
        # cache_input_ids += tokenizer([context_str, behavior, after_tc, target], padding=False, add_special_tokens=False)['input_ids']
        cache_input_ids += tokenizer([before_adv, after_adv, after_tc, target], padding=False, add_special_tokens=False)['input_ids']
        cache_input_ids = [torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids] # make tensor separately because can't return_tensors='pt' in tokenizer
        before_ids, before_adv_ids, after_adv_ids, after_ids, target_ids = cache_input_ids
        before_embeds, before_adv_embeds, after_adv_embeds, after_embeds, target_embeds = [embed_layer(input_ids) for input_ids in cache_input_ids]
        # before_ids, context_ids, behavior_ids, after_ids, target_ids = cache_input_ids
        # before_embeds, context_embeds, behavior_embeds, after_embeds, target_embeds = [embed_layer(input_ids) for input_ids in cache_input_ids]
        
        if self.use_prefix_cache:
            # precompute KV cache for everything before the optimized tokens
            input_embeds = torch.cat([before_embeds, before_adv_embeds], dim=1)
            with torch.no_grad():
                outputs = model(inputs_embeds=input_embeds, use_cache=True)
                self.prefix_cache = outputs.past_key_values

        # ========== run optimization ========== #
        all_losses = []
        all_suffix = []
        pbar = tqdm(range(num_steps))
        for i in range(num_steps):
            # ========== compute coordinate token_gradient ========== #
            # create input
            optim_ids_onehot = torch.zeros((1, num_optim_tokens, vocab_size), device=device, dtype=model.dtype)
            optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
            optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)

            # forward pass
            # input_embeds = torch.cat([before_embeds, before_adv_embeds, optim_embeds, after_adv_embeds, after_embeds, target_embeds], dim=1)
            # outputs = model(inputs_embeds=input_embeds)
            if self.use_prefix_cache:
                input_embeds = torch.cat([optim_embeds, after_adv_embeds, after_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache)
            else:
                input_embeds = torch.cat([before_embeds, before_adv_embeds, optim_embeds, after_adv_embeds, after_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds)
            
            logits = outputs.logits

            # compute loss
            # Shift so that tokens < n predict n
            tmp = input_embeds.shape[1] - target_embeds.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

            # ========== Sample a batch of new tokens based on the coordinate gradient. ========== #
            sampled_top_indices = sample_control_autoprompt(optim_ids.squeeze(0), token_grad.squeeze(0), search_width, topk=256, temp=1, not_allowed_tokens=not_allowed_tokens)

            # ========= Filter sampled_top_indices that aren't the same after detokenize->retokenize ========= #
            sampled_top_indices_text = tokenizer.batch_decode(sampled_top_indices)
            new_sampled_top_indices = []
            count = 0
            for j in range(len(sampled_top_indices_text)):
                # tokenize again
                tmp = tokenizer(sampled_top_indices_text[j], return_tensors="pt", add_special_tokens=False).to(device)['input_ids'][0]
                # if the tokenized text is different, then set the sampled_top_indices to padding_top_indices
                if not torch.equal(tmp, sampled_top_indices[j]):
                    count += 1
                    continue
                else:
                    new_sampled_top_indices.append(sampled_top_indices[j])
            
            if len(new_sampled_top_indices) == 0:
                # print('All removals; defaulting to keeping all')
                count = 0
            else:
                sampled_top_indices = torch.stack(new_sampled_top_indices)
            
            # if count >= search_width // 2:
            #     print('\nLots of removals:', count)
            
            new_search_width = search_width - count

            # ========== Compute loss on these candidates and take the argmin. ========== #
            # create input
            sampled_top_embeds = embed_layer(sampled_top_indices)
            
            if self.use_prefix_cache:
                input_embeds = torch.cat([sampled_top_embeds,
                                            after_adv_embeds.repeat(new_search_width, 1, 1),
                                            after_embeds.repeat(new_search_width, 1, 1),
                                            target_embeds.repeat(new_search_width, 1, 1)], dim=1)
            else:
                input_embeds = torch.cat([before_embeds.repeat(new_search_width, 1, 1),
                                    before_adv_embeds.repeat(new_search_width, 1, 1),
                                    sampled_top_embeds,
                                    after_adv_embeds.repeat(new_search_width, 1, 1),
                                    after_embeds.repeat(new_search_width, 1, 1),
                                    target_embeds.repeat(new_search_width, 1, 1)], dim=1)
            
            # Auto Find Batch Size for foward candidates (each time go OOM will decay search_batch_size // 2)
            loss = find_executable_batch_size(self.compute_candidates_loss, self.search_batch_size)(input_embeds, target_ids)

            # ========== Update the optim_ids with the best candidate ========== #
            optim_ids = sampled_top_indices[loss.argmin()].unsqueeze(0)
            suffix = tokenizer.decode(optim_ids[0])
            # test_case_ids = torch.cat([context_ids, optim_ids, behavior_ids], dim=1)
            
            current_loss = loss.min().item()
            all_losses.append(current_loss)
            all_suffix.append(suffix)

            pbar.update(1)
            pbar.set_description(f'Training Loss: {current_loss}')
        pbar.close()
        logs = {'final_loss': current_loss, 
                'all_losses': all_losses, 
                'suffix': all_suffix,
        }

        return logs

    def compute_candidates_loss(self, search_batch_size, input_embeds, target_ids):
        if self.search_batch_size != search_batch_size:
            print(f"INFO: Setting candidates search_batch_size to {search_batch_size})")
            self.search_batch_size = search_batch_size
            torch.cuda.empty_cache()
            gc.collect()

        all_loss = []
        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]

                if self.use_prefix_cache:
                    # Expand prefix cache to match batch size
                    prefix_cache_batch = self.prefix_cache
                    current_batch_size = input_embeds_batch.shape[0]
                    prefix_cache_batch = []
                    for i in range(len(self.prefix_cache)):
                        prefix_cache_batch.append([])
                        for j in range(len(self.prefix_cache[i])):
                            prefix_cache_batch[i].append(self.prefix_cache[i][j].expand(current_batch_size, -1, -1, -1))
                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)
            logits = outputs.logits

            # compute loss
            # Shift so that tokens < n predict n
            tmp = input_embeds_batch.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids.repeat(input_embeds_batch.shape[0], 1)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(input_embeds_batch.shape[0], -1).mean(dim=1)
            all_loss.append(loss)

        return torch.cat(all_loss, dim=0)