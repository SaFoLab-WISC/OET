import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from tqdm import tqdm
from ..baseline import SingleBehaviorRedTeamingMethod
from ..model_utils import get_template
import json

# ============================== GBDA CLASS DEFINITION ============================== #
class GBDA(SingleBehaviorRedTeamingMethod):
    def __init__(self,
                target_model,
                num_optim_tokens=20,
                num_generate=5,
                num_steps=50,
                lr=0.2,
                noise_scale=0.2,
                targets_path=None,
                **model_kwargs):
        """
        :param target_model: a dictionary specifying the target model (kwargs to load_model_and_tokenizer)
        :param num_optim_tokens: the number of tokens in each test case
        :param num_steps: the number of optimization steps to use
        :param lr: the learning rate to use
        :param noise_scale: amount of noise to use for random initialization
        :param targets_path: the path to the targets JSON file
        """
        super().__init__(target_model, **model_kwargs)
        self.num_optim_tokens = num_optim_tokens
        self.num_steps = num_steps
        self.lr = lr
        self.noise_scale = noise_scale
        self.num_generate = num_generate

        template = get_template(**target_model)['prompt']
        self.template = template
        self.before_tc, self.after_tc = template.split("{instruction}")

    def generate_test_cases_single_behavior(self, behavior_dict, target_sentence=None, verbose=False):
        """
        Generates test cases for a single behavior

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param num_generate: the number of test cases to generate in parallel
        :param verbose: whether to print progress
        :return: a list of test case and a list of logs
        """
        # ========== Behavior and Target str ==========
        # get required variables from behavior dictionary
        # behavior = behavior_dict['Behavior']
        # context_str = behavior_dict['ContextString']
        after_adv = behavior_dict['after_adv']
        before_adv = behavior_dict['before_adv']
        num_generate = self.num_generate 

        if target_sentence:
            target = target_sentence
        elif 'target' in behavior_dict:
            target = behavior_dict['target']

        # handle setup for behaviors with context strings
        # if context_str:
            # behavior = f"{context_str}\n\n---\n\n{behavior}"

        # GBDA hyperparams
        num_optim_tokens = self.num_optim_tokens
        num_steps = self.num_steps
        lr = self.lr
        noise_scale = self.noise_scale

        # Vars
        model = self.model
        tokenizer = self.tokenizer
        device = model.device
        before_tc = self.before_tc
        after_tc = self.after_tc
        embed_layer = self.model.get_input_embeddings()

        print(f'Before Adv: {before_adv}\n')
        # print(f'Init Adv: {adv_string_init}\n')
        print(f'After Adv: {after_adv}\n')
        print(f'target: {target}\n')
        print('\n')

        # ========== Init Cache Embeds ========
        cache_input_ids = tokenizer([before_tc], padding=False)['input_ids'] # some tokenizer have <s> for before_tc
        # cache_input_ids += tokenizer([context_str, behavior, after_tc, target], padding=False, add_special_tokens=False)['input_ids']
        cache_input_ids += tokenizer([before_adv, after_adv, after_tc, target], padding=False, add_special_tokens=False)['input_ids']
        cache_input_ids = [torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids] # make tensor separately because can't return_tensors='pt' in tokenizer
        before_ids, before_adv_ids, after_adv__ids, after_ids, target_ids = cache_input_ids
        before_embeds, before_adv_embeds, after_adv_embeds, after_embeds, target_embeds = [embed_layer(input_ids) for input_ids in cache_input_ids]  

        # ========== setup log_coeffs (the optimizable variables) ========== #
        with torch.no_grad():
            embeddings = embed_layer(torch.arange(0, self.tokenizer.vocab_size, device=device).long())

        # ========== setup log_coeffs (the optimizable variables) ========== #
        log_coeffs = torch.zeros(num_generate, num_optim_tokens, embeddings.size(0))
        log_coeffs += torch.randn_like(log_coeffs) * noise_scale  # add noise to initialization
        log_coeffs = log_coeffs.cuda()
        log_coeffs.requires_grad_()

        # ========== setup optimizer and scheduler ========== #
        optimizer = torch.optim.Adam([log_coeffs], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)
        taus = np.linspace(1, 0.1, num_steps)

        all_losses = []
        # all_test_cases = []
        suffixs = []
        # ========== run optimization ========== #
        pbar = tqdm(range(num_steps))
        for i in range(num_steps):
            coeffs = torch.nn.functional.gumbel_softmax(log_coeffs, hard=False, tau=taus[i]).to(embeddings.dtype) # B x T x V
            optim_embeds = (coeffs @ embeddings[None, :, :]) # B x T x D

            input_embeds = torch.cat([before_embeds.repeat(num_generate, 1, 1), 
                                    # context_embeds.repeat(num_generate, 1, 1),
                                    # behavior_embeds.repeat(num_generate, 1, 1), 
                                    before_adv_embeds.repeat(num_generate, 1, 1),
                                    optim_embeds, 
                                    after_adv_embeds.repeat(num_generate, 1, 1),
                                    after_embeds.repeat(num_generate, 1, 1), 
                                    target_embeds.repeat(num_generate, 1, 1)
                                    ], dim=1)

            outputs = model(inputs_embeds=input_embeds)
            logits = outputs.logits
            
            # ========== compute loss ========== #
            # Shift so that tokens < n predict n
            tmp = input_embeds.shape[1] - target_embeds.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids.repeat(num_generate, 1)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(num_generate, -1).mean(dim=1)
        
            # ========== update log_coeffs ========== #
            optimizer.zero_grad()
            loss.mean().backward(inputs=[log_coeffs])
            optimizer.step()
            scheduler.step()

            # ========== retrieve optim_tokens and test_cases========== #
            optim_tokens = torch.argmax(log_coeffs, dim=2)
            suffix = tokenizer.batch_decode(optim_tokens, skip_special_tokens=True)
            # test_cases_tokens = torch.cat([context_ids.repeat(num_generate, 1), optim_tokens, behavior_ids.repeat(num_generate, 1)], dim=1)
            # test_cases = tokenizer.batch_decode(test_cases_tokens, skip_special_tokens=True)

            # ========= Saving and logging test cases ========= #
            current_loss = loss.detach().cpu().tolist()
            best_idx = np.argmin(current_loss)

            # all_test_cases.append(test_cases[best_idx])
            all_losses.append(current_loss[best_idx])
            suffixs.append(suffix[best_idx])

            pbar.update(1)
            best_loss = current_loss[best_idx]
            pbar.set_description(f'Training Loss: {best_loss}')

            # all_test_cases.append(test_cases)
            # all_losses.append(current_loss)
        pbar.close()
        # logs = [{
        #         'final_loss': current_loss[i],
        #         'all_losses': [loss[i] for loss in all_losses],
        #         'all_test_cases': [test_cases[i] for test_cases in all_test_cases],
        #         'suffix': [test_cases[i] for test_cases in all_test_cases]
        #     } for i in range(num_generate)]
        logs = {
                'all_losses': all_losses, 
                # 'all_test_cases': all_test_cases, 
                'suffix': suffixs,
            }
        return logs
