import json
import gc
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
# https://huggingface.co/docs/accelerate/v0.11.0/en/memory#accelerate.find_executable_batch_size
from accelerate.utils import find_executable_batch_size
from ..baseline import SingleBehaviorRedTeamingMethod
from ..model_utils import load_model_and_tokenizer, get_template
# from ..check_refusal_utils import check_refusal_completions
from .templates import ADV_STRING_INIT, REFERENCES
from .utils import autodan_sample_control, load_mutate_model

class AutoDAN(SingleBehaviorRedTeamingMethod):

    def __init__(self,
                target_model, # Target model
                mutate_model, # Mutate model (GPT/Opensource models)
                # targets_path,
                num_steps=100,
                eval_steps=5,
                eval_with_check_refusal=True,
                batch_size=128,
                num_elites=0.1,
                crossover=0.5,
                num_points=5,
                mutation=0.01,
                forward_batch_size=None, # if None then will use automated batch_size
                target_sentence='Sure, Here is how to',
                init_adv_string='! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !',
                ref_path='',
                **kwargs,
                ):
        """
        :param target_model: a dictionary specifying the target model (kwargs to load_model_and_tokenizer)
        :param mutate_model: a dictionary specifying the mutate model (kwargs to load_mutate_model)
        :param targets_path: the path to the targets JSON file
        :param num_steps: the number of optimization steps to use
        :param eval_steps: the number of steps between each evaluation
        :param eval_with_check_refusal: whether to generate and check for refusal every eval_steps; used for early stopping
        :param batch_size: the number of candidates to consider at each step
        :param num_elites: AutoDAN mutation parameter
        :param crossover: AutoDAN mutation parameter
        :param num_points: AutoDAN mutation parameter
        :param mutation: AutoDAN mutation parameter
        :param forward_batch_size: the batch size to use for computing candidates
        """
        
        model_short_name = target_model.pop('model_short_name', None)
        developer_name = target_model.pop('developer_name', None)

        assert model_short_name, "Developer name needed to be specified for references template, for example: Vicuna or Llama2"
        assert developer_name, "Developer name needed to be specified for references template, for example: developer_name of Llama2 is Meta and developer_name of vicuna is LMSYS"

        # super().__init__(target_model)
        self.num_steps = num_steps
        self.eval_steps = eval_steps
        self.eval_with_check_refusal = eval_with_check_refusal
        self.batch_size = batch_size
        self.num_elites = int(self.batch_size * num_elites)
        self.crossover = crossover
        self.num_points = num_points
        self.mutation = mutation
        self.target_sentence = target_sentence

        # self.autodan_target_prefix = "(🔓Developer Mode Output) "

        reference = torch.load(ref_path, map_location='cpu')
        for o in range(len(reference)):
            reference[o] = reference[o].replace('[MODEL]', model_short_name)
            reference[o] = reference[o].replace('[KEEPER]', developer_name)
        self.references = reference
        self.forward_batch_size = forward_batch_size

        # ===== Init Mutate Model =====
        self.mutate_model, allocated_gpus = load_mutate_model(**mutate_model)

        # ===== Init Target Model =====
        self.model_name_or_path = target_model['model_name_or_path']

        # manually allocate target model into unallocated gpus
        num_gpus = target_model.get("num_gpus", 1)
        max_memory = {i: torch.cuda.mem_get_info(i)[0] if i >= allocated_gpus else 0 for i in range(0, allocated_gpus + num_gpus)}
        self.model, self.tokenizer = load_model_and_tokenizer(**target_model, max_memory=max_memory)
        
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token 

        # ====== Init instruction template ====
        self.template = get_template(**target_model)['prompt']
        self.before_tc, self.after_tc = self.template.split("{instruction}")

    def generate_test_cases_single_behavior(self, behavior_dict, target_sentence=None, verbose=False, **kwargs):
        """
        Generates test cases for a single behavior

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param verbose: whether to print progress
        :return: a test case and logs
        """
        # Initialize conversations
        tokenizer = self.tokenizer
        model = self.model

        # ==== Method Variables ====
        batch_size = self.batch_size
        num_elites = self.num_elites
        crossover = self.crossover
        num_points = self.num_points
        mutation = self.mutation
        references = self.references
        # autodan_target_prefix = self.autodan_target_prefix
        num_steps = self.num_steps
        eval_steps =self.eval_steps
        eval_with_check_refusal = self.eval_with_check_refusal
        template = self.template
        before_tc = self.before_tc
        after_tc = self.after_tc

        self.forward_batch_size = batch_size if self.forward_batch_size is None else self.forward_batch_size  # starting foward_batch_size, will auto reduce batch_size later if go OOM
        mutate_model = self.mutate_model
        
        # ========== Behavior meta data ==========
        after_adv = behavior_dict['after_adv']
        before_adv = behavior_dict['before_adv']
        
        if target_sentence:
            target = target_sentence
        elif 'target' in behavior_dict:
            target = behavior_dict['target']
    
        # ==== Early stopping vars =====
        previous_loss = None
        early_stopping_threshold = 20
                                           
        # ===== init target embeds =====
        embed_layer = model.get_input_embeddings()
        target_ids = tokenizer([target], padding=False, add_special_tokens=False, return_tensors='pt')['input_ids'].to(model.device)
        target_embeds = embed_layer(target_ids)
        new_adv_prefixes = self.references[:batch_size]

        logs = []
        test_case = before_adv + '\n' + after_adv
        pbar = tqdm(range(num_steps))
        for i in range(num_steps):
            # ======== compute loss and best candidates =======
            # candidate = USER: adv_prefix + behavior + ASSISTANT:

            # candidates = [before_tc + context_str + '\n' + p + behavior + after_tc for p in new_adv_prefixes]
            candidates = [before_tc + before_adv + '\n' + p + '\n' + after_adv + after_tc for p in new_adv_prefixes]
            candidates_tokens = tokenizer(candidates, max_length=2048, padding=True, add_special_tokens=False, return_tensors='pt').input_ids
            candidates_embeds = embed_layer(candidates_tokens.to(model.device))

            input_embeds = torch.cat([candidates_embeds, target_embeds.repeat(batch_size, 1, 1)], dim=1)
            loss = find_executable_batch_size(self.compute_candidates_loss, self.forward_batch_size)(input_embeds=input_embeds, target_ids=target_ids)

            best_new_adv_prefix_id = loss.argmin()
            best_new_adv_prefix = new_adv_prefixes[best_new_adv_prefix_id]
            current_loss = loss.min().item()
            
            # test_case = context_str + '\n' + best_new_adv_prefix + '\n + behavior'
            test_case = before_adv + '\n' + best_new_adv_prefix + '\n' + after_adv
            logs.append(dict(loss=current_loss, test_case=test_case, suffix=best_new_adv_prefix))

            # ========== Eval and Early Stopping ========== #
            # if (i % eval_steps == 0) or (i == num_steps - 1):
            #     log_str = f'\n===>Step {i}\n===>Test Case: {test_case}\n===>Loss: {current_loss}'
                
            #     # ====== Checking for refusal =======
            #     if eval_with_check_refusal:
            #         input_str = template.format(instruction=test_case)
            #         is_refusal, completions, _ = check_refusal_completions(model, tokenizer, inputs=[input_str])
            #         log_str += f"\n===>Completion: {completions[0]}"
            #         if not is_refusal[0]:
            #             break
                        
            #     if verbose:
            #         print(log_str)

            # ====== Checking for early stopping if loss stuck at local minima ======
            if previous_loss is None or current_loss < previous_loss:
                previous_loss = current_loss
                early_stopping_threshold = 10
            else:
                early_stopping_threshold -= 1
                if early_stopping_threshold <= 0:
                    break

            unfiltered_new_adv_prefixes = autodan_sample_control(control_prefixes=new_adv_prefixes,
                                                                score_list=loss.float().cpu().numpy().tolist(),
                                                                num_elites=num_elites,
                                                                batch_size=batch_size,
                                                                crossover=crossover,
                                                                num_points=num_points,
                                                                mutation=mutation,
                                                                mutate_model=mutate_model,
                                                                reference=references,
                                                                no_mutate_words=[])

            new_adv_prefixes = unfiltered_new_adv_prefixes
            pbar.update(1)
            pbar.set_description(f'Training Loss: {current_loss}')
        pbar.close()
        logs = {
            'all_losses': [item['loss'] for item in logs],
            'suffix': [item['suffix'] for item in logs],
            'all_test_cases': [item['test_case'] for item in logs],
        }
        return logs

    def compute_candidates_loss(self, forward_batch_size, input_embeds, target_ids):
        if self.forward_batch_size != forward_batch_size:
            print(f"INFO: Setting candidates forward_batch_size to {forward_batch_size})")
            self.forward_batch_size = forward_batch_size

        all_loss = []
        for i in range(0, input_embeds.shape[0], forward_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+forward_batch_size].to(torch.bfloat16)
                outputs = self.model(inputs_embeds=input_embeds_batch)

            logits = outputs.logits

            # compute loss
            # Shift so that tokens < n predict n
            tmp = input_embeds.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids.repeat(forward_batch_size, 1)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(forward_batch_size, -1).mean(dim=1)
            all_loss.append(loss)

        return torch.cat(all_loss, dim=0)        