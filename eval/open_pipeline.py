

import random
import csv
import pandas as pd
import os
import yaml
from tqdm import tqdm
import numpy as np
import json
import torch
import yaml
import ray
import json

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from optimizer.direct import get_method_class, init_method
from optimizer.model_utils import load_model_and_tokenizer, get_template
import logging

from eval.utils import *
from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template


class EvalOptimizerModel(object):

    def __init__(
        self, 
        optimizer_name, 
        attack_goal,
        optimizer_config, 
        eval_config, 
        adv_root,
        adv_position='before',
        train=True,
        target_sentence=None, 
        n_training_samples=10, 
    ):
        """
        :param optimizer_name: optimizer name
        :param attack_goal: attack goal
        :param optimizer_config: optimization configuration
        :param eval_config: infereence configuration
        :param adv_position: position of attack string in completion stage (before or after behavior)
        :param train: whether in train mode
        :param target_sentence: optimization target
        :param n_training_samples: the number of sampled training examples 
        """

        self.train_config = optimizer_config    
        self.eval_config = eval_config
        self.optimizer_config = optimizer_config
        self.optimizer_name = optimizer_name 
        self.n_training_samples = n_training_samples
        self.attack_goal = attack_goal
        self.target_sentence = target_sentence
        self.model_name = optimizer_config['model_name']
        self.adv_position = adv_position
        self.train_mode = train
        self.adv_root = adv_root
        self.method_name = optimizer_config['method_name']
        self.chat_template = optimizer_config['target_model']['chat_template']
        
        if self.train_mode:
            # ==== Filtering existed runs =====
            method_class = get_method_class(optimizer_name)

            # ==== Init Method ====
            self.optimizer = init_method(method_class, optimizer_config)
        

    
    def save_log(self, save_dir, infos, method_name, dataset_name):

        """
        :param save_dir: save root 
        :param infos: save target
        :param model_name: model name
        :param method_name: optimizer name
        :param dataset_name: tested dataset name
        """

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path_file = os.path.join(save_dir, f'{method_name}_{dataset_name}_suffixs.json')
        with open(save_path_file, 'w') as f:
            json.dump(infos, f, indent=4)


    def train(self, train_case_path, dataset_name, system_message='You are a helpful assistant.', verbose=False):

        """
        :param train_case_path: train data path
        :param dataset_name: train dataset name
        :param verbose: whether to print optimizer config
        """

        if verbose:
            print('='*50)
            print('Optimizer Config')
            print(self.optimizer_config)
            print('='*50)

        ######################### read training data ######################

        if train_case_path.endswith('csv'):
            with open(train_case_path) as f:
                reader = csv.DictReader(f)
                train_behavior = list(reader)
        elif train_case_path.endswith('pkl'):
            train_behavior = pd.read_pickle(train_case_path)
            train_behavior = train_behavior.to_dict(orient='records')
        elif train_case_path.endswith('json'):
            with open(train_case_path, 'r') as f:
                train_behavior = json.load(f)
        else:
            raise Exception('File Not Found')
        
        if self.n_training_samples:
            sampled_train_behavior = random.sample(train_behavior, k=self.n_training_samples)
        else:
            sampled_train_behavior = train_behavior
        

        ######################### Start Training ######################
        pbar = tqdm(total=len(sampled_train_behavior))
        save_dir = self.optimizer_config['save_root']
        
        print('Training Adversial String....')
        infos = []
        adv_strings = []
        os.makedirs(save_dir, exist_ok=True)
        save_logging = os.path.join(save_dir, f"training_{self.optimizer_name}_{self.model_name}.txt")
        logging.basicConfig(
            filename=save_logging,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info('Training Adversial String....')
        logging.info(f'Dataset: {dataset_name}')

        if self.optimizer_name != 'PAIR':
            tokenizer = self.optimizer.tokenizer
        else:
            # ref = self.optimizer.targetLM.model.get_model.remote()
            # ref = self.optimizer.targetLM.model.get_model
            # _, tokenizer = ray.get(ref)
            _, tokenizer = self.optimizer.targetLM.model.get_model()

        for i, item in enumerate(sampled_train_behavior):

            info = {}
            info['log'] = {}
            info['log']["loss"] = []
            info["log"]["suffix"] = []

            suffix_manager = build_prompt(
                item=item,
                optimizer_name=self.optimizer_name,
                attack_goal=self.attack_goal,
                tokenizer=tokenizer,
                target_sentence=self.target_sentence,
                template_name=self.chat_template,
                system_message=system_message,
                position=self.eval_config['adv_position'],
            )
            inputs = {}
            inputs['target'] = self.target_sentence

            prompt = suffix_manager.get_prompt()
            before_adv = tokenizer(prompt).input_ids[:suffix_manager._goal_slice.stop]
            before_adv = torch.tensor(before_adv)
            before_adv = tokenizer.decode(before_adv.tolist()).strip()
            after_adv = tokenizer(prompt).input_ids[suffix_manager._goal_slice.stop:suffix_manager._assistant_role_slice.stop]
            after_adv = torch.tensor(after_adv)
            after_adv = tokenizer.decode(after_adv.tolist()).strip()

            inputs['before_adv'] = before_adv
            inputs['after_adv'] = after_adv

            logs = self.optimizer.generate_test_cases_single_behavior(
                behavior_dict=inputs, target_sentence=self.target_sentence)
            losses = logs['all_losses']
            suffixs = logs['suffix']

            if self.method_name.lower() not in ['pair']:
                best_new_adv_suffix_id = np.argmin(losses)
            else:
                best_new_adv_suffix_id = np.argmax(losses)
            best_new_adv_suffix = suffixs[best_new_adv_suffix_id]
            adv_suffix = best_new_adv_suffix
            

            info["log"]["loss"] = str(losses)
            info["log"]["suffix"] = suffixs
            info["final_suffix"] = adv_suffix
            info['ContextString'] = item['ContextString']
            info['Behavior'] = item['Behavior']
            info["target"] = self.target_sentence

            infos.append(info)
            start_loss, end_loss = losses[0], losses[-1]
            logging.info(f'mean Loss of {i}th adv string: {start_loss} to {end_loss}')

            self.save_log(save_dir=save_dir, infos=infos, method_name=self.optimizer_name, dataset_name=dataset_name)
            adv_strings.append(adv_suffix)
            pbar.update(1)
        pbar.close()

        ######################### Calculate Training ASR ######################
        logging.info('='*50)
        logging.info('Inference on training data')
        if self.method_name.lower() not in ['pair']:
            model = self.optimizer.model
            tokenizer = self.optimizer.tokenizer
        else:
            # model, tokenizer = ray.get(self.optimizer.targetLM.model.get_model.remote())
            model, tokenizer = self.optimizer.targetLM.model.get_model()
        
        template_name = self.train_config['target_model']['chat_template']
        gen_kwargs = self.eval_config['gen_config']
        gen_config = model.generation_config
        gen_config.max_new_tokens = gen_kwargs['max_new_tokens']
        gen_config.temperature = gen_kwargs['temperature']
        gen_config.do_sample = gen_kwargs['do_sample']
        keyword = self.eval_config['test_word']
        
        ASR = inference(
            optimizer_name=self.optimizer_name, 
            template_name=template_name,
            attack_goal=self.attack_goal, 
            data=infos, 
            model=model, 
            adv_position=self.eval_config['adv_position'],
            tokenizer=tokenizer, 
            gen_config=gen_config, 
            keyword=keyword, 
            target_sentence=self.target_sentence,
            special_tokens=None,
            system_message=system_message,
            running_count=self.eval_config['running_count'], 
            debug=False
            # debug=True
            )
        logging.info(f"{self.optimizer_name} Training Average ASR: {ASR}")
        print(f"{self.optimizer_name} Training Average ASR: {ASR}")

        
    def complete(self, 
        test_case_path, 
        test_sample_path, 
        dataset_name, 
        verbose=False, 
        padding_side='left',
        pad_token=None,
        system_message='You are a helpful assistant',
      ):
        
        """
        :param test_case_path: test data path
        :param test_sample_path: suffix/prefix path
        :param verbose: whether to print completion config
        :param system_message: system prompt
        """

        if verbose:
            print('='*50)
            print('Evaluation Config')
            print(self.eval_config)
            print('='*50)

    
        gen_kwargs = self.eval_config['gen_config']
        device = self.eval_config['device']
        save_path = os.path.join(self.eval_config['save_root'], f'{self.optimizer_name}_{dataset_name}_completion.json')
        model_path = self.eval_config['model_name_or_path']
        model, tokenizer = load_model_and_tokenizer(model_path, padding_side=padding_side, pad_token=pad_token, device_map=device)

        print('Reading Data & Adv Strings')
        with open(test_case_path, 'r') as json_file:
            loaded_data = json.load(json_file)

        if test_sample_path.endswith('csv'):
            with open(test_sample_path) as f:
                reader = csv.DictReader(f)
                test_behavior = list(reader)
        elif test_sample_path.endswith('pkl'):
            test_behavior = pd.read_pickle(test_sample_path)
            test_behavior = test_behavior.to_dict(orient='records')
        elif test_sample_path.endswith('json'):
            test_behavior = json.load(open(test_sample_path))

        print('Generating completion....')
        completions = []
        gen_config = model.generation_config
        gen_config.max_new_tokens = gen_kwargs['max_new_tokens']
        gen_config.temperature = gen_kwargs['temperature']
        gen_config.do_sample = gen_kwargs['do_sample']

        for j in tqdm(range(len(loaded_data))):
            adv_string = loaded_data[j]['final_suffix']
            pbar = tqdm(total=len(test_behavior))
            for i, item in enumerate(test_behavior):

                context = item['ContextString']
                behavior = item['Behavior']       
                suffix_manager = build_prompt(
                    item=item,
                    optimizer_name=self.optimizer_name,
                    template_name=self.train_config['target_model']['chat_template'],
                    tokenizer=tokenizer,
                    attack_goal=self.attack_goal,
                    adv_string=adv_string,
                    target_sentence=self.target_sentence,
                    position=self.adv_position,
                    system_message=system_message,
                )
                input_ids = suffix_manager.get_input_ids(special_tokens=None)
                assistant_role_slice = suffix_manager._assistant_role_slice

                tokens = torch.tensor(input_ids[:assistant_role_slice.stop])
                prompt = tokenizer.decode(tokens, skip_special_tokens=True)

                completion = ''
                completion_all = []
                
                for _ in range(self.eval_config['running_count']):
                    output = generate(
                        model=model, 
                        tokenizer=tokenizer, 
                        input_ids=input_ids, 
                        assistant_role_slice=assistant_role_slice, 
                        gen_config=gen_config,
                    )
                    completion = tokenizer.decode(output.cpu().detach().tolist()).strip()
                    completion_all.append(completion)

                completions.append({
                    'adv string': adv_string, 
                    'inputs': prompt, 
                    'completions': completion_all,
                    'instruction': context,
                    'user_input': behavior
                })
                pbar.update(1)
                
                ### Save Result
                with open(save_path, 'w') as json_file:
                    json.dump(completions, json_file, indent=4)

            pbar.close()


    def check_refusal_completions(self, completion_path, dataset_name):
        
        """
        :param completion_path: completion path 
        :param dataset_name: tested dataset
        """
        print(f'Dataset: {dataset_name}')
        print('Reading completions')
        
        with open(completion_path, 'r') as json_file:
            loaded_data = json.load(json_file)

        infos = {}
        success_rates = []
        evaluate_data = {}
        keyward = self.eval_config['test_word']

        for item in loaded_data:
            key = item['adv string']
            if key not in evaluate_data.keys():
                evaluate_data[key] = [item] 
            else:
                evaluate_data[key].append(item)
        keys = evaluate_data.keys()
        
        for key in keys:
            injection_prompt = key
            infos[injection_prompt] = []

            items = evaluate_data[key]
            injection_success = []

            for item in tqdm(items, total=len(items)):
                completions = item['completions']      
                success = False 

                for completion in completions:
                    if keyward != None and keyward.lower() in completion.lower():
                            success = True
                            for prefix in REFUSAL_PREFIXES:
                                if prefix in completion:
                                    success = False
                    if success:
                        break 

                if success: 
                    injection_success.append(1)
                else:
                    injection_success.append(0)
                
            success_rates.append(np.mean(injection_success))

        success_rates = np.mean(success_rates)
        print("Average ASR: ", np.mean(success_rates))


    def custom_train(self, train_case_path, dataset_name, verbose=False):
        raise NotImplementedError

    def custom_complete(self, test_case_path, test_sample_path, verbose=False):
        raise NotImplementedError

    def custom_check(self):
        raise NotImplementedError