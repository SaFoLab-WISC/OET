
import argparse

import pandas as pd
import random
import csv
import os
import yaml
from tqdm import tqdm
import numpy as np
import json
import torch
import yaml

from eval.utils import *
from eval.open_pipeline import EvalOptimizerModel

from optimizer.model_utils import load_model_and_tokenizer
from copy import copy 

import logging


class StruqEval(EvalOptimizerModel):

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
        :param eval_config: completion configuration
        :param adv_position: position of attack string in completion stage (before or after behavior)
        :param train: whether in train mode
        :param target_sentence: optimization target
        :param n_training_samples: the number of sampled training examples 
        """
        super().__init__(
            optimizer_name=optimizer_name, 
            attack_goal=attack_goal,
            optimizer_config=optimizer_config, 
            eval_config=eval_config, 
            adv_root=adv_root,
            adv_position=adv_position,
            train=train,
            target_sentence=target_sentence, 
            n_training_samples=n_training_samples, 
        )

    def custom_train(self, train_case_path, dataset_name, verbose=False):

        if verbose:
            print('='*50)
            print('Method Config')
            print(self.optimizer_config)
            print('='*50)

        if train_case_path.endswith('csv'):
            with open(train_case_path) as f:
                reader = csv.DictReader(f)
                train_behavior = list(reader)
        elif train_case_path.endswith('pkl'):
            train_behavior = pd.read_pickle(train_case_path)
            train_behavior = train_behavior.to_dict(orient='records')
        if self.n_training_samples:
            sampled_train_behavior = random.sample(train_behavior, k=self.n_training_samples)

        pbar = tqdm(total=len(sampled_train_behavior))
        save_dir = self.optimizer_config['save_root']
        infos = []

        save_logging = os.path.join(save_dir, f"training_{self.optimizer_name}_{self.model_name}.txt")
        logging.basicConfig(
            filename=save_logging,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info(f'dataset: {dataset_name}')

        print('Training Attack Prompt....')

        for i, item in enumerate(sampled_train_behavior):

            info = {}
            info['log'] = {}
            info['log']["loss"] = []
            info["log"]["suffix"] = []

            suffix_manager = build_prompt(
                item=item,
                optimizer_name=self.optimizer_name,
                attack_goal=self.attack_goal,
                tokenizer=self.optimizer.tokenizer,
                target_sentence=self.target_sentence,
                template_name=self.chat_template,
                system_message='You are a helpful assistant.',
                position=self.eval_config['adv_position'],
            )
            inputs = {}
            inputs['target'] = self.target_sentence

            prompt = suffix_manager.get_prompt(special_tokens=DELIMITERS['SpclSpclSpcl'])
            before_adv = self.optimizer.tokenizer(prompt).input_ids[:suffix_manager._goal_slice.stop]
            before_adv = torch.tensor(before_adv)
            before_adv = self.optimizer.tokenizer.decode(before_adv.tolist()).strip()
            after_adv = self.optimizer.tokenizer(prompt).input_ids[suffix_manager._goal_slice.stop:suffix_manager._assistant_role_slice.stop]
            after_adv = torch.tensor(after_adv)
            after_adv = self.optimizer.tokenizer.decode(after_adv.tolist()).strip()

            inputs['before_adv'] = before_adv
            inputs['after_adv'] = after_adv

            logs = self.optimizer.generate_test_cases_single_behavior(
                behavior_dict=inputs, target_sentence=self.target_sentence)
            losses = logs['all_losses']
            suffixs = logs['suffix']

            if len(losses) > 0:
                best_new_adv_suffix_id = np.argmin(losses)
                best_new_adv_suffix = suffixs[best_new_adv_suffix_id]
                adv_suffix = best_new_adv_suffix
            else:
                losses = []
                suffixs = suffixs
                adv_suffix = suffixs[-1]
            
            info["log"]["loss"] = str(losses)
            info["log"]["suffix"] = suffixs
            info['ContextString'] = item['ContextString']
            info['Behavior'] = item['Behavior']
            info["final_suffix"] = adv_suffix
            info["target"] = self.target_sentence
            infos.append(info)

            start_loss, end_loss = losses[0], losses[-1]
            logging.info(f'mean Loss of {i}th adv string: {start_loss} to {end_loss}')

            self.save_log(save_dir=save_dir, infos=infos, model_name=self.model_name, method_name=self.optimizer_name, dataset_name=dataset_name)
            pbar.update(1)
        pbar.close()
        
        ######################### Get Training ASR ######################
        print('inference training data')
        model, tokenizer = load_model_and_tokenizer(model_name_or_path=self.train_config['model_name_or_path'], padding_side='right')


        device = model.device
        gen_kwargs = self.eval_config['gen_config']
        gen_config = model.generation_config
        gen_config.max_new_tokens = gen_kwargs['max_new_tokens']
        gen_config.temperature = gen_kwargs['temperature']
        gen_config.do_sample = gen_kwargs['do_sample']
        keyword = self.eval_config['test_word']

        ASR = inference(
            optimizer_name=self.optimizer_name, 
            template_name=self.train_config['target_model']['chat_template'],
            attack_goal=self.attack_goal, 
            data=infos, 
            model=model, 
            adv_position=self.eval_config['adv_position'],
            tokenizer=tokenizer, 
            gen_config=gen_config, 
            target_sentence=target_sentence,
            keyword=keyword, 
            # system_message='You are a helpful assistant',
            system_message='You are a helpful assistant, answer the question in the instruction tag.',
            special_tokens=DELIMITERS['SpclSpclSpcl'],
            running_count=self.eval_config['running_count'], 
            debug=True
        )
        logging.info(f'Training ASR: {ASR}')

        
        
    def custom_complete(self, 
        test_case_path, 
        test_sample_path, 
        dataset_name, 
        verbose=False, 
        system_message='You are a helpful assistant'
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
        save_root = '/data/xiaogeng_liu/Memorization/OptimizationEval/output/StruQ'

        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, f'{self.optimizer_name}_{dataset_name}_completion.json')

        model_path = self.eval_config['model_name_or_path']
        model, tokenizer = load_model_and_tokenizer(model_path, device_map=device)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token

        print('Reading suffixs')
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

        completions = []
        print('Generating completion....')

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
                    system_message=system_message
                )

                input_ids = suffix_manager.get_input_ids(special_tokens=DELIMITERS['SpclSpclSpcl'])
                assistant_role_slice = suffix_manager._assistant_role_slice
                prompt = suffix_manager.get_prompt(special_tokens=DELIMITERS['SpclSpclSpcl'])
                
                tokens = tokenizer(prompt).input_ids[:suffix_manager._assistant_role_slice.stop]
                input_ids = torch.tensor(tokens)
                prompt = tokenizer.decode(input_ids.tolist()).strip()
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
                    'suffix': adv_string, 
                    'inputs': prompt, 
                    'completions': completion_all,
                    'instruction': context,
                    'user_input': behavior
                })
                pbar.update(1)
                with open(save_path, 'w') as json_file:
                    json.dump(completions, json_file, indent=4)
            pbar.close()



