from eval.open_pipeline import EvalOptimizerModel
import yaml
import argparse

def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config_path",
            type=str,
            default="/workspace/Eval_optimizer/pipeline/optimizer_GCG_config.yaml",
        )
        parser.add_argument(
            "--train_path",
            type=str,
            default="/workspace/Eval_optimizer/data/train_table.csv",
        )
        parser.add_argument(
            "--test_path",
            type=str,
            default="/workspace/Eval_optimizer/data/test_table.csv",
        )
        parser.add_argument(
            "--test_case_path",
            type=str,
            default="/workspace/Eval_optimizer/output/model_StruQ/method_GCG_suffixs_table.json",
        )
        parser.add_argument(
            "--completion_path",
            type=str,
            default="",
        )
        parser.add_argument(
            "--dataset_name",
            type=str,
            default="hb",
        )
        parser.add_argument(
            "--train",
            action='store_true',
            default=False
        )
        parser.add_argument(
            "--evaluate",
            action='store_true',
            default=False
        )
        parser.add_argument(
            "--check_refuse",
            action='store_true',
            default=False
        )

        return parser.parse_args()

if __name__ == '__main__':


    args = parse_args()
    config_path = args.config_path
    train_path = args.train_path
    test_path = args.test_path
    test_case_path = args.test_case_path
    dataset_name = args.dataset_name
    train = args.train
    evaluate = args.evaluate
    check_refuse = args.check_refuse
    completion_path = args.completion_path
    
    with open(config_path, 'r') as f:
        config = yaml.full_load(f)
    
    optimizer_config = config['optimizer_config']
    eval_config = config['eval_config']
    optimizer_name = optimizer_config['method_name']
    attack_goal = config['attack_goal']
    target_sentence = config['target_sentence']
    adv_root = config['adv_root']
    adv_position = config['eval_config']['adv_position']
    n_training_samples = config['n_training_samples']

    if train:
        pipeline = EvalOptimizerModel(
            optimizer_name=optimizer_name, 
            attack_goal=attack_goal,
            optimizer_config=optimizer_config, 
            eval_config=eval_config, 
            target_sentence=target_sentence,
            adv_root=adv_root,
            train=True,
            n_training_samples=n_training_samples,
        )

        pipeline.train(train_case_path=train_path, dataset_name=dataset_name, verbose=True)
    else:
        pipeline = EvalOptimizerModel(
            optimizer_name=optimizer_name, 
            attack_goal=attack_goal,
            optimizer_config=optimizer_config, 
            eval_config=eval_config, 
            target_sentence=target_sentence,
            adv_root=adv_root,
            adv_position=adv_position,
            train=False,
        )
        if evaluate:
            pipeline.complete(test_case_path=test_case_path, test_sample_path=test_path, dataset_name=dataset_name, verbose=True)
        elif check_refuse:
            pipeline.check_refusal_completions(completion_path, dataset_name=dataset_name)
            

