
import argparse
import os 

from eval.close_pipeline import EvalAPIModel

def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            default="",
        )
        parser.add_argument(
            "--key",
            type=str,
            default="",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.2,
        )
        parser.add_argument(
            "--top_p",
            type=float,
            default=0.95,
        )
        parser.add_argument(
            "--running_count",
            type=int,
            default=5,
        )
        parser.add_argument(
            "--max_n_tokens",
            type=int,
            default=64,
        )
        parser.add_argument(
            "--save_root",
            type=str,
            default="",
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
            "--dataset_name",
            type=str,
            default="hb",
        )
        parser.add_argument(
            "--attack_goal",
            type=str,
            default="print hello",
        )
        parser.add_argument(
            "--test_word",
            type=str,
            default="sudo rm -rf",
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
    running_count = args.running_count
    model_name_or_path = args.model_name_or_path
    attack_goal = args.attack_goal
    key = args.key 
    temperature = args.temperature 
    max_n_tokens = args.max_n_tokens
    top_p = args.top_p
    test_path = args.test_path
    test_case_path = args.test_case_path
    dataset_name = args.dataset_name
    save_root = args.save_root
    evaluate = args.evaluate
    check_refuse = args.check_refuse

    pipeline = EvalAPIModel(model_name_or_path=model_name_or_path, attack_goal=attack_goal,
        key=key, temperature=temperature, max_n_tokens=max_n_tokens, top_p=top_p, save_root=save_root, running_count=running_count)

    if evaluate:
        print(args)
        pipeline.complete(test_case_path=test_case_path, test_sample_path=test_path, dataset_name=dataset_name, verbose=True)
            
    elif check_refuse:
        pipeline.check_refusal(os.path.join(save_root, f'{model_name_or_path}_{dataset_name}.json'), args.test_word, dataset_name)
