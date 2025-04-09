import os 
from tqdm import tqdm
import random
import json

from datasets import load_dataset 
from datasets import load_from_disk
from eval.data import DataGenerator

class CustomGenerator(DataGenerator):

    def __init__(self):
        super().__init__()
    
    def custom_generate(self, save_path, processed_path, dataset_name='llamafactory/PubMedQA'):
        self.download(save_path=save_path, dataset_name=dataset_name)
        print('Dataset downloaded.')

        dataset = load_from_disk(save_path)
        self.process_pubmed(dataset=dataset, processed_path=processed_path)
        print('Done')

    def download(self, save_path, dataset_name='llamafactory/PubMedQA'):
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(save_path)

    def process_pubmed(self, dataset, processed_path, k=400):
        train = dataset['train']
        test = dataset['test']
        train_list, test_list = [], []
        root = processed_path

        os.makedirs(root, exist_ok=True)
        for row in tqdm(train):
            question = row['input']
            label = row['output']
            inputs = row['instruction']

            train_list.append({
                'ContextString': inputs, 
                'Behavior': question, 
                'answer': label
            })
        for row in tqdm(test):
            question = row['input']
            label = row['output']
            inputs = row['instruction']

            test_list.append({
                'ContextString': inputs, 
                'Behavior': question, 
                'answer': label
            })
        
        train_list = random.sample(train_list, k=k)
        test_list = random.sample(test_list, k=k)
        
        self.save_to_json(train_list, os.path.join(root, 'train_pubmed.json'))
        self.save_to_json(test_list, os.path.join(root, 'test_pubmed.json'))
        # save_to_csv(train_list, os.path.join(root, 'train_aqua.csv'))
        # save_to_csv(test_list, os.path.join(root, 'test_aqua.csv'))

    def save_to_json(self, data, save_path):
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)



if __name__ == '__main__':
    generator = CustomGenerator()
    save_path = '/home/jinsheng/OET/data'
    process_path = '/home/jinsheng/OET/data'
    generator.custom_generate(save_path=save_path, processed_path=process_path)
