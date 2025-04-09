
import random 
import csv 

class DataGenerator:

    def __init__(self, sample_size=400):
        self.sample_size = sample_size
    
    def custom_generate(self, data, save_root):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError
    
    def save_to_csv(self, data, file_name):
        keys = data[0].keys()

        with open(file_name, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=keys, escapechar=' ')
            writer.writeheader() 
            writer.writerows(data)
        



