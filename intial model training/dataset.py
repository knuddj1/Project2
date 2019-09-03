import csv
import torch
from torch.utils import data

def load_data(data_path):
    """
        Params:
            data_path -> Path to a dataset on disk. Dataset must be csv file and with format: input | label

        Returns:
            list of tuples in with two elements in format: index 0 = input, index 1 = label  
    """
    with open(data_path, 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader) # remove headers
        data = [(inp, int(label)  + 1) for inp, label in reader]
        return data

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_path, tokenizer, max_seq_len):
        'Initialization'
        self.data = load_data(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        input_text, label = self.data[index]
        input_tokens = self.tokenizer.tokenize(input_text)[:self.max_seq_len - 2]
        input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        
        # Pad input ids and mask to max seq length
        while len(input_ids) < self.max_seq_len:
            input_ids.append(0)
            input_mask.append(0)

        segment_ids = [0] * len(input_ids)

        return  (torch.tensor(input_ids),
                 torch.tensor(input_mask),
                 torch.tensor(segment_ids),
                 torch.tensor(label))