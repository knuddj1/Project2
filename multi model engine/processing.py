import csv
import torch
from torch.utils import data
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.tokenization_roberta import RobertaTokenizer

class DataProcessor:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(data):
        input_tokens = self.tokenizer.tokenize(input_text)[:self.max_seq_len - 2]
        input_tokens = [self.tokenizer.cls_token] + input_tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        
        # Pad input ids and mask to max seq length  
        while len(input_ids) < self.max_seq_len:
            input_ids.append(0)
            input_mask.append(0)

        segment_ids = [0] * len(input_ids)
        positional_ids = list(range(len(input_ids)))

        return input_ids, segment_ids, input_mask, positional_ids


class DataFetcher(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, max_seq_len, labels=None):
        'Initialization'
        self.data = data
        self.data_processor = DataProcessor(tokenizer, max_seq_len)
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        sample = self.data[index]
        input_ids, segment_ids, input_mask, positional_ids = self.data_processor(sample)
        label = None
        if self.labels:
            label = self.labels[index]

        return  (torch.tensor(input_ids),
                 torch.tensor(input_mask),
                 torch.tensor(segment_ids),
                 torch.tensor(positional_ids),
                 torch.tensor(label))