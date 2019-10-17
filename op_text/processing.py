import csv
from torch import tensor
from torch.utils.data import Dataset

class DataProcessor:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, text):
        input_tokens = self.tokenizer.tokenize(text)[:self.max_seq_len - 2]
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


class DataFetcher(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, max_seq_len, rtn_seg_pos=True, labels=None):
        'Initialization'
        self.data = data
        self.data_processor = DataProcessor(tokenizer, max_seq_len)
        self.rtn_seg_pos = rtn_seg_pos
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        text = self.data[index]
        input_ids, segment_ids, input_mask, positional_ids = self.data_processor(text)

        sample = {}
        sample["input_ids"] = tensor(input_ids)
        sample["attention_mask"] = tensor(input_mask)
        if self.rtn_seg_pos:
            sample["token_type_ids"] = tensor(segment_ids)
            sample["position_ids"] = tensor(positional_ids)
        if self.labels:
            sample["labels"] = tensor(self.labels[index])

        return sample