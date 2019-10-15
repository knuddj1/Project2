from torch.nn import Softmax
from torch.utils.data import DataLoader
from multi_model_engine.processing import DataFetcher

def convert_indices_to_sentiments(preds, converter):
    sentiments = []
    for label in preds:
        sentiments.append(converter[label])
    return sentiments


def setup_dataloader(data, labels, tokenizer, rtn_seg_pos, max_seq_len, batch_size, shuffle=False):
    data_fetcher = DataFetcher(data, tokenizer, max_seq_len, rtn_seg_pos, labels)
    dataloader = DataLoader(data_fetcher, shuffle=shuffle, batch_size=batch_size)
    return dataloader


def setup_optim(named_params, learning_rate, adam_epsilon, warmup_steps, num_train_optim_steps):
    param_optimizer = list(named_params) # model.named_parameters()
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optim_steps)
    return optimizer, scheduler

def calculate_accuracy(logits, labels):
    criterion = Softmax(dim=-1)
    _, predictions = criterion(logits).max(-1)
    results = predictions == labels
    return results.sum().item()