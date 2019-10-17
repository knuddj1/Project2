from torch.nn import Softmax
from torch.utils.data import DataLoader
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from op_text.processing import DataFetcher


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


def get_confidence_scores(logits):
    return Softmax(dim=-1)(logits)


def calculate_accuracy(logits, labels):
    _, pred_indices = get_confidence_scores(logits).max(-1)
    results = pred_indices == labels
    return results.sum().item()


class LabelConverter:
    """Utility class used to convert prediction indices to text labels"""
    def __init__(self, converter_dict):
        """
        Parameters:
            converter_dict : dictionary - Dictionary to convert indices to string labels
                                          
            EXAMPLE FORMAT:
            {
                0: "Negative",
                1: "Postive"
            }
        """
        self.converter = converter_dict

    def convert(self, indice):
        "Returns string label"
        return self.converter[indice]

    def convert_indices(self, indices):
        return [convert(indice) for indice in indices]

    def __len__(self):
        "Return length of label converter"
        return len(self.converter)
