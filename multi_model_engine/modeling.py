import os

import torch
from torch.utils.data import RandomSampler, DataLoader
from torch.nn import Softmax

from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from tqdm import tqdm, trange
from multi_model_engine.processing import DataFetcher


class TransformerModel:
    def __init__(self, model, tokenizer, device, num_labels, rtn_seg_pos):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.num_labels = num_labels
        self.rtn_seg_pos = rtn_seg_pos


    def train(self, data, labels, output_dir, batch_size, max_seq_len,
              n_epochs, learning_rate, adam_epsilon, warmup_steps):
        """Train model on dataset"""
        num_train_optim_steps = int(len(data) / batch_size) * n_epochs
        optimizer, scheduler = self._setup_optim(learning_rate, adam_epsilon, warmup_steps, num_train_optim_steps)
        train_dataloader = self._setup_dataloader(data, labels, max_seq_len,  batch_size, shuffle=True, sampler=True)  

        self.model.train()
        for _ in trange(n_epochs, desc="Epoch"):
            for batch in tqdm(train_dataloader, desc="Iteration"):
                batch = {k: t.to(self.device) for k, t in batch.items()}
                outputs = self.model(**batch)
                loss = outputs[0]
                loss.backward()
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


    def test(self, data, labels, batch_size, max_seq_len):
        """Test model on a dataset"""
        test_dataloader = self._setup_dataloader(data, labels, max_seq_len, batch_size)
        accuracy = 0
        criterion = Softmax(dim=-1)
        for batch_num, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
            with torch.no_grad():
                batch = {k: t.to(self.device) for k, t in batch.items()}
                outputs = self.model(**batch)
                logits = outputs[0]
                _, predictions = criterion(logits).max(-1)
                results = predictions == labels
                accuracy += results.sum().item()
        accuracy = accuracy / len(data)
        return accuracy


    def predict(self, data, rtn_text_labels, label_converter, batch_size, max_seq_len):
        """Test model on a dataset"""
        predictions_dataloader = self._setup_dataloader(data, None, max_seq_len, batch_size)
        predictions = []
        criterion = Softmax(dim=-1)
        for batch_num, batch in enumerate(tqdm(predictions_dataloader, desc="Iteration")):
            with torch.no_grad():
                batch = {k: t.to(self.device) for k, t in batch.items()}
                outputs = self.model(**batch)
                logits = outputs[0]
                confidence_scores = criterion(logits)
                indices = confidence_scores.max(-1)[-1].tolist()
                
                results = (confidence_scores.tolist(), indices)
                if rtn_text_labels:
                    text_labels = self._convert_indices_to_sentiments(indices, label_converter)
                    results = results + (text_labels,)
                predictions += list(zip(*results))
        return predictions


    def _convert_indices_to_sentiments(self, preds):
        sentiments = []
        for label in preds:
            sentiments.append(self.label_converter[label])
        return sentiments


    def _setup_dataloader(self, data, labels, max_seq_len, batch_size, sampler=None, shuffle=False):
        data_fetcher = DataFetcher(data, self.tokenizer, max_seq_len, self.rtn_seg_pos, labels)
        if sampler:
            sampler = RandomSampler(data_fetcher)
        dataloader = DataLoader(data_fetcher, shuffle=True, sampler=sampler, batch_size=batch_size)
        return dataloader


    def _setup_optim(self, learning_rate, adam_epsilon, warmup_steps, num_train_optim_steps):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optim_steps)
        return optimizer, scheduler



