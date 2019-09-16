import os
import json

import torch
from torch.utils.data import RandomSampler, DataLoader
from torch.nn import Softmax

from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from tqdm import tqdm, trange
from multi_model_engine.processing import DataFetcher


class TransformerModel:
    def __init__(self, model, tokenizer, device, rtn_seg_pos):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.rtn_seg_pos = rtn_seg_pos


    def train(self, X_train, y_train, val_set, nb_epoch,
              model_save_dir, batch_size, max_seq_len,
              learning_rate, adam_epsilon, warmup_steps):
        """Train model on dataset"""
        num_train_optim_steps = int(len(X_train) / batch_size) * nb_epoch
        optimizer, scheduler = self._setup_optim(learning_rate, adam_epsilon, warmup_steps, num_train_optim_steps)
        train_dataloader = self._setup_dataloader(X_train, y_train, max_seq_len,  batch_size, shuffle=True)

        self.model.train()
        for i in range(nb_epoch):
            for batch in tqdm(train_dataloader, desc="Iteration"):
                batch = {k: t.to(self.device) for k, t in batch.items()}
                outputs = self.model(**batch)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            chkpt_name = "chkpt epochs={0}".format(i + 1)
            self.save(model_save_dir, chkpt_name)

            ## Testing model
            results = {}
            for testset_name, label_sets in val_set.items():
                results[testset_name] = {}
                for label_name, testset  in label_sets.items():
                    results[testset_name][label_name] = self.test(testset["data"], testset["labels"])
            
            # Save test results
            with open(os.path.join(model_save_dir, chkpt_name, "test_accuracy.json"), 'w') as f:
                json.dump(results, f, indent=4)


    def test(self, data, labels, batch_size, max_seq_len):
        """Test model on a dataset"""
        test_dataloader = self._setup_dataloader(data, labels, max_seq_len, batch_size)
        accuracy = 0
        criterion = Softmax(dim=-1)
        for batch_num, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
            with torch.no_grad():
                labels = batch["labels"].to(self.device)
                batch = {k: t.to(self.device) for k, t in batch.items() if k != "labels"}
                outputs = self.model(**batch)
                logits = outputs[0]
                _, predictions = criterion(logits).max(-1)
                results = predictions == labels
                accuracy += results.sum().item()
        accuracy = accuracy / len(data) * 100
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

    def save(self, output_dir, save_model_name):
        save_path = os.path.join(output_dir, save_model_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def _convert_indices_to_sentiments(self, preds, converter):
        sentiments = []
        for label in preds:
            sentiments.append(converter[label])
        return sentiments


    def _setup_dataloader(self, data, labels, max_seq_len, batch_size, shuffle=False):
        data_fetcher = DataFetcher(data, self.tokenizer, max_seq_len, self.rtn_seg_pos, labels)
        dataloader = DataLoader(data_fetcher, shuffle=True, batch_size=batch_size)
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



