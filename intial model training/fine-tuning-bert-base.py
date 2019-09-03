import os
import csv
import torch
from pytorch_transformers.modeling_bert import BertForSequenceClassification, BertConfig
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from torch.utils import data
from tqdm import tqdm, trange
from dataset import Dataset


def main():
    batch_size = 32
    max_seq_len = 128
    n_epochs = 3
    bert_model = 'bert-base-uncased'
    learning_rate = 3e-5
    adam_epsilon = 1e-8
    warmup_steps=0
    num_labels = 1
    output_dir = "fine_tuned--{0}--SEQ_LEN={1}--BATCH_SIZE={2}--HEAD={3}".format(bert_model, max_seq_len, batch_size, num_labels)
    dataset_dir = "dataset\custom_training_set.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = BertConfig.from_pretrained(bert_model)
    config.num_labels = num_labels
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model = BertForSequenceClassification(config)
    model.to(device)

    train_dataset = Dataset(dataset_dir, tokenizer, max_seq_len)
    num_train_optimization_steps = int(len(train_dataset) / batch_size) * n_epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

    train_sampler = data.RandomSampler(train_dataset)    
    train_dataloader = data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    model.train()
    for _ in trange(n_epochs, desc="Epoch"):
        for batch in tqdm(train_dataloader, desc="Iteration"):
            batch = (t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, labels = batch
            outputs = model(input_ids, input_mask, segment_ids, labels)
            loss = outputs[0]
            loss.backward()
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    


if __name__ == '__main__':
    main()