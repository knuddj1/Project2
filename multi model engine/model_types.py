from torch.utils.data import RandomSampler, DataLoader
from pytorch_transformers.modeling_bert import BertForSequenceClassification, BertConfig
from pytorch_transformers.tokenization_bert import BertTokenizer
from processing import DataFetcher

class BERT

    DOWNLOADABLE_MODELS = {
        'bert-base-uncased',
        'bert-large-uncased',
        'bert-base-cased',
        'bert-large-cased',
        'bert-large-uncased-whole-word-masking',
        'bert-large-cased-whole-word-masking'
    }


    def __init__(self, bert_model, num_labels=3):
        config = bert_model
        if bert_model in BERT.DOWNLOADABLE_MODELS:
            config = BertConfig.from_pretrained(BERT.DOWNLOADABLE_MODELS[bert_model])
            config.num_labels = num_labels
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.model = BertForSequenceClassification(config)


    def train(self, data, labels, output_dir, device, batch_size=32, max_seq_len=128,
              n_epochs=3, learning_rate=3e-5, adam_epsilon=1e-8, warmup_steps=0):

        self.model.to(device)

        data_fetcher = DataFetcher(data, self.tokenizer, max_seq_len, labels)
        num_train_optimization_steps = int(len(train_dataset) / batch_size) * n_epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

        sampler = RandomSampler(data_fetcher)    
        dataloader = DataLoader(data_fetcher, sampler=sampler, batch_size=batch_size)

        self.model.train()
        for _ in trange(n_epochs, desc="Epoch"):
            for batch in tqdm(train_dataloader, desc="Iteration"):
                batch = (t.to(device) for t in batch if t != None)
                input_ids, segment_ids, input_masks, positional_ids, labels = batch
                outputs = self.model(input_ids,
                                     token_type_ids=segment_ids,
                                     attention_mask=input_masks,
                                     labels=labels,
                                     position_ids=positional_ids)
                loss = outputs[0]
                loss.backward()
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)



