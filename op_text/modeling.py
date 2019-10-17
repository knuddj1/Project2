import os
import json
import torch

from pytorch_transformers.modeling_bert import BertConfig, BertForSequenceClassification
from pytorch_transformers.tokenization_bert import BertTokenizer

from pytorch_transformers.modeling_distilbert import DistilBertConfig, DistilBertForSequenceClassification
from pytorch_transformers.tokenization_distilbert import DistilBertTokenizer

from pytorch_transformers.modeling_roberta import RobertaConfig, RobertaForSequenceClassification
from pytorch_transformers.tokenization_roberta import RobertaTokenizer

from tqdm import tqdm, trange
from op_text.utils import setup_dataloader, setup_optim, get_confidence_scores, calculate_accuracy, LabelConverter

class TransformerModel:
    def __init__(self, model_path, config_cls, tokenizer_cls, model_cls, downloadables, rtn_seg_pos, num_labels):
        assert model_path in downloadables or os.path.isdir(model_path), "model_path must be from either one of {0} or a path to the directory of a local model".format(downloadables)
        self.config = config_cls(model_path)
        if model_path in downloadables:
            self.config.num_labels = num_labels
        self.tokenizer = tokenizer_cls.from_pretrained(model_path)
        self.model = model_cls(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model.to(self.device)
        self.rtn_seg_pos = rtn_seg_pos


    def fit(self, X_train, y_train, validation_split=None,
            chkpt_model_every=None, model_save_dir=None, nb_epoch=1, batch_size=32,
            max_seq_len=128, learning_rate=3e-5, adam_epsilon=1e-8,
            warmup_steps=0, gradient_accumulation_steps=1):
        """Train model on dataset"""
        
        if chkpt_model_every:
            assert type(chkpt_model_every), "@Param: 'chkpt_model_every' must be an integer"
            assert type(chkpt_model_every), "@Param: 'chkpt_model_every' must be greater than zero"
            assert os.path.isdir(model_save_dir), (f"@Param: 'model_save_dir' == {model_save_dir}."
                                                    " Directory does not exist"
                                                    " Must supply an existing directory if @Param: "
                                                    "'chkpt_model_every' is used")

        validation_dataloader=None
        if validation_split:
            split_percent = int(len(X_train) * validation_split)
            X_train = X_train[:split_percent]
            y_train = y_train[:split_percent]
            X_val = X_train[split_percent:]
            y_val = y_train[split_percent:]

        num_train_optim_steps = int(len(X_train) / batch_size) * nb_epoch
        optimizer, scheduler = setup_optim(learning_rate, adam_epsilon, warmup_steps, num_train_optim_steps)
        train_dataloader = setup_dataloader(X_train, y_train, max_seq_len,  batch_size, shuffle=True)


        self.model.zero_grad()
        self.model.train()
        for i in range(nb_epoch):
            step = 0
            train_accuracy = 0
            for batch in tqdm(train_dataloader, desc="Iteration"):
                batch = {k: t.to(self.device) for k, t in batch.items()}
                outputs = self.model(**batch)
                loss, logits = outputs[:2]
                loss.backward()
                train_accuracy += calculate_accuracy(logits, batch["labels"])

                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                step += 1

                batch = {k: t.detach().cpu() for k, t in batch.items()}
                del batch
                torch.cuda.empty_cache()
            
            validation_accuracy = None
            if validation_dataloader:
                validation_accuracy = self.evaluate(X_val, y_val, batch_size, max_seq_len)
            
            if chkpt_model_every:
                results = {
                    "train accuracy": train_accuracy / len(train_dataloader),
                    "validation_accuracy": validation_accuracy
                }
                chkpt_name = "chkpt epochs={0}".format(i + 1)
                self.save(model_save_dir, chkpt_name, results)

            
    def evaluate(self, data, labels, batch_size=32, max_seq_len=128):
        """Test model on a dataset"""
        test_dataloader = setup_dataloader(data, labels, max_seq_len, batch_size)
        accuracy = 0
        
        for batch in tqdm(test_dataloader, desc="Iteration"):
            with torch.no_grad():
                labels = batch["labels"]
                batch = {k: t.to(self.device) for k, t in batch.items() if k != "labels"}
                outputs = self.model(**batch)
                logits = outputs[0]
                accuracy += calculate_accuracy(logits, labels)
            
            batch = {k: t.detach().cpu() for k, t in batch.items()}
            del batch
            torch.cuda.empty_cache()

        accuracy = accuracy / len(test_dataloader)
        return accuracy


    def predict(self, data, label_converter=None, batch_size=32, max_seq_len=128):
        """Test model on a dataset"""
        if label_converter:
            assert type(label_converter) == LabelConverter, "@Param: 'label_converter' must be of Type: LabelConverter"
            assert len(label_converter) == self.config.num_labels, (f"@Param: 'label_coverter has length of {len(label_converter)}."
                                                                    f"Must have same length as config.num_labels: {self.config.num_labels}")

        predictions_dataloader = setup_dataloader(data, None, max_seq_len, batch_size)
        predictions = []
        for batch in tqdm(predictions_dataloader, desc="Iteration"):
            with torch.no_grad():
                batch = {k: t.to(self.device) for k, t in batch.items()}
                outputs = self.model(**batch)
                logits = outputs[0]
                confidence_scores = get_confidence_scores(logits)
                pred_indices = confidence_scores.max(-1)[-1].tolist()
                preds = (confidence_scores.tolist(), pred_indices)
                if label_converter:
                    preds =  preds + (label_converter.convert_indices(pred_indices),)
                predictions += list(zip(*preds))
                batch = {k: t.detach().cpu() for k, t in batch.items()}
                del batch
                torch.cuda.empty_cache()
        return predictions


    def save(self, output_dir, save_model_name, save_results=None):
        assert os.path.isdir(output_dir), (f"@Param 'output_dir' == {output_dir}." 
                                           " Directory does not exist.")

        save_path = os.path.join(output_dir, save_model_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        if save_results:
            fp = os.path.join(output_dir, save_model_name, "test_accuracy.json")
            json.dump(save_results, open(fp, 'w', encoding='utf-8'), indent=4)


class Bert(TransformerModel):
    DOWNLOADABLES = ['bert-base-uncased', 'bert-large-uncased']
    def __init__(self, model_path, num_labels=2):
        super(TransformerModel, self).__init__(model_path,
                                               BertConfig, 
                                               BertTokenizer, 
                                               BertForSequenceClassification, 
                                               Bert.DOWNLOADABLES, 
                                               True, 
                                               num_labels)


class Roberta(TransformerModel):
    DOWNLOADABLES = ['roberta-base', 'roberta-large']
    def __init__(self, model_path, num_labels=2):
        super(TransformerModel, self).__init__(model_path,
                                               RobertaConfig, 
                                               RobertaTokenizer, 
                                               RobertaForSequenceClassification, 
                                               Roberta.DOWNLOADABLES, 
                                               True, 
                                               num_labels)


class DistilBert(TransformerModel):
    DOWNLOADABLES = ['distilbert-base-uncased']
    def __init__(self, model_path, num_labels=2):
        super(TransformerModel, self).__init__(model_path,
                                               DistilBertConfig, 
                                               DistilBertTokenizer, 
                                               DistilBertForSequenceClassification, 
                                               DistilBert.DOWNLOADABLES, 
                                               False, 
                                               num_labels)
    



