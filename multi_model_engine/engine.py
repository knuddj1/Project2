import os
import torch
from multi_model_engine.modeling import TransformerModel

from pytorch_transformers.modeling_bert import BertConfig, BertForSequenceClassification
from pytorch_transformers.tokenization_bert import BertTokenizer

from pytorch_transformers.modeling_distilbert import DistilBertConfig, DistilBertForSequenceClassification
from pytorch_transformers.tokenization_distilbert import DistilBertTokenizer

from pytorch_transformers.modeling_roberta import RobertaConfig, RobertaForSequenceClassification
from pytorch_transformers.tokenization_roberta import RobertaTokenizer

MODEL_NAMES = {
        "bert" : (BertConfig, BertTokenizer, BertForSequenceClassification, {'bert-base-uncased', 'bert-large-uncased'}, True),
        "distilbert" : (DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification, {'distilbert-base-uncased'}, False),
        "roberta" : (RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, {'roberta-base', "roberta-large"}, True)
    }

class SentimentEngine:
    def __init__(self, model_name, model_path, num_labels=2):
        assert num_labels > 1, "num_labels must be greater than 1!"
        assert model_name in MODEL_NAMES, "model_name must be one of {0}".format(MODEL_NAMES.keys())

        config, tokenizer_cls, model_cls, downloadables, rtn_seg_pos = MODEL_NAMES[model_name]

        assert model_path in downloadables or os.path.isdir(model_path), "model_path must be from either one of {0} or a path to the directory of a local model".format(downloadables)

        model_config = config.from_pretrained(model_path)
        if model_path in downloadables:
            model_config.num_labels = num_labels
        else:
            num_labels = model_config.num_labels
        tokenizer = tokenizer_cls.from_pretrained(model_path)
        model = model_cls(model_config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        self.num_labels = num_labels
        self.model = TransformerModel(model, tokenizer, device, rtn_seg_pos)


    def train(self, X_train, y_train, val_set, model_save_dir,
              nb_epoch=5, batch_size=32, max_seq_len=128,
              learning_rate=3e-5, adam_epsilon=1e-8, warmup_steps=0):
            self.model.train(X_train, y_train, val_set,
                             nb_epoch, model_save_dir, batch_size, max_seq_len,
                             learning_rate, adam_epsilon, warmup_steps)
            
            

    def test(self, data, labels, batch_size=32, max_seq_len=128):
        return self.model.test(data, labels, batch_size, max_seq_len)


    def predict(self, data, rtn_text_labels=False, label_converter=None, batch_size=32, max_seq_len=128):
        if rtn_text_labels:
            assert label_converter != None, "label_converter must be supplied if rtn_text_labels is set to True."
            assert len(label_converter) == self.num_labels, "label_converter has: {0} labels. Must have the same amount of labels as self.num_labels == {1}.".format(label_converter, self.model.num_labels)
        return self.model.predict(data, rtn_text_labels, label_converter, batch_size, max_seq_len)

    def save(self, output_dir, save_model_name):
        assert os.path.isdir(output_dir), "output_dir must be an existing directory"
        self.model.save(output_dir, save_model_name)