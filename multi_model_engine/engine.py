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
        assert model_name in MODEL_NAMES, "model_name must be one of {0}".format(MODEL_NAMES)

        config, tokenizer, model, downloadables, rtn_seg_pos = MODEL_NAMES[model_name]

        assert model_path in downloadables or os.path.isdir(model_path), "model_path must be from either one of {0} or a path to the directory of a local model".format(downloadables)

        model_config = config.from_pretrained(model_path)
        model_config.num_labels = num_labels
        tokenizer = tokenizer(model_path)
        model = model(model_config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        self.model = TransformerModel(model, tokenizer, device, num_labels, rtn_seg_pos)


    def train(self, data, labels, output_dir, batch_size=32, max_seq_len=128,
              n_epochs=5, learning_rate=3e-5, adam_epsilon=1e-8, warmup_steps=0):
        assert os.path.isdir(output_dir), "output_dir must be an existing directory"
        self.model.train(data, labels, output_dir, batch_size, max_seq_len,
                         n_epochs, learning_rate, adam_epsilon, warmup_steps)

    def test(self, data, labels, output_dir, output_filename="results", batch_size=32, max_seq_len=128):
        assert os.path.isdir(output_dir), "output_dir must be an existing directory"
        self.model.test(data, labels, output_dir, output_filename, batch_size, max_seq_len)

    def predict(self, data, rtn_text_labels=False, label_converter=None, batch_size=32, max_seq_len=128):
        if rtn_text_labels:
            assert label_converter != None, "label_converter must be supplied if rtn_text_labels is set to True."
            assert len(label_converter) == self.num_labels, "label_converter has: {0} labels. Must have the same amount of labels as self.num_labels == {1}.".format(label_converter, self.model.num_labels)
        self.model.predict(data, rtn_text_labels, label_converter, batch_size, max_seq_len)