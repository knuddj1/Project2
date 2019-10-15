import os
import torch
from multi_model_engine.modeling import TransformerModel


class SentimentEngine:
    def __init__(self, model_path, num_labels=2):
        pass


    def train(self, X_train, y_train, val_set, model_save_dir,
              nb_epoch=5, batch_size=32, max_seq_len=128,
              learning_rate=3e-5, adam_epsilon=1e-8, warmup_steps=0, gradient_accumulation_steps=1):
            self.model.train(X_train, y_train, val_set,
                             nb_epoch, model_save_dir, batch_size, max_seq_len,
                             learning_rate, adam_epsilon, warmup_steps, gradient_accumulation_steps)
            

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