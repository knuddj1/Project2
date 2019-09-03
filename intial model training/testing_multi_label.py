import os
import torch
import csv
from pytorch_transformers.modeling_bert import BertForSequenceClassification
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.utils import data
from torch.nn import Softmax
from dataset import Dataset

def main():
    batch_size = 16
    max_seq_len = 128
    model_dir = 'fine_tuned--bert-base-uncased--SEQ_LEN=128--BATCH_SIZE=32--HEAD=1'
    output_filename = os.path.join(model_dir, "fine-tuned-sent-classifer-test-results.csv")
    test_sets_dir = "dataset\custom_test_set"
    test_files = [filename for filename in os.listdir(test_sets_dir)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    criterion = Softmax()

    accuracies = {}

    for filename in test_files:
        print("Testing on dataset: {}".format(filename))
        file_path = os.path.join(test_sets_dir, filename)
        test_dataset = Dataset(file_path, tokenizer, max_seq_len)
        test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size)
        accuracy = 0
        for batch in test_dataloader:
            with torch.no_grad():
                batch = (t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, labels = batch
                outputs = model(input_ids, input_mask, segment_ids)
                logits = outputs[0]
                _, predictions = criterion(logits).max(-1)
                results = predictions == labels
                accuracy += results.sum().item()

        accuracy = accuracy / len(test_dataset)
        print("Model achieved {}'%' accuracy".format(accuracy))
        dataset_name = filename.split('.')[0]
        accuracies[dataset_name] = accuracy

    with open(output_filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=accuracies.keys())
        writer.writeheader()
        writer.writerow(accuracies)

    
if __name__ == '__main__':
    main()