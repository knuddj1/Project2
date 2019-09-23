import os
import csv
import torch
from random import shuffle, sample
from multi_model_engine.engine import SentimentEngine


def load_data(fp):
    with open(fp, 'r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        return [(inp, int(label)  + 1) for inp, label in reader]


def load_trainsets(fp, ss):
    train_data = load_data(fp)
    train_divided = {0:[], 1:[], 2:[]}
    for d, l in train_data:
        train_divided[l].append((d,l))
    
    sampled_sets = {}
    for s in ss:
        total_per_label = int(s / 3)
        sampled_sets[s] = {"data": [], "labels": []}
        for l, ls in train_divided.items():
            data, labels = list(zip(*sample(ls, total_per_label)))
            sampled_sets[s]["data"] += data
            sampled_sets[s]["labels"] += labels
    return sampled_sets


def load_testsets(fps):
    test_sets = {"test_sets": {}}
    for fp, name in fps:
        d, l = zip(*load_data(fp))
        test_sets["test_sets"][name] = {
            "data" : d,
            "labels" : l
        }
    return test_sets


def main():
    models=[("bert", "bert-base-uncased"), ("roberta", "roberta-base"), ("distilbert", "distilbert-base-uncased")]
    output_dir=os.path.join("F:/", "small_models")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    nb_epoch=5
    num_labels=3
    sample_sizes=[100, 1000, 10000]

    train_fp = "intial model training/dataset/custom_training_set.csv"
    test_fps = [(os.path.join("intial model training/dataset/custom_test_set", n), n) for n in os.listdir("intial model training/dataset/custom_test_set")]

    train_data = load_trainsets(train_fp, sample_sizes)
    test_sets = load_testsets(test_fps)
    
    for sample_size in sample_sizes:
        for model_name, model_type in models:
            model_save_name = f"{model_type}--samples={sample_size}--labels={num_labels}"

            model_save_dir = os.path.join(output_dir, model_save_name)
            if not os.path.isdir(model_save_dir):
                os.mkdir(model_save_dir)

            clsf = SentimentEngine(model_name, model_type, num_labels=num_labels)
            # Training model
            clsf.train(train_data[sample_size]["data"], train_data[sample_size]["labels"], test_sets, model_save_dir)

            del clsf
            torch.cuda.empty_cache()
        

            
if __name__ == '__main__':
    main()
    