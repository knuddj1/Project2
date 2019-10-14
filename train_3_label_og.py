import argparse
import os
import csv
import json
from random import shuffle
from multi_model_engine.engine import SentimentEngine


def load_data(data_path):
    """
        Params:
            data_path -> Path to a dataset on disk. Dataset must be csv file and with format: input | label

        Returns:
            list of tuples in with two elements in format: index 0 = input, index 1 = label  
    """
    with open(data_path, 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader) # remove headers
        data = [(inp, int(label)  + 1) for inp, label in reader]
        return list(zip(*data))

def get_data(data_dir):
    train_data, train_labels = load_data(os.path.join(data_dir, "custom_training_set.csv"))

    test_sets_dir_name = "custom_test_set"
    test_sets_dir = os.path.join(data_dir, test_sets_dir_name)
    test_sets = {test_sets_dir_name : {}}

    for test_set in os.listdir(test_sets_dir):
        fp = os.path.join(test_sets_dir, test_set)
        test_data, test_labels = load_data(fp)
        test_sets[test_sets_dir_name][test_set] = {
            "data" : test_data,
            "labels" : test_labels
        }

    combined = list(zip(train_data, train_labels))
    for _ in range(10):
        shuffle(combined)
    train_data[:], train_labels[:] = zip(*combined)
    return train_data, train_labels, test_sets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, required=True)
    parser.add_argument('-model_path', type=str, required=True)
    parser.add_argument('-output_dir', type=str, default=os.getcwd())
    parser.add_argument('-save_model_name', type=str, required=True)
    parser.add_argument('-nb_epoch', type=int, default=5)
    args = parser.parse_args()
    
    assert os.path.isdir(args.output_dir), "output_dir must exist!"
    
    model_save_dir = os.path.join(args.output_dir, args.save_model_name)
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)

    train_data, train_labels, test_sets = get_data("intial model training/dataset")
    
    clsf = SentimentEngine(args.model_name, args.model_path, num_labels=3)
    # Training model
    clsf.train(train_data, train_labels, test_sets, model_save_dir)
        
            
if __name__ == '__main__':
    main()
    