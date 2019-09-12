import argparse
import os
import csv
import json
from multi_model_engine.engine import SentimentEngine


def get_data(data_path, split_percent):
    train_data = []
    train_labels = []
    test_sets = {}

    for root, dirs, files in os.walk(data_path, topdown=False):
        dataset_name = root.split("\\")[-1]
        if len(files) > 0: test_sets[dataset_name] = {}
        for name in files:
            temp_data = []
            temp_labels = []
            with open(os.path.join(root, name)) as csvfile:
                reader = csv.reader(csvfile)
                next(reader) # remove header
                for line in reader:
                    text, label = line
                    temp_data.append(text) 
                    temp_labels.append(int(float(label))-1)

            train_test_split = int(len(temp_data) * split_percent)

            train_data += temp_data[:train_test_split]
            train_labels += temp_labels[:train_test_split]

            test_sets[dataset_name][name] = {
                "data" : temp_data[train_test_split:],
                "labels" : temp_labels[train_test_split:]
            }

    return train_data, train_labels, test_sets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, required=True)
    parser.add_argument('-model_path', type=str, required=True)
    parser.add_argument('-data_path', type=str, required=True)
    parser.add_argument('-output_dir', type=str, default=os.getcwd())
    parser.add_argument('-save_model_name', type=str, required=True)
    parser.add_argument('-nb_epoch', type=int, default=5)
    parser.add_argument('-split_percent', type=float, default=0.8)
    args = parser.parse_args()
    
    assert os.path.isdir(args.output_dir), "output_dir must exist!"
    
    model_save_dir = os.path.join(args.output_dir, args.save_model_name)
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)

    train_data, train_labels, test_sets = get_data(args.data_path, args.split_percent)
    
    clsf = SentimentEngine(args.model_name, args.model_path, num_labels=5)

    for i in range(args.nb_epoch):
        # Training model
        clsf.train(train_data, train_labels)

        chkpt_name = "chkpt epochs={0}".format(i)
        clsf.save(model_save_dir, chkpt_name)

        ## Testing model
        results = {}
        for testset_name, label_sets in test_sets.items():
            results[testset_name] = {}
            for label_name, testset  in label_sets.items():
                results[testset_name][label_name] = clsf.test(testset["data"], testset["labels"])
        
        # Save test results
        with open(os.path.join(model_save_dir, chkpt_name, "test_accuracy.json"), 'w') as f:
            json.dump(results, f, indent=4)

            
if __name__ == '__main__':
    main()
    