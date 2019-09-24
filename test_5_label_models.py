import os
import csv
import torch
import json
from random import shuffle, sample
from multi_model_engine.engine import SentimentEngine


def load_data(fp):
    with open(fp, 'r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        return [(inp, int(label)  + 1) for inp, label in reader]

def load_testsets(fps):
    test_sets = {"test_sets": {}}
    for fp, name in fps:
        d, l = zip(*load_data(fp))
        test_sets["test_sets"][name] = {
            "data" : d,
            "labels" : l
        }
    return test_sets


def substitute_label(l):
    if l == 0 or l == 1:
        return 0
    elif l == 2:
        return 1
    else:
        return 2  

def main():
    models=[
        ("bert", r"F:\fine_tuned_models\finetuned-bert-uncased-labels-5", "finetuned-bert-uncased-labels-5"),
        ("roberta", r"F:\fine_tuned_models\finetuned-roberta-base-labels-5", "finetuned-roberta-base-labels-5"), 
        ("distilbert", r"F:\fine_tuned_models\finetuned-distilbert-base-uncased-labels-5", "finetuned-distilbert-base-uncased-labels-5")
    ]
    test_fps = [(os.path.join("intial model training/dataset/custom_test_set", n), n) for n in os.listdir("intial model training/dataset/custom_test_set")]
    test_sets = load_testsets(test_fps)
    
    for model_name, model_type, model_save in models:
        clsf = SentimentEngine(model_name, model_type)
        
        results = {}
        for testset_name, label_sets in test_sets.items():
            results[testset_name] = {}
            for label_name, testset  in label_sets.items():
                preds = clsf.predict(testset["data"])
                labels = testset["labels"]
                indices = [substitute_label(x[-1]) for x in preds]
                acc = 0
                for pred, truth in zip(indices, labels):
                    acc += 1 if pred == truth else 0
                results[testset_name][label_name] = acc / len(indices)


        # Save test results
        with open(os.path.join(os.getcwd(), model_save + "_test_accuracy.json"), 'w') as f:
            json.dump(results, f, indent=4)

        del clsf
        torch.cuda.empty_cache()
    

            
if __name__ == '__main__':
    main()
    