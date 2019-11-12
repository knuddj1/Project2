import matplotlib.pyplot as plt
import os
import json
import re


def trim_model_name(s):
    is_five_labeled = False
    if "finetuned" in s:
        is_five_labeled = True
    to_replace = ["-base", "-uncased", "finetuned-"]
    pattern = re.compile("|".join(to_replace))
    s = pattern.sub("", s)
    s = s.replace("--", "-")
    return s, is_five_labeled

def plot_results():
    base_dir = "fine tuned models test results"
    model_names = [name for name in os.listdir(base_dir)]

    dataset_results = {}

    for model_name in model_names:
        model_results_dir = os.path.join(base_dir, model_name)
        model_name, is_five_labeled = trim_model_name(model_name)

        for i, epoch_results in enumerate(os.listdir(model_results_dir)):
            epoch_results_fp = os.path.join(model_results_dir, epoch_results)
            epoch_results = json.load(open(epoch_results_fp, 'r', encoding='utf-8'))["test_sets"]

            for dataset_name, value in epoch_results.items():
                dataset_name = dataset_name.replace(".csv", "")
            
                if dataset_name not in dataset_results:
                    dataset_results[dataset_name] = {}
                
                if model_name not in dataset_results[dataset_name]:
                    dataset_results[dataset_name][model_name] = {}
                
                if is_five_labeled:
                    value = value * 100

                dataset_results[dataset_name][model_name][i + 1] = value / 100


    fig, axes = plt.subplots(3, 2, figsize=(20,20))
    for idx, (dataset_name, model_results) in enumerate(dataset_results.items()):
        axes[idx % 3, idx % 2].set_title(dataset_name)
        axes[idx % 3, idx % 2].set_ylim(0, 1.0)
        for model_name, test_results in model_results.items():
            axes[idx % 3, idx % 2].plot(list(range(1, len(test_results.values()) + 1)), list(test_results.values()), '.-')

    labels = list(dataset_results["hand made"].keys())
    fig.legend(labels, ncol=len(labels) // 2, loc='upper center', 
            columnspacing=2.0, labelspacing=1.0,
            handletextpad=1.0, handlelength=2.5,
            # bbox_to_anchor=(0.5, -0.05),
            prop={'size': 10})

    # fig.tight_layout()
    plt.show()
