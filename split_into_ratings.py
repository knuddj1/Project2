import os
import pandas as pd
import json
import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

columns = ["reviewText", "overall"]
score_vals = [1.0, 2.0, 3.0, 4.0, 5.0]

cwd = os.getcwd()
data_dir = os.path.join(cwd, "amazon review data")

split_dir = os.path.join(cwd, "amazon review data rating split")
if os.path.exists(split_dir) is False:
    os.mkdir(split_dir)

num_samples_per_rating_per_file = {}

for filename in os.listdir(data_dir)[:2]:
    print("Loading '{0}' into dataframe".format(filename))
    df = getDF(os.path.join(data_dir, filename))[columns]

    category_name = filename.replace('reviews_', '').replace(".json.gz", '')
    save_dir = os.path.join(split_dir, category_name)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    print("Saving split files into directory '{0}'".format(save_dir))
    num_samples_per_rating_per_file[category_name] = {}
    for val in score_vals:
        print("\t ==> reviews with '{0}' label currently being worked on".format(val))
        q = df.loc[df['overall'] == val]
        save_path = os.path.join(save_dir, '{0}.csv'.format(val))
        q.to_csv(save_path, header=["text", "label"], index=False)
        num_samples_per_rating_per_file[category_name][val] = len(q)

    print("File '{0}' successfully split into ratings".format(filename))
    print()

json_out_filepath = os.path.join(cwd, "samples_per_rating.json")
with open(json_out_filepath, 'w') as json_file:
    json.dump(num_samples_per_rating_per_file, json_file, indent=4)

