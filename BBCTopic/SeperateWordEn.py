from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
import numpy as np
import gensim
import os 
import pickle

directory_path = os.path.dirname(os.path.realpath(os.getcwd()))
directory_path = os.path.join(directory_path, 'Data')

# load data
def get_data(path_folder_train_data):
  contents = []
  labels = []
  dirs = os.listdir(path_folder_train_data)
  for path in tqdm(dirs):
    file_paths = os.listdir(os.path.join(path_folder_train_data, path))
    for file_path in tqdm(file_paths):
      with open(os.path.join(path_folder_train_data, path, file_path), 'r') as f:
        lines = f.readlines()
        lines = ' '.join(lines)
        lines = gensim.utils.simple_preprocess(lines)
        lines = ' '.join(lines)
        lines = ViTokenizer.tokenize(lines)
        contents.append(lines)
        labels.append(path)

    return contents, labels

train_path = os.path.join(directory_path, r"D:\Workspace\DoAN\Data\BBC\Train")
X_data, y_data = get_data(train_path)

# write to pkl file
pickle.dump(X_data, open('X_data.pkl', 'wb'))
pickle.dump(y_data, open('y_data.pkl', 'wb'))
