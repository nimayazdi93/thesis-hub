
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from matplotlib.patches import Rectangle
import scipy.io as sio
from FeatureExtractor import FeatureExtractor
from RetrieverSSA import RetrieverSSA


devkit_path = Path('data-set/cars/devkit')
train_path = Path('data-set/cars/cars_train/cars_train')
test_path = Path('data-set/cars/cars_test/cars_test')


cars_meta = sio.loadmat(devkit_path/'cars_meta.mat')
cars_annos=sio.loadmat(devkit_path/'cars_annos.mat')
cars_train_annos = sio.loadmat(devkit_path/'cars_train_annos.mat')
cars_test_annos = sio.loadmat(devkit_path/'cars_test_annos.mat')

labels = [c for c in cars_meta['class_names'][0]]
labels = pd.DataFrame(labels, columns=['labels'])
print(labels.head(10))

frame = [[i.flat[0] for i in line] for line in cars_train_annos['annotations'][0]]
columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
df_cars = pd.DataFrame(frame, columns=columns)
df_cars['class'] = df_cars['class']-1 # Python indexing starts on zero.
df_cars['fname'] = [train_path/f for f in df_cars['fname']] #  Appending Path
print(df_cars.head())
print(len(df_cars))
df_cars = df_cars.merge(labels, left_on='class', right_index=True)
df_cars = df_cars.sort_index()
print(df_cars.head())





all_results=sio.loadmat('cached-data/all_results.mat')
frame = [[i.flat[0] for i in line] for line in all_results['all_results'][0]]
columns = ['input', 'class', 'results']

print(all_results)