
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from PIL import Image
from pathlib import Path 
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
df_cars['fname'] = [f for f in df_cars['fname']] #  Appending Path
print(df_cars.head())
print(len(df_cars))
df_cars = df_cars.merge(labels, left_on='class', right_index=True)
df_cars = df_cars.sort_index()
print(df_cars.head())
  
with open('cached-data/all_results.json', 'r') as openfile: 
    all_results = json.load(openfile)

new_records=[]
for res in all_results:
    new_record={
        'input':res['input'],
        'class':res['class'],
        'output':[]
    }
    output=res['results']
    real_label=res['class']
    for out in output:
        out_label=df_cars[df_cars['fname']==out]['labels']
        if(len(out_label.values)==0):
            new_record['output'].append({'out_name':out, 'out_class':''})
        else:
            new_record['output'].append({'out_name':out,'out_class':str(out_label.values[0])})
    new_records.append(new_record)

json_object = json.dumps(new_records, indent=4)
with open("cached-data/all_new_results.json", "w") as outfile:
    outfile.write(json_object)
