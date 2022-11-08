import os
from time import time
from cv2 import sqrt 
import pandas as pd
import json 
from pathlib import Path 
import scipy.io as sio
from FeatureExtractor import FeatureExtractor
from RetrieverSSA import RetrieverSSA
import numpy as np
devkit_path = Path('data-set/cars/devkit')
train_path = Path('data-set/cars/cars_train/cars_train') 

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

del df_cars['bbox_x1']
del df_cars['bbox_x2']
del df_cars['bbox_y1']
del df_cars['bbox_y2']
print(df_cars.head())
print(len(df_cars))
df_cars = df_cars.merge(labels, left_on='class', right_index=True)
df_cars = df_cars.sort_index()
print(df_cars.head())

df_train = df_cars.sample(frac=0.9, random_state=1)

df_test = df_cars.loc[~df_cars.index.isin(df_train.index)]

SSA_features=sio.loadmat('cached-data/new/cars/ssa_features_cars.mat')['ssa_features']
imgPath_obj=[str(x) for x in sio.loadmat('cached-data/new/cars/img_path_obj_cars.mat')['img_path_obj']]
fe=FeatureExtractor()
all_results=[]
length_of_test=len(df_test)
counter=0
last_time=time()
for index,row in df_test.iterrows():
    counter=counter+1

    index_of_input=imgPath_obj.index(os.getcwd()+'/'+str(train_path)+'/'+row['fname'])
    input_ssa=SSA_features[index_of_input]
    score_of_trained=[]
    for index,row_train in df_train.iterrows():
        index_of_trained=imgPath_obj.index(os.getcwd()+'/'+str(train_path)+'/'+str(row_train['fname']))
        trained_ssa=SSA_features[index_of_trained]
        distance= sqrt( sum(np.multiply(trained_ssa - input_ssa,trained_ssa - input_ssa)))
        score_of_trained.append([distance[0][0],row_train['fname'],str(row_train['labels'])]) 
    score_of_trained.sort()
    score_of_trained=score_of_trained[:5]
    new_rec={
        'input':str(row['fname'])
        ,'class':str(row['labels']),
        'output':[{'name':x[1],'class':x[2]} for x in score_of_trained]
    }
    all_results.append(new_rec) 
    print('step: '+str(counter) +'/'+str(length_of_test)+' --- progress:'+str(int((counter/length_of_test)*100))+'% --- time of step:'+(str(time()-last_time)))
    last_time=time()
json_object = json.dumps(all_results, indent=4)
with open("cached-data/new/cars/all_results.json", "w") as outfile:
    outfile.write(json_object) 
  