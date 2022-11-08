import os
import pandas as pd
import numpy as np
from SSA import SSA_H_Plus
from FeatureExtractor import FeatureExtractor 
import scipy.io as sio  
fe=FeatureExtractor() 
image_paths = []  
image_folder = os.getcwd()+'/data-set/cars/cars_train/'
print(image_folder)
for path, subdirs, files in os.walk(image_folder):
    for name in files:
        if(name.endswith('.jpg')):
            image_paths.append(os.path.join(path, name)) 

imgPath_feature=[]
imgPath_obj=[]
k=1
for img_path in image_paths:
    feature=fe.extract(img_path)
    if k==1:
        feature_matrix=feature
        k=0
    else:
        feature_matrix=np.vstack([feature_matrix,feature]) 
    imgPath_obj.append(img_path) 

imgPath_obj 
sio.savemat('cached-data/new/img_path_obj_cars.mat',{"img_path_obj":imgPath_obj})
 
k=1
for fet in feature_matrix:
    ssa_plus=SSA_H_Plus(fet) 
    if k==1:
        ssa_plus_features=ssa_plus 
        k=0
    else:
        ssa_plus_features=np.vstack([ssa_plus_features,ssa_plus])  
sio.savemat('cached-data/new/SSA_features_cars.mat',{'SSA_features':ssa_plus_features})









