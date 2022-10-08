import os
import pandas as pd
import numpy as np
from SSA import SSA_H_Plus,SSA_H_Mul
from FeatureExtractor import FeatureExtractor 
import scipy.io as sio  
fe=FeatureExtractor() 
image_paths = [] 
dataset='dogs'
method='ssa-mul'
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
sio.savemat('cached-data/feature_matrix'+dataset+'.mat',{'feature_matrix':feature_matrix})
sio.savemat('cached-data/img_path_obj'+dataset+'.mat',{"img_path_obj":imgPath_obj})
 
k=1
for fet in feature_matrix:
    ssa_plus=SSA_H_Plus(fet)
    ssa_mul=SSA_H_Mul(fet)
    if k==1:
        ssa_plus_features=ssa_plus
        ssa_mul_features=ssa_mul
        k=0
    else:
        ssa_plus_features=np.vstack([ssa_plus_features,ssa_plus])
        ssa_mul_features=np.vstack([ssa_mul_features,ssa_mul])

sio.savemat('cached-data/SSA_features'+dataset+'-mul.mat',{'SSA_features':ssa_mul_features})
sio.savemat('cached-data/SSA_features'+dataset+'-plus.mat',{'SSA_features':ssa_plus_features})


