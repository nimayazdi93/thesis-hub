import os 
from SSA import SSA_H_Plus
from FeatureExtractor import FeatureExtractor 
import scipy.io as sio

fe=FeatureExtractor() 
image_paths = []  
image_folder = os.getcwd()+'/data-set/cars/cars_train'

for path, subdirs, files in os.walk(image_folder):
    for name in files:
        if(name.endswith('.jpg')):
            image_paths.append(os.path.join(path, name)) 

imgPath_feature=[]
imgPath_obj=[] 
feature_matrix=[]
k=0
for img_path in image_paths:
    feature=fe.extract(img_path)
    feature_matrix.append(feature)
    imgPath_obj.append(img_path)
    k=k+1
    print('VGG:'+str(k))
imgPath_obj 
sio.savemat('cached-data/new/cars/img_path_cars.mat',{"img_path_obj":imgPath_obj})

k=0
ssa_plus_features=[]
for fet in feature_matrix:
    ssa_plus=SSA_H_Plus(fet)   
    ssa_plus_features.append(ssa_plus)
    k=k+1
    print('SSA:'+str(k))
sio.savemat('cached-data/new/cars/ssa_features_cars.mat',{'ssa_features':ssa_plus_features})