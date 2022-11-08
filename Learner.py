import os 
from SSA import SSA_H_Plus
from FeatureExtractor import FeatureExtractor 
import scipy.io as sio  
from matplotlib.pylab import plot
fe=FeatureExtractor() 
image_paths = [] 
dataset='dogs' 
image_folder = os.getcwd()+'/data-set/cars/cars_train/'
print(image_folder)
for path, subdirs, files in os.walk(image_folder):
    for name in files:
        if(name.endswith('.jpg')):
            image_paths.append(os.path.join(path, name)) 

imgPath_feature=[]
imgPath_obj=[]
k=1
feature_matrix=[]
for img_path in image_paths:
    feature=fe.extract(img_path) 
    # if k==2000:
    #     break
    # if k==1:
    #     feature_matrix=feature
       
    # else:
    #     feature_matrix=np.vstack([feature_matrix,feature])
    k=k+1
    feature_matrix.append(feature)
    imgPath_obj.append(img_path) 

imgPath_obj 
sio.savemat('cached-data/new/cars/img_path_obj_cars.mat',{"img_path_obj":imgPath_obj})
 
k=1
ssa_plus_features=[]
for fet in feature_matrix:
    ssa_plus=SSA_H_Plus(fet)  
    # if k==1:
    #     ssa_plus_features=ssa_plus 
    #     k=0
    # else:
    #     ssa_plus_features=np.vstack([ssa_plus_features,ssa_plus]) 
    ssa_plus_features.append(ssa_plus)
sio.savemat('cached-data/new/cars/ssa_features_cars.mat',{'ssa_features':ssa_plus_features})


