import scipy.io as sio
import numpy as np 
img_path_dogs=[str(x).replace(' ','') for x in sio.loadmat('cached-data/new/LFW/img_path_obj_lfw.mat')['img_path_obj']]
SSA_features=[x for x in sio.loadmat('cached-data/new/LFW/ssa_features_lfw.mat')['ssa_features']]
indices=list(range(int(len(img_path_dogs))))
np.random.shuffle(indices)
new_img_path_dogs=[]
new_ssa_features=[] 
for index in indices:
     new_img_path_dogs.append(img_path_dogs[index])
     new_ssa_features.append(SSA_features[index])
sio.savemat('cached-data/new/LFW/img_path_obj_lfw.mat',{"img_path_obj":new_img_path_dogs})
sio.savemat('cached-data/new/LFW/ssa_features_lfw.mat',{"ssa_features":new_ssa_features})