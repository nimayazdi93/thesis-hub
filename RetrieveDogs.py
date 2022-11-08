from time import time
import scipy.io as sio
import numpy as np
import json  
from math import sqrt 
from FeatureExtractor import FeatureExtractor
from Saliency import Saliency 
from SSA import SSA_H_Plus

from IPython.display import HTML, display
import tabulate
def getFolderAndFileName(imgPath):
    img_path_str=str(imgPath)
    img_path_splitted= img_path_str.split("/")
    return [img_path_splitted[len(img_path_splitted)-2],img_path_splitted[len(img_path_splitted)-1].replace(' ','')] 
img_path_dogs=[str(x) for x in sio.loadmat('cached-data/new/dogs/img_path_obj_dogs_shuffle.mat')['img_path_obj']]
SSA_features=[x for x in sio.loadmat('cached-data/new/dogs/ssa_features_dogs_shuffle.mat')['ssa_features']]
img_names=[getFolderAndFileName(x)[1] for x in img_path_dogs] 
img_folders=[getFolderAndFileName(x)[0] for x in img_path_dogs]  
all_results=[]
split_point=int(len(img_names)*.1)
split_point=10
training, test = img_names[split_point:], img_names[:split_point] 
counter=0
last_time=time()
fe=FeatureExtractor()
slc=Saliency()

 





for input in test:
    counter=counter+1
    index_of_test=img_names.index(input)
    input_folder=img_folders[index_of_test]
 
    salient=slc.GetSaliency('data-set/dogs/images/Images/'+input_folder+'/'+input,70)
    salient_feature=fe.extract_by_array(salient)
    input_ssa=SSA_H_Plus(salient_feature)

    # input_ssa=SSA_features[index_of_test]
    score_of_trained=[]
    for trained in training:
        index_of_trained=img_names.index(trained)
        trained_ssa=SSA_features[index_of_trained]
        distance= sqrt( sum(np.multiply(trained_ssa - input_ssa,trained_ssa - input_ssa)))
        trained_folder=img_folders[index_of_trained]
        score_of_trained.append([distance,trained,trained_folder])

    score_of_trained.sort()
    score_of_trained=score_of_trained[:5]
    new_rec={
        'input':input
        ,'class':input_folder,
        'output':[{'name':x[1],'class':x[2]} for x in score_of_trained]
    }
    all_results.append(new_rec) 
    print('step: '+str(counter) +'/'+str(len(test))+' --- progress:'+str(int((counter/len(test))*100))+'% --- time of step:'+(str(time()-last_time)))
    last_time=time()
json_object = json.dumps(all_results, indent=4)
with open("cached-data/new/dogs/all_results.json", "w") as outfile:
    outfile.write(json_object)
 
