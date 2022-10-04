from math import sqrt
import scipy.io as sio
import numpy as np
from FeatureExtractor import FeatureExtractor
from SSA import SSA, SSA_H_Mul, SSA_H_Plus 
import sys
def RetrieverSSA(input_img,dataset,method):
    # input_img=sys.argv[1]
    # dataset=sys.argv[2]
    # method=sys.argv[3]
    fe=FeatureExtractor()
    SSA_features=sio.loadmat('cached-data/SSA_features'+str(dataset)+'-'+method+'.mat')['SSA_features']
    imgPath_obj=sio.loadmat('cached-data/img_path_obj'+str(dataset)+'.mat')['img_path_obj']
    input_feature=fe.extract(input_img)
    if method=='mul':
        input_ssa=SSA_H_Mul(input_feature)
    else:
        input_ssa=SSA_H_Plus(input_feature)
    input_ssa=SSA(input_feature) 
    img_score=[]
    for k in range(len(SSA_features)): 
        ssa_f=SSA_features[k]
        distance= sqrt( sum(np.multiply(ssa_f - input_ssa,ssa_f - input_ssa)))
        if distance==0:
            continue
        img_score.append([distance,k]) 
    img_score.sort()

    output_src=[]
    for i in range(20):
        outputpath=img_score[i]
        outputpath1=outputpath[1] 
        outputsource=imgPath_obj[outputpath1] 
        output_src.append(outputsource) 
    return output_src