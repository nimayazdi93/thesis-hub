import scipy.io as sio
from FeatureExtractor import FeatureExtractor
from SSA import SSA
import numpy as np 
from PIL import Image
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import easygui


def RetrieverVGG(input_img,dataset):
    fe=FeatureExtractor()
    feature_matrix=sio.loadmat('cached-data/feature_matrix'+str(dataset)+'.mat')['feature_matrix']
    imgPath_obj=sio.loadmat('cached-data/img_path_obj'+str(dataset)+'.mat')['img_path_obj']
 
    input_feature=fe.extract(input_img) 

    img_score=[]
    for k in range(len(feature_matrix)): 
        fe_vgg=feature_matrix[k]
        imagePath=imgPath_obj[k]
        distance= sum(abs(input_feature - fe_vgg))
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

# f, axarr = plt.subplots(4,2) 
# axarr[0,0].imshow(mpimg.imread(input_img))
# axarr[1,0].imshow(mpimg.imread(output_src[0].replace(" ","")))
# axarr[1,1].imshow(mpimg.imread(output_src[1].replace(" ","")))
# axarr[2,0].imshow(mpimg.imread(output_src[2].replace(" ","")))
# axarr[2,1].imshow(mpimg.imread(output_src[3].replace(" ","")))
# axarr[3,0].imshow(mpimg.imread(output_src[4].replace(" ","")) )
# axarr[3,1].imshow(mpimg.imread(output_src[5].replace(" ","")) )
# plt.grid()
# plt.show()