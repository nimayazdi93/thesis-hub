import scipy.io as sio
from FeatureExtractor import FeatureExtractor 


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