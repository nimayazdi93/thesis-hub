from keras.preprocessing import image
from keras.applications.vgg16 import VGG16,preprocess_input
import keras.preprocessing as preprocessing
from keras.models import Model
from keras.utils import load_img 
import numpy as np

class FeatureExtractor:
    def __init__(self) :
        base_model=VGG16(weights='vgg16_weights.h5') 
        self.model=Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
        #7x7x512
    def extract(self,img_path):
        img = load_img(img_path, target_size=(224, 224)) 
        img=img.convert('RGB')
        x=preprocessing.image.image_utils.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        feature=self.model.predict(x)[0]
        return feature/np.linalg.norm(feature)
    def extract_by_array(self,img_array): 
        x=np.expand_dims(img_array,axis=0)
        x=preprocess_input(x)
        feature=self.model.predict(x)[0]
        return feature/np.linalg.norm(feature)