import tensorflow as tf
import numpy as np
import PIL.Image 
from keras.applications.vgg16 import VGG16,preprocess_input 
from keras.models import Model 
import saliency.core as saliency 
class Saliency:
    def __init__(self) :
        m = VGG16(weights='vgg16_weights.h5')
        conv_layer = m.get_layer('fc1')
        self.model = Model([m.inputs], [conv_layer.output, m.output])
        
        self.class_idx_str = 'class_idx_str'
    def LoadImage(self,file_path):
        im = PIL.Image.open(file_path)
        im = im.resize((224,224))
        im = np.asarray(im)
        return im
    def PreprocessImage(self,im):
        im = preprocess_input(im)
        return im
    def call_model_function(self,images, call_model_args=None, expected_keys=None):
        target_class_idx =  call_model_args[self.class_idx_str]
        images = tf.convert_to_tensor(images)
        with tf.GradientTape() as tape:
            if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
                tape.watch(images)
                _, output_layer = self.model(images)
                output_layer = output_layer[:,target_class_idx]
                gradients = np.array(tape.gradient(output_layer, images))
                return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
            else:
                conv_layer, output_layer = self.model(images)
                gradients = np.array(tape.gradient(output_layer, conv_layer))
                return {saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                        saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}
    
    def GetSaliency(self,input,trs):
        im_orig = self.LoadImage(input)
        im = self.PreprocessImage(im_orig)
        _, predictions = self.model(np.array([im]))
        prediction_class = np.argmax(predictions[0])
        call_model_args = {self.class_idx_str: prediction_class}
        #print("Prediction class: " + str(prediction_class))  # Should be a doberman, class idx = 236
        xrai_object = saliency.XRAI()
        xrai_attributions = xrai_object.GetMask(im, self.call_model_function, call_model_args, batch_size=20)
        mask = xrai_attributions >= np.percentile(xrai_attributions, trs)
        X_True=[]
        Y_True=[]
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if(mask[i][j]):
                    X_True.append(i)
                    Y_True.append(j)
        Xmin=min(X_True)
        XMax=max(X_True)
        Ymin=min(Y_True)
        YMax=max(Y_True)
        im_mask = np.array(im_orig)
        im_mask[~mask] = 0

        im_mask=im_mask[Xmin:XMax,Ymin:YMax]  

        arr2im = PIL.Image.fromarray(im_mask)
        im = arr2im.resize((224,224))
        im = np.asarray(im)
        return im
