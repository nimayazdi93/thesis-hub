import json
import re
from matplotlib import pyplot as plt 
import numpy as np
from keras.utils import load_img
import keras.preprocessing as preprocessing


def LoadImage(file_path):
    img = load_img(file_path, target_size=(224, 224)) 
    img=img.convert('RGB')
    x=preprocessing.image.image_utils.img_to_array(img)
    return x
 
with open('cached-data/new/dogs/all_results.json', 'r') as openfile: 
    all_results = json.load(openfile)
count_of_true=0
count_of_all=0
for rec in all_results: 
    count_of_this_true=len([x for x in rec['output'] if  x['class']==rec['class']])
    print(count_of_this_true)
    count_of_true=count_of_true+count_of_this_true
    count_of_all=count_of_all+len(rec['output']) 

print(count_of_true)
print(count_of_all)