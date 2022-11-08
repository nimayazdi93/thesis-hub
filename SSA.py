
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
import tensorflow as tf

def SSA_H_Plus(features,alpha=1):
    A=tf.reshape(features,[7*7*512]).numpy()
   
    B = A
    C = A
    AxB = np.multiply(A, B) 
    S = softmax(AxB)
    CxS = np.multiply(S, C)
    AlphaxCxS=alpha*CxS 
    H = np.add(A,AlphaxCxS)
    return H


def SSA_H_Mul(features,alpha=1):
    A= features
    print(len(A))
    B = A
    C = A
    AxB = np.multiply(A, B) 
    S = softmax(AxB)
    CxS = np.multiply(S, C)
    AlphaxCxS=alpha*CxS
    H = np.multiply(A,AlphaxCxS)
    return H
def SSA(features):
    return SSA_H_Plus(features)
def softmax(x): 
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x