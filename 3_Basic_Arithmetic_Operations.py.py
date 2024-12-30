import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
import numpy as np


tensorA = tf.constant([20,6,7])
tensorB = tf.constant([45,7,21])

print(tensorA)
#tf.Tensor([20  6  7], shape=(3,), dtype=int32)
print(tensorB)
#tf.Tensor([45  7 21], shape=(3,), dtype=int32)



#Tensor Addition
tensorC = tensorA + tensorB
print(tensorC)
tensorC = tf.add(tensorA, tensorB)
print(tensorC)
#tf.Tensor([65 13 28], shape=(3,), dtype=int32)



#Tensor Substraction
tensorD = tf.subtract(tensorA,tensorB)
print(tensorD)
#tf.Tensor([-25  -1 -14], shape=(3,), dtype=int32)



#Tensor Divition
tensorE = tf.divide(tensorA,tensorB)
print(tensorE)
#tf.Tensor([0.44444444 0.85714286 0.33333333], shape=(3,), dtype=float64)



#Tensor Multiplication
tensorF = tf.multiply(tensorA,tensorB)
print( tensorF)
#tf.Tensor([900  42 147], shape=(3,), dtype=int32)


# Dot product
tensorA = tf.constant(np.array([[1,2],[3,4]]))
tensorB = tf.constant(np.array([[11,12],[13,14]]))

tensorG = tf.tensordot(tensorA,tensorB, axes =1)
print(tensorG)
#tf.Tensor(
#[[37 40]
# [85 92]], shape=(2, 2), dtype=int32)