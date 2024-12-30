import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf

tensorA = tf.constant(8)
print(tensorA)
#tf.Tensor(8, shape=(), dtype=int32)


tensorA = tf.constant(8.0)
print(tensorA)
#tf.Tensor(8.0, shape=(), dtype=float32)



tensorA = tf.constant([[10,20,30],[40,50,60]],shape=(3,2))
print(tensorA)
#tf.Tensor(
#[[10 20]
# [30 40]
# [50 60]], shape=(3, 2), dtype=int32)



#change dtype
tensorA= tf.constant(8,shape=(1,1), dtype=tf.float32)
print(tensorA)
#tf.Tensor([[8.]], shape=(1, 1), dtype=float32)



# 1 matrix
tensorA=tf.ones((4,4))
print(tensorA)
#tf.Tensor(
#[[1. 1. 1. 1.]
#[1. 1. 1. 1.]
#[1. 1. 1. 1.]
#[1. 1. 1. 1.]], shape=(4, 4), dtype=float32)




# 0 matrix
tensorA=tf.zeros((4,4))
print(tensorA)
#[[0. 0. 0. 0.]
# [0. 0. 0. 0.]
# [0. 0. 0. 0.]
# [0. 0. 0. 0.]], shape=(4, 4), dtype=float32)



# Identity matrix
tensorA=tf.eye(4)
print(tensorA)
#tf.Tensor(
#[[1. 0. 0. 0.]
# [0. 1. 0. 0.]
# [0. 0. 1. 0.]
# [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)