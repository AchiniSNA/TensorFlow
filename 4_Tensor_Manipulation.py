import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf

x = tf.constant([0,10,20,15,5,13,56,90])
print(x)      #tf.Tensor([ 0 10 20 15  5 13 56 90], shape=(8,), dtype=int32)
print(x[:])     #tf.Tensor([ 0 10 20 15  5 13 56 90], shape=(8,), dtype=int32)
print(x[1:])    #tf.Tensor([10 20 15  5 13 56 90], shape=(7,), dtype=int32)
print(x[0:4])   #tf.Tensor([ 0 10 20 15], shape=(4,), dtype=int32)


#skip values
print(x[::2])      # print 0,2,4,6
#tf.Tensor([ 0 20  5 56], shape=(4,), dtype=int32)
print(x[::3])      # print 0,3,6,9
#tf.Tensor([ 0 15 56], shape=(3,), dtype=int32)


#Reverse
print(x[::-1])     #tf.Tensor([90 56 13  5 15 20 10  0], shape=(8,), dtype=int32)


# create a tensor from selected elements using indexes
selectedindices = tf.constant([0,3])
indexedTensor = tf.gather(x, indices= selectedindices)
print(indexedTensor)      #tf.Tensor([ 0 15], shape=(2,), dtype=int32)


x = tf.constant([[10,20],[30,40],[50,60],[70,80]])
print(x)
#tf.Tensor(
#[[10 20]
# [30 40]
# [50 60]
# [70 80]], shape=(4, 2), dtype=int32)


#get first raw with all the elements               # x[rows , columns]
print(x[0])            #tf.Tensor([10 20], shape=(2,), dtype=int32)    
print(x[0, :2])        #tf.Tensor([10 20], shape=(2,), dtype=int32)
print(x[0:2 , :1])
#tf.Tensor(
#[[10]
# [30]], shape=(2, 1), dtype=int32)

print(x[0:3 , :1])
#tf.Tensor(
#[[10]
# [30]
# [50]], shape=(3, 1), dtype=int32)



# Reshaping
tensorA= tf.range(16)
print(tensorA)     #tf.Tensor([ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15], shape=(16,), dtype=int32)



x= tf.reshape(tensorA,(4,4))
print(x)
#tf.Tensor(
#[[ 0  1  2  3]
# [ 4  5  6  7]
# [ 8  9 10 11]
# [12 13 14 15]], shape=(4, 4), dtype=int32)

y= tf.reshape(tensorA,(2,8))
print(y)
#tf.Tensor(
#[[ 0  1  2  3  4  5  6  7]
# [ 8  9 10 11 12 13 14 15]], shape=(2, 8), dtype=int32)