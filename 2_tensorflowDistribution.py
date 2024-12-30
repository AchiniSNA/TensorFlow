import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf

# Normal Distribution
x = tf.random.normal((4,4), mean=0, stddev=1)
print(x)
#tf.Tensor(
#[[-1.963138    0.18634577 -0.72879213  0.6124001 ]
# [ 1.1883339   0.14522259 -0.07838288 -0.5395782 ]
# [-1.0537488   1.1570854   0.53452635 -0.76802135]
# [-0.4907918  -2.175478    0.75556064 -0.11852986]], shape=(4, 4), dtype=float32)



# Uniform Distribution
x = tf.random.uniform((1,5), minval=0, maxval=2)
print(x)
#tf.Tensor([[1.8979106 1.2854643 1.1873655 1.4244471 1.0397713]], shape=(1, 5), dtype=float32)



#range function
x = tf.range(start=5, limit=20, delta=3)
print(x)
# tf.Tensor([ 5  8 11 14 17], shape=(5,), dtype=int32)



# Casting
y = tf.cast(x, dtype=tf.float64)
print(y)
#tf.Tensor([ 5.  8. 11. 14. 17.], shape=(5,), dtype=float64)