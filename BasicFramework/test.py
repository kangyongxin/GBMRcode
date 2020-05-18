import tensorflow as tf
import numpy as np
with tf.device('/cpu:0'):
    x = np.array([[[1,1,1,3,4,5],[2,2,7,4,6,7],[3,3,4,3,2,4]],[[4,3,3,4,5,64],[2,3,5,6,9,5],[3,0,23,2,4,6]]])
    print("x[0]",x[0,:,:])
    xx = tf.cast(x,tf.float32) 
    mean_all = tf.reduce_mean(xx)
    mean_0 = tf.reduce_mean(xx, axis=0)
    mean_1 = tf.reduce_mean(xx, axis=1)
    mean_2 = tf.reduce_mean(xx, axis=2)

    print(mean_all)
    print(mean_0)
    print(mean_1)
    print(mean_2)