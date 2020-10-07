import tensorflow as tf

def max_op(x,N):
    if x >= 1/N:
        return x
    else:
        return 0

import numpy as np
np_max_op = np.vectorize(max_op)
    

x=np.array([1,2,4,0.2,0.3,1,1,-0.1,-2,0,1,2]).reshape([3,2,2])
x1=  tf.convert_to_tensor(x, np.float32)
w=np.array([0,1,0])
w1=  tf.convert_to_tensor(w, np.float32)
N=2
mask = tf.where(x1 > 1/N, tf.ones_like(x1), tf.zeros_like(x1))

x2 = tf.multiply(x1, mask)
with tf.Session() as sess: 
    xx= x1.eval()
    cc=x2.eval()

print(xx)
print('##############')
print(cc)
