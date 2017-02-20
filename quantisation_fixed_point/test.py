# 0.5, 0.25, 0.125

# if i have 4 bits
# I can track from 0-15(1111)
import tensorflow as tf
import numpy as np

matrix1 = tf.Variable([0.3, 0.13])

# 0.75
check = matrix1 > 0.75
# q1 = tf.to_float(check) * 0.75
# q2 = tf.to_float(tf.logical_and(matrix1 <= 0.75, matrix1 >0.5)) * 0.5
# q3 = tf.to_float(tf.logical_and(matrix1 <= 0.5, matrix1 >0.25)) * 0.25
# q4 = tf.to_float(matrix1 <= 0.25) * 0
# quan = q1 + q2 + q3 + q4
# q = tf.floordiv(matrix1, 0.25) * 0.25
q = matrix1 - tf.mod(matrix1 , 0.25)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(q.eval())
