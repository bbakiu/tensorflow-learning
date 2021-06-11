import tensorflow as tf

print(tf.compat.v1.executing_eagerly())

a = tf.constant(5, name='a')
b = tf.constant(7, name='b')
c = tf.add(a, b, name='sum')
#
# sess = tf.compat.v1.Session()
# sess.run(c)

print(c)