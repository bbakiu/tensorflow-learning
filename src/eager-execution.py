import tensorflow as tf

print(tf.executing_eagerly())

x = [[10.]]
res = tf.matmul(x, x)

print(res)

a = tf.constant([[10, 20],
                 [30, 40]])

b = tf.add(a, 2)
print(b)

print(a * b)

m = tf.Variable([4.0, 5.0, 6.0], tf.float32, name='m')
c = tf.Variable([1.0, 1.0, 1.0], tf.float32, name='c')

x = tf.Variable([100.0, 100.0, 100.0], tf.float32, name='x')

y = m*c + x

print(y)