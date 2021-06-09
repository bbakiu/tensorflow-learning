import tensorflow as tf

v1 = tf.Variable([[1.5, 2, 5], [2, 6, 7]])
print(v1)

v2 = tf.Variable([[1, 2, 5], [2, 6, 7]])
print(v2)

v3 = tf.Variable([[1, 2, 5], [2, 6, 7]], dtype=tf.float32)
print(v3)

v4 = tf.add(v1, v3)
print(v4)
print(tf.convert_to_tensor(v1))

print(v4.numpy())

v1[0, 0].assign(100)
print(v1.numpy())

v1.assign_add([[1, 1, 1], [1, 1, 1]])
print(v1)

var_a = tf.Variable([2.0, 3.0])
var_b = tf.Variable(var_a)
print(var_b)

var_b.assign([200,300])
print(var_a.numpy())
print(var_b.numpy())
