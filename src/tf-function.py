import tensorflow as tf


@tf.function
def add(a, b):
    return a + b


@tf.function
def sub(a, b):
    return a - b


@tf.function
def div(a, b):
    return a / b


@tf.function
def mul(a, b):
    return a * b


@tf.function
def matmul(a, b):
    return tf.matmul(a, b)


@tf.function
def linear(a, b, d):
    return add(matmul(a, b), d)


@tf.function
def pos_neg_check(x):
    reduce_sum = tf.reduce_sum(x)

    if reduce_sum > 0:
        return tf.constant(1)
    elif reduce_sum == 0:
        return tf.constant(0)
    else:
        return tf.constant(-1)


@tf.function
def add_times(x1):
    for i in tf.range(x1):
        num.assign_add(x1)


print(sub(tf.constant(5), tf.constant(3)))
print(mul(tf.constant(5), tf.constant(3)))
print(div(tf.constant(5), tf.constant(3)))

m = tf.constant([4.0, 5.0, 6.0], tf.float32)
c = tf.constant([[1.0]], tf.float32)

x = tf.Variable([[100.0], [100.0], [100.0]], tf.float32)

# print(linear(m, x, c))

print(pos_neg_check([100, 100]))
print(pos_neg_check([100, -100]))
print(pos_neg_check([100, -101]))

num = tf.Variable(7)

add_times(5)

print(num)
