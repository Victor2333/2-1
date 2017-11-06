import tensorflow as tf
import numpy as np

x_data = np.random.rand(100)
y_data = x_data * 10.2 + 2.2

k = tf.Variable(0.)
b = tf.Variable(0.)
y = x_data * k + b

loss = tf.reduce_mean(tf.square(y_data - y))

optimizer = tf.train.GradientDescentOptimizer(0.1)

train = optimizer.minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        sess.run(train)
        if step % 10 == 0:
            print(sess.run([k, b]))
