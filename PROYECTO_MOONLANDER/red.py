import tensorflow as tf
import numpy as np

n_inputs = 8
n_hidden1 = 32
n_hidden2 = 16
n_outputs = 1

X = tf.placeholder(tf.float32, shape=(None, n_inputs))
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
outputs = tf.layers.dense(hidden2, n_outputs, activation=tf.nn.sigmoid)

y = tf.placeholder(tf.float32, shape=(None, n_outputs))
loss = tf.reduce_mean(tf.square(y - outputs))
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
n_epochs = 100
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(len(training_data) // batch_size):
            X_batch, y_batch = next_batch(batch_size, training_data)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        mse = loss.eval(feed_dict={X: X_train, y: y_train})
        print(epoch, "MSE:", mse)

y_pred = outputs.eval(feed_dict={X: X_test})
