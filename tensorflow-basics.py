import numpy as np
import tensorflow as tf

#numpy multiplying random matrices

x = np.random.normal(size=[10,10])
y = np.random.normal(size=[10,10])
print(np.dot(x,y))

#tensorflow multiplication with random matrices
a = tf.random_normal([10,10])
b = tf.random_normal([10,10])
c = tf.matmul(x,y)
sess = tf.Session()
z_val = sess.run(c)
print(z_val)

#loss function estimation

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
#stack values in matrice like
w = tf.get_variable("w", shape=[3,1])
f = tf.stack([tf.square(x), x, tf.ones_like(x)],1)
yhat = tf.squeeze(tf.matmul(f,w),1)

loss = tf.nn.l2_loss(yhat-y) + 0.1 * tf.nn.l2_loss(w)

train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

def generate_data():
    x_val = np.random.uniform(-10.0, 10.0, size=100)
    y_val = 5 * np.square(x_val) + 3
    return x_val, y_val

sess.run(tf.global_variables_initializer())

for _ in range(1000):
    x_val, y_val = generate_data()
    _, loss_val = sess.run([train_op ,loss], {x:x_val, y:y_val})
    print(loss_val)

print(sess.run([w]))
