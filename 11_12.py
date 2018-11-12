# -*- coding: utf-8 -*-
"""11/12.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LH5m4u1gyebUhWR1R9hZDZ2GQKBkGYOk

Use shift-enter to execute a code block and move to the next one.
"""

# 1.1 Import tensorflow and other libraries.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pylab


x=np.linspace(-10, 0)
y=(x+5)**2
plt.plot(x,y)
plt.show()

x = tf.Variable(tf.random_uniform([1], -10, 0.0))

# 1.4 Build training graph.
loss = tf.reduce_mean(tf.square(5+x))  # Create an operation that calculates loss.
optimizer = tf.train.GradientDescentOptimizer(0.5)  # Create an optimizer.
train = optimizer.minimize(loss)  # Create an operation that minimizes loss.
init = tf.initialize_all_variables()  # Create an operation initializes all the variables.

# 1.6 Create a session and launch the graph.
sess = tf.Session()
sess.run(init)

print(sess.run([x]))

# 1.7 Perform training.
for step in range(201):
    sess.run(train)


print(sess.run([x]))

"""<p>Back to [0_tf_hello_world.ipynb](0_tf_hello_world.ipynb).</p>
<p>Next to [2_mnist.ipynb](2_mnist.ipynb).</p>
"""