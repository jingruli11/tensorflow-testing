import numpy as np 
import tensorflow as tf 

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

print node1, node2

sess = tf.Session()
print sess.run([node1, node2])

node3 = tf.add(node1, node2)
print "node3: ", node3
print "sess.run(node3): ", sess.run(node3)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b # + is a shortcut for tf.add()

print sess.run(adder_node, {a: 3, b: 4.5})
print sess.run(adder_node,{a: [1,3], b:[2,4]})

add_and_triple = adder_node * 3 
print sess.run(add_and_triple,{a: 3, b: 4.5})

w = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)
linear_model = w * x + b

init = tf.global_variables_initializer()
sess.run(init)

print sess.run(linear_model, {x:[1,2,3,4]})

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})

# change the value of w and b
fixw = tf.assign(w, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixw,fixb])
print sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]})

# gradient descent to minimize loss, and find optimal value for w and b
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset value of w and b to initial
for i in range(0,1000):
	sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})

# evaluating training accuracy
ew, eb, eloss = sess.run([w,b,loss],{x:[1,2,3,4], y:[0,-1,-2,-3]})
print ew, eb, eloss



