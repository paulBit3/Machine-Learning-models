import tensorflow as tf



### This is my TensorFlow training activities ###
### Starting on Sep.6, 2025


#let print the tensor version
#print(tf.version)

#Before doing anything, we need to import TensorFlow library
#This is what I did on the first Top of this activity

#let create some mutable tensors using tf.Variable()
string = tf.Variable("a string", tf.string)
number = tf.Variable(325, tf.int32)
gloating = tf.Variable(4.389, tf.float64)

#let create shape

#tf.ones() creates a shape [1,2,3] tensor full of ones
tensor1 = tf.ones([1,2,3])

#we print it out to see
print(tensor1)

#reshape existing data to shape [2,3,1]
tensor2 = tf.reshape(tensor1, [2,3,1])

#we print it out to see
print(tensor2)


#-1 tells the tensor to calculate the size
#of the dimension in that place 
#this will reshape the tensor to [3,2]
tensor3 = tf.reshape(tensor2, [3, -1])

#we print it out to see
print(tensor3)

t = tf.zeros([5,5,5,5])
#print it out
print(t)

#let reshape t
t = tf.reshape(t, [625])

#print it out
print(t)