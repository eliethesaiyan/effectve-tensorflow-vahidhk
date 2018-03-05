import tensorflow as tf

a = tf.placeholder(tf.float32, [None,128])
static_shape = a.shape.as_list()
print(static_shape)
# a dynamic shape is also a tensor
dynamic_shape = tf.shape(a)
print(dynamic_shape) 
a.set_shape([32, 128])
a.set_shape([None, 128])
a = tf.reshape(a, [32,128])


#returns static shape if available,dynamic 
#in other case
#returns dims tensor

def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape =tf.unstack(tf.shape(tensor))
  dims = [ s[1] if s[0] is None else s[0] 
           for s in zip(static_shape, dynamic_shape)]
