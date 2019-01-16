"""
    source : https://tensorflow.google.cn/tutorials/eager/custom_layers
    The best way to implement your own layer is extending the tf.keras.Layer class and implementing:
    * __init__ ,where you can do all input-independent initialization
    * build, where you know the shapes of the input tensors and can do the rest of the initialization
    * call,where you do the forward computation

    Note that you don't have to wait until build is called to create your variables,you can also create them in __init__.
    However, the advantage of creating them in build is that it enables late variable creation based on the shape of the inputs the layer will operate on.
    On the other hand, creating variables in __init__ would mean that shapes required to create the variables will need to be explicitly specified.
"""

import tensorflow as tf
tf.enable_eager_execution()


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[int(input_shape[-1]),
                                               self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


if __name__ == '__main__':
    layer = MyDenseLayer(10)
    print(layer(tf.zeros([10, 5])))
    print(layer.variables)

