# -*- coding:utf-8 -*-
# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Layers.py
#this keras also need to be substituded by tensorflow.keras
#import keras


import tensorflow as tf
import tensorflow.keras as Binkeras
#from tf.keras import layers

class AddSingletonDepth(Binkeras.layers.Layer):
    #这里的x就是调用这个类的时候后面跟的那个输入
    def call(self, x, mask=None):
        #在最右侧加一个维度
        x = Binkeras.backend.expand_dims(x, -1)  # add a dimension of the right
        
        #如果维度是4的话，需要对不同维度进行转换
        if Binkeras.backend.ndim(x) == 4:
            return Binkeras.backend.permute_dimensions(x, (0, 3, 1, 2))
        #这个程序中维度应该是3，然后直接返回
        else:
            return x

    #get_output_shape_for keras是前面这个方法，tf.keras貌似是需要更改的
    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3:
            return input_shape[0], 1, input_shape[1], input_shape[2]
        else:
            return input_shape[0], input_shape[1], 1


class Subtract(Binkeras.layers.Layer):

    def __init__(self, **kwargs):
        super(Subtract, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[0] - x[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Slice(Binkeras.layers.Layer):
    #这里的x就是调用这个类的时候后面跟的那个输入
    def __init__(self, selector, output_shape, **kwargs):
        self.selector = selector
        self.desired_output_shape = output_shape
        super(Slice, self).__init__(**kwargs)

    def call(self, x, mask=None):

        selector = self.selector
        if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
            x = tf.keras.backend.permute_dimensions(x, [0, 2, 1])
            selector = (self.selector[1], self.selector[0])
        #截取target_field长度的x
        y = x[selector]

        if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
            y = tf.keras.backend.permute_dimensions(y, [0, 2, 1])
        #最后的输出维度和输入维度保持一致
        return y


    def compute_output_shape(self, input_shape):

        output_shape = (None,)
        for i, dim_length in enumerate(self.desired_output_shape):
            if dim_length == Ellipsis:
                output_shape = output_shape + (input_shape[i+1],)
            else:
                output_shape = output_shape + (dim_length,)
        return output_shape
