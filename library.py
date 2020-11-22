import tensorflow as tf
import tensorflow.keras as tfk
#import numpy as np

class HiddenLayer (tfk.layers.Layer):
    def __init__ (self, inshape, outshape, **kwargs):
        super().__init__(**kwargs)
        self.D = inshape
        self.N_h = outshape

    def call (self):
        pass

class NADE (tfk.Model):
    def __init__(self, inshape, num_hidden, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = inshape
        self.D = inshape[0]*inshape[1]
        self.N_h = num_hidden
        self.flatten = tfk.layers.Flatten(input_shape=inshape)
        self.HL = []
        self.output_layer = []
        for _ in range(self.D):
            self.HL.append(tfk.layers.Dense(self.N_h, 'sigmoid'))
            self.output_layer.append (tfk.layers.Dense (1,'sigmoid'))

        #Implementing weight sharing between hidden layers
        with tf.name_scope(self.HL[0].name):
            self.HL[0].build(self.D-1)
        for i in range (1,self.D-1):
            with tf.name_scope(self.HL[i].name):
                self.HL[i].build(self.D-i-1)
            self.HL[i].kernel = self.HL[0].kernel[:,:(self.D-i-1)]
            self.HL[i].bias = self.HL[0].bias
            self.HL[i]._trainable_weights = []
            self.HL[i]._trainable_weights.append(self.HL[i].kernel)
            self.HL[i]._trainable_weights.append(self.HL[i].bias)

        with tf.name_scope(self.HL[self.D-1].name):
            self.HL[self.D-1].build(1)
        self.HL[self.D-1].kernel = self.HL[0].kernel[:, :1]
        self.HL[self.D-1].bias = self.HL[0].bias
        self.HL[self.D-1]._trainable_weights = []
        self.HL[self.D-1]._trainable_weights.append(self.HL[self.D-1].kernel)
        self.HL[self.D-1]._trainable_weights.append(self.HL[self.D-1].bias)

    def call (self, x):
        x = self.flatten (x)
        h = []
        for i in range (self.D-1):
            h.append (self.HL[i] (x[:self.D-i-1]))
        #Just make sure self.HL[self.D-1] receives only zero as input
        h.append (self.HL[self.D-1](tf.constant (0)))
        p = tf.zeros (self.D)
        for i in range(self.D):
            p[i] = (0.5*(1-x[i]))+x[i]*self.output_layer[i] (h[self.D-i-1])
        return tf.reduce_prod (p) #This quantity is the joint probability of the input vector x


    @tf.function
    def sample (self):
        x = []
        prob = self.output_layer[0](self.HL[self.D-1](0))
        x.append (1 if tf.random.uniform(shape=()) < prob else -1)
        for i in reversed (range (self.D-1)):
            x_tensor = tf.constant (x)
            prob = self.output_layer[self.D-1-i](self.HL[i](x_tensor))
            x.append(1 if tf.random.uniform(shape=()) < prob else -1)
        return tf.reshape (tf.constant (x), self.input_shape)
