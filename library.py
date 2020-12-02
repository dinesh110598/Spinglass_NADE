# %%
import tensorflow.keras as tfk
from tensorflow import math as tfm
import tensorflow as tf
import numpy as np
# %%
class NADE_orig (tfk.Model):
    def __init__(self, inshape, num_hidden, **kwargs):
        super().__init__(**kwargs)
        self.shape = inshape
        self.D = inshape[0]*inshape[1]
        self.N_h = num_hidden
        self.flatten = tfk.layers.Flatten ()
        self.output_layer = []
        for i in range(self.D-1):
            self.output_layer.append (tfk.layers.Dense (1,'sigmoid'))
            self.output_layer[i].build ([self.N_h])

        #Implementing weight sharing between hidden layers
        rng = tf.random.Generator.from_non_deterministic_state ()
        self.kernel = []
        for i in range (self.D-1):
            self.kernel.append (tf.Variable (rng.uniform(shape=[self.N_h]),
                                trainable=True, name='kernel:'+str(i)))
        self.bias = tf.Variable (tf.zeros (shape=(self.N_h,)),
                                trainable=True)
        self.loss_tracker = tfk.metrics.Mean(name="logits")
        self.optimizer = tfk.optimizers.SGD()

    def call(self, x):
        """Calculates the probability of sample x
        Args:
            x (int32): Value of input lattice
        """

        def SplDense(x, n):
            """We are using this "layer" instead of regular keras Dense
            layer to facilitate use of common kernel and bias"""
            kernel = tf.stack(self.kernel[:n])
            return tfk.activations.sigmoid(tf.matmul(x, kernel) + self.bias)

        x = self.flatten(x)
        p = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        Id = tf.ones([x.shape[0]])
        p = p.write(0, Id)
        #The first lattice point is fixed at +1
        for i in range(1, self.D):
            x1 = tf.gather(x, tf.range(i), axis=1)
            h = SplDense(x1, i)
            y = tf.squeeze(self.output_layer[i-1](h))
            x2 = tf.gather(x, tf.constant(i), axis=1)
            p = p.write(i, 0.5*(Id-x2) + (x2*y))
        return tfm.reduce_mean(p.stack(), axis=0)
        #Above quantity is the joint probability of the input vector x

    def sample (self): #Convert lists here to TensorArrays

        def SplDense(x, n):
            """We are using this "layer" instead of regular keras Dense
            layer to facilitate use of common kernel and bias"""
            kernel = tf.stack(self.kernel[:n])
            return tfk.activations.sigmoid(tf.matmul(x, kernel) + self.bias)

        x = tf.TensorArray (tf.float32, size=0, dynamic_size=True)
        rng = tf.random.Generator.from_non_deterministic_state()
        prob = 1.
        x = x.write (0, 1.)
        for i in range (1, self.D):
            prob = tf.squeeze(self.output_layer[i-1] (SplDense (tf.expand_dims(x.stack(),axis=0),
                                i)))
            x = x.write(i,1. if rng.uniform(shape=[]) < prob else -1.)
        return tf.reshape (x.stack(), self.shape)
    
    @tf.function
    def train_step (self, x):
        with tf.GradientTape() as tape:
            p = self.call (x)
            loss = -tf.math.log (p)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Updates loss metric by averaging current value with prev ones
        self.loss_tracker.update_state (loss)
        print ("Traced")
        return self.loss_tracker.result()
# %%
class OutputLayer (tfk.layers.Layer):
    def __init__(self, D, N_h, **kwargs):
        super().__init__(**kwargs)
        rng = tf.random.Generator.from_non_deterministic_state()
        self.kernel = tf.Variable (rng.uniform ([N_h, D-1]), trainable=True)
        self.bias = tf.Variable (tf.zeros([D-1]), trainable=True)

    def call (self, x):
        y = tf.einsum ("ijk,kj->ij", x, self.kernel) + self.bias
        return tfk.activations.sigmoid (y)

    def sample (self, h, i):
        p = tf.matmul (h, self.kernel[:,i]) + self.bias[i]
        return tfk.activations.sigmoid (p)

class NADE_fast (tfk.Model):
    def __init__(self, inshape, num_hidden, **kwargs):
        super().__init__(**kwargs)
        self.shape = inshape
        self.D = inshape[0]*inshape[1]
        self.N_h = num_hidden
        self.flatten = tfk.layers.Flatten()
        self.mask = tf.convert_to_tensor(np.tril(np.ones((self.D-1, self.D-1)
                                        ,np.float32)))
        self.output_layer = OutputLayer (self.D, self.N_h)

        #Implementing weight sharing between hidden layers
        rng = tf.random.Generator.from_non_deterministic_state()
        self.kernel = tf.Variable(rng.uniform(shape=[self.D-1,self.N_h]),
                                trainable=True)
        self.bias = tf.Variable(rng.uniform(shape=[self.N_h,]),
                                trainable=True)
        self.loss_tracker = tfk.metrics.Mean(name="logits")
        self.optimizer = tfk.optimizers.SGD()

    @tf.function
    def call(self, x):
        """Calculates the probability of sample x
        Args:
            x (int32): Value of input lattice
        """

        def SplDense(x):
            """We are using this "layer" instead of regular keras Dense
            layer to facilitate use of common kernel and bias"""
            return tfk.activations.sigmoid(tf.matmul(x, self.kernel)
                                        + self.bias)

        x = self.flatten(x)
        x0 = tf.gather (x, tf.range(self.D-1), axis=1)
        mask = tf.broadcast_to (self.mask, [x.shape[0]]+self.mask.shape)
        x1 = tf.broadcast_to (tf.expand_dims(x0,1),
                             (x.shape[0],self.D-1,self.D-1))
        x1 = mask*x1
        #For each data inside batch, x1 contains a concat of all
        #dependencies of each element inside
        h = SplDense(x1)
        y = self.output_layer(h)
        x2 = tf.gather (x, tf.range(1,self.D),axis=1)
        p = 0.5*(1-x2) + (x2*y)
        return tf.reduce_mean (p, axis=1)

    def sample(self):
        "Draws a single sample from the distribution learned by the model"

        def SplDense(x, n):
            """We are using this "layer" instead of regular keras Dense
            layer to facilitate use of common kernel and bias"""
            kernel = tf.stack(self.kernel[:n])
            return tfk.activations.sigmoid(tf.matmul(x, kernel) + self.bias)

        x = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        rng = tf.random.Generator.from_non_deterministic_state()
        prob = 1.
        x = x.write(0, 1.)
        for i in range(1, self.D):
            prob = SplDense(x.stack(), i)
            prob = tf.squeeze(self.output_layer.sample (prob,i-1))
            x = x.write(i, 1. if rng.uniform(shape=[])<prob else -1.)
        return tf.reshape(x.stack(), self.shape)

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            p = self.call(x)
            loss = -tf.math.log(p)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Updates loss metric by averaging current value with prev ones
        self.loss_tracker.update_state(loss)
        print("Traced")
        return self.loss_tracker.result()
# %%
