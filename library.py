# %%
import tensorflow as tf
import tensorflow.keras as tfk
# %%
class NADE (tfk.Model):
    def __init__(self, inshape, num_hidden, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = inshape
        self.D = inshape[0]*inshape[1]
        self.N_h = num_hidden
        self.flatten = tfk.layers.Flatten(input_shape=inshape)
        self.output_layer = []
        for _ in range(self.D-1):
            self.output_layer.append (tfk.layers.Dense (1,'sigmoid'))

        #Implementing weight sharing between hidden layers
        rng = tf.random.Generator.from_non_deterministic_state ()
        self.kernel = []
        for _ in range (self.D-1):
            self.kernel.append (tf.Variable (rng.uniform(shape=[self.N_h]), trainable=True))
        self.bias = tf.Variable (tf.zeros (shape=(self.N_h,)), trainable=True)
        self.loss_tracker = tfk.metrics.Mean(name="logit")
    
    def __SplDense (self, x):
        kernel = tf.stack (self.kernel [:x.shape[...,0]])
        return tfk.activations.sigmoid(tf.tensordot(x, kernel) + self.bias)

    def call (self, x):
        """Calculates the probability of sample x
        Args:
            x (int32): Value of input lattice
        """
        x = self.flatten (x)
        h = 0.
        p = []
        p.append (0.5) #The first lattice point is chosen arbitrarily
        for i in reversed(range(self.D-1)):
            h = self.__SplDense (x[:self.D-1-i])
            p.append((0.5*(1-x[i]))+x[i]*self.output_layer[i](h))
        return tf.reduce_prod (p, axis=-1) 
        #Above quantity is the joint probability of the input vector x

    @tf.function
    def sample (self):
        x = []
        rng = tf.random.Generator.from_non_deterministic_state()
        prob = 0.5
        x.append (1 if rng.uniform(shape=()) < prob else -1)
        for i in reversed(range (self.D-1)):
            x_tensor = tf.constant (x)
            prob = self.output_layer[i] (self.__SplDense (x_tensor))
            x.append(1 if tf.random.uniform(shape=()) < prob else -1)
        return tf.reshape (tf.constant (x), self.input_shape)
    
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
        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state (loss)
        # Return a dict mapping metric names to current value
        return {"logits": self.loss_tracker.result()}
# %%
