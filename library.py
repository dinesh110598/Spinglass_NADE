# %%
import tensorflow as tf
import tensorflow.keras as tfk
# %%
class NADE_orig (tfk.Model):
    def __init__(self, inshape, num_hidden, **kwargs):
        super().__init__(**kwargs)
        self.shape = inshape
        self.D = inshape[0]*inshape[1]
        self.N_h = num_hidden
        self.flatten = tfk.layers.Flatten ()
        self.output_layer = []
        for _ in range(self.D-1):
            self.output_layer.append (tfk.layers.Dense (1,'sigmoid'))

        #Implementing weight sharing between hidden layers
        rng = tf.random.Generator.from_non_deterministic_state ()
        self.kernel = []
        for _ in range (self.D-1):
            self.kernel.append (tf.Variable (rng.uniform(shape=[self.N_h]),
                                trainable=True))
        self.bias = tf.Variable (tf.zeros (shape=(self.N_h,)),
                                trainable=True)
        self.loss_tracker = tfk.metrics.Mean(name="logits")

    def call (self, x):
        """Calculates the probability of sample x
        Args:
            x (int32): Value of input lattice
        """

        def SplDense(x, n):
            """We are using this "layer" instead of regular keras Dense
            layer to facilitate use of common kernel and bias"""
            kernel = tf.stack(self.kernel[:n])
            return tfk.activations.sigmoid(tf.matmul(x, kernel) + self.bias)
        
        x = self.flatten (x)
        h = 0.
        p = []
        p.append (tf.ones ([x.shape[0]])) #The first lattice point is fixed at +1
        for i in range (1,self.D):
            h = SplDense (x[:,:i], i)
            y = tf.squeeze (self.output_layer[i-1] (h))
            p.append (0.5*(1-x[:,i]) + x[:,i]*y)
            #output_layer[i](h) produces the probabilty of x[i]=1
        p = tf.stack (p, axis=-1)
        return tf.reduce_mean (p, axis=-1)
        #Above quantity is the joint probability of the input vector x

    @property
    def metrics(self):
        "Records the metrics that need to be refreshed at the end of every epoch"
        return [self.loss_tracker]

    @tf.function
    def sample (self):
        x = []
        rng = tf.random.Generator.from_non_deterministic_state()
        prob = 1.
        x.append (1.)
        for i in reversed(range (self.D-1)):
            x_tensor = tf.constant ([x])
            prob = self.output_layer[i] (self.__SplDense (x_tensor))
            x.append(1 if tf.random.uniform(shape=()) < prob else -1)
        return tf.reshape (tf.constant (x), self.shape)
    
    @tf.function
    def train_step (self, x):
        with tf.GradientTape() as tape:
            p = self (x)
            loss = -tf.math.log (p)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Updates loss metric by averaging current value with prev ones
        self.loss_tracker.update_state (loss)
        print ("Traced")
        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}

# %%
tf.float64