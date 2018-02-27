import tensorflow as tf


# The CNN model has [Conv2d -> Pooling2d]*3 + [FC -> ReLU]*2 + FC
class CNNModel:

    def __init__(self, resolution, channels):
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None] + resolution + [channels],
                                                name='image')
        self.hold_prob = tf.placeholder(tf.float32, shape=[], name='hold_prob')

        x = tf.layers.conv2d(inputs=self.input_placeholder,
                             filters=32,
                             kernel_size=4)
        x = tf.layers.max_pooling2d(inputs=x,
                                    pool_size=2,
                                    strides=2)
        x = tf.layers.conv2d(inputs=x,
                             filters=64,
                             kernel_size=4)
        x = tf.layers.max_pooling2d(inputs=x,
                                    pool_size=2,
                                    strides=2)
        x = tf.layers.conv2d(inputs=x,
                             filters=128,
                             kernel_size=4)
        x = tf.layers.max_pooling2d(inputs=x,
                                    pool_size=2,
                                    strides=2)
        x = tf.layers.conv2d(inputs=x,
                             filters=256,
                             kernel_size=4)
        x = tf.layers.max_pooling2d(inputs=x,
                                    pool_size=2,
                                    strides=2)

        x = tf.reshape(x, shape=[-1, x.shape[1]*x.shape[2]*x.shape[3]])
        x = tf.layers.dense(inputs=x,
                            units=2048,
                            activation=tf.nn.relu)
        x = tf.layers.dense(inputs=x,
                            units=2048,
                            activation=tf.nn.relu)
        # Dropout layer to prevent potential over-fitting
        x = tf.nn.dropout(x, keep_prob=self.hold_prob)

        x = tf.layers.dense(inputs=x,
                            units=2)

        self.predictions = x