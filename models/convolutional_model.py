import tensorflow as tf


class ConvModel:

    def __init__(self, resolution, channels):
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None] + resolution + [channels],
                                                name='image')

        x = tf.layers.conv2d(inputs=self.input_placeholder,
                             filters=32,
                             kernel_size=5) # 60 124
        x = tf.layers.max_pooling2d(inputs=x,
                                    pool_size=2,
                                    strides=2)   # 30 62
        x = tf.layers.conv2d(inputs=x,
                             filters=64,
                             kernel_size=5) # 26 58
        x = tf.layers.max_pooling2d(inputs=x,
                                    pool_size=2,
                                    strides=2)  # 13 29
# ===================================================
        x = tf.layers.conv2d(inputs=x,
                             filters=128,
                             kernel_size=4) # 10 26
        x = tf.layers.max_pooling2d(inputs=x,
                                    pool_size=2,
                                    strides=2)  # 5 13
# ===================================================
        x = tf.reshape(x, shape=[-1, x.shape[1]*x.shape[2]*x.shape[3]])
        x = tf.layers.dense(inputs=x,
                            units=2048,
                            activation=tf.nn.relu)
        x = tf.layers.dense(inputs=x,
                            units=5)

        self.predictions = x
