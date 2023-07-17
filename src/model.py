import tensorflow as tf
import sys

from keras.models import Model
import keras.losses as KL

class DeepFuzzyCMeanModel(Model):

    def __init__(self, m=2.0, norm_factor=2.0, gamma=0.000001, *args, **kwargs):
        super(DeepFuzzyCMeanModel, self).__init__(*args, **kwargs)
        self.m = m
        self.norm_factor = norm_factor
        self.gamma = gamma
        self.c_loss_tracker = tf.keras.metrics.Mean(name="c_loss")
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
    
    # def compile(self, optimizer, loss):
    #     super(DeepFuzzyCMeanModel, self).compile(optimizer=optimizer, loss=loss)
    #     self.clustering_layer.compile(optimizer=optimizer, loss=loss)

    def loss_fn(self, features, label):

        cluster_features = self.get_layer(name='encoder_0')(features)
        cluster_features = self.get_layer(name='encoder_1')(cluster_features)
        cluster_features = self.get_layer(name='encoder_2')(cluster_features)
        cluster_features = self.get_layer(name='encoder_3')(cluster_features)

        decoded_features = self.get_layer(name='decoder_3')(cluster_features)
        decoded_features = self.get_layer(name='decoder_2')(decoded_features)
        decoded_features = self.get_layer(name='decoder_1')(decoded_features)
        decoded_features = self.get_layer(name='decoder_0')(decoded_features)

        @tf.custom_gradient
        def fcm_loss(x, y):

            v = self.get_layer(name='clustering').clusters
            u = self.get_layer(name='clustering')(cluster_features)

            loss_result = tf.reduce_sum(tf.square(tf.expand_dims(x, axis=1) - v), axis=2)
            loss_result = tf.reduce_sum((u ** self.m) * loss_result)
            
            def gradient(upstream_grad, variables):

                variables_grad = []
                assert variables is not None

                A = self.norm_factor
                M = self.m

                x_grad = tf.reduce_sum(
                    (tf.expand_dims(x, axis=1) - v) *
                    (tf.expand_dims(u, axis=2) ** M),
                    axis=1
                ) * (2 * A)
                x_grad = upstream_grad * x_grad

                y_grad = None

                v_grad = tf.reduce_sum(
                    (tf.expand_dims(x, axis=1) - v) *
                    (tf.expand_dims(u, axis=2) ** M),
                    axis=0
                ) * (-2 * A)
                v_grad = upstream_grad * v_grad

                variables_grad.append(v_grad)

                return (x_grad, y_grad, y_grad), variables_grad
            
            return loss_result, gradient
        
        # tf.print(self.compiled_loss(features, decoded_features))

        cluster_loss = self.gamma * fcm_loss(cluster_features, label)
        decoder_loss = self.compiled_loss(features, decoded_features)

        return cluster_loss, decoder_loss
    

    def train_step(self, data):

        x, y = data
        with tf.GradientTape() as tape:
            loss = self.loss_fn(x, y)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        c_loss, d_loss = loss
        self.c_loss_tracker.update_state(c_loss)
        self.d_loss_tracker.update_state(d_loss)
        return {
            "c_loss": self.c_loss_tracker.result(),
            "d_loss": self.d_loss_tracker.result()
        }
