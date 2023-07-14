import tensorflow as tf
import sys

from keras.models import Model

class DeepFuzzyCMeanModel(Model):

    def __init__(self, m=2.0, norm_factor=2.0, *args, **kwargs):
        super(DeepFuzzyCMeanModel, self).__init__(*args, **kwargs)
        self.m = m
        self.norm_factor = norm_factor
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    

    # def compile(self, optimizer, loss):
    #     super(DeepFuzzyCMeanModel, self).compile(optimizer=optimizer, loss=loss)
    #     self.clustering_layer.compile(optimizer=optimizer, loss=loss)

    def loss_fn(self, features, label):

        cluster_features = self.get_layer(name='encoder_0')(features)
        cluster_features = self.get_layer(name='encoder_1')(cluster_features)
        cluster_features = self.get_layer(name='encoder_2')(cluster_features)
        cluster_features = self.get_layer(name='encoder_3')(cluster_features)

        @tf.custom_gradient
        def fcm_loss(x, y):

            v = self.get_layer(name='clustering').clusters
            u = self.get_layer(name='clustering')(cluster_features)

            loss_result = tf.reduce_sum(tf.square(tf.expand_dims(x, axis=1) - v), axis=2)
            loss_result = tf.reduce_sum(u * loss_result)
            
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
                ) * (-2 * A * 0.1)
                v_grad = upstream_grad * v_grad

                variables_grad.append(v_grad)

                tf.print("x_grad: ", x_grad[0], output_stream=sys.stdout)
                tf.print("v_grad: ", v_grad[0], output_stream=sys.stdout)

                return (x_grad, y_grad), variables_grad
            
            return loss_result, gradient
        
        loss = fcm_loss(cluster_features, label)

        return loss
    

    def train_step(self, data):

        x, y = data
        with tf.GradientTape() as tape:
            loss = self.loss_fn(x, y)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
