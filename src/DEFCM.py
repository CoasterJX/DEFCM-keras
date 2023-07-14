import numpy as np
import keras.backend as K
import skfuzzy as fuzz
import metrics
import os
import csv
import csv_generator

from time import time
from keras.layers import Layer, InputSpec, Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from keras.utils.vis_utils import plot_model
from MulticoreTSNE import MulticoreTSNE as TSNE
from clusterng_layer import ClusteringLayer

from model import DeepFuzzyCMeanModel

os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"

np.float = float
np.bool = bool
np.int = int


def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')



class DEFCM:

    def __init__(self, dims, n_clusters=10, fuzzifier=1.5, m=2.0, norm_factor=2.0, init='glorot_uniform', exp_name='test', new_version=True, improved=True, **kwargs) -> None:

        super(DEFCM, self).__init__()

        self.dims, self.input_dim, self.n_stacks = dims, dims[0], len(dims) - 1
        self.n_clusters = n_clusters
        self.fuzzifier = fuzzifier
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

        # prepare DEFCM model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering', m=m, norm_factor=norm_factor)(self.encoder.output)
        model_output = clustering_layer if not improved else [clustering_layer, self.autoencoder.output]
        if not new_version:
            self.model = Model(inputs=self.encoder.input, outputs=model_output)
        else:
            self.model = DeepFuzzyCMeanModel(inputs=self.encoder.inputs, outputs=model_output, m=m, norm_factor=norm_factor)
            self.model.build(self.encoder.input.shape)

        # set up the name of the experiment, can be the name of dataset
        self.exp_name = exp_name

        # other debugging parameters
        self._cluster_saving_epochs = kwargs.get('cluster_csv_epochs', [])
        self._idefcm = improved
    

    def compile(self, optimizer='sgd', loss='kld'):
        c_loss = loss if not self._idefcm else {
            'clustering': 'kld',
            'decoder_0': 'mse'
        }
        self.model.compile(optimizer=optimizer, loss=c_loss, loss_weights=[0.1, 1] if self._idefcm else None)
        plot_model(self.model, to_file="../image/model-structure.png", show_shapes=True)

    
    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, save_dir='../data/autoencoders'):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ae_path = os.path.join(save_dir, f'ae_weights_{self.exp_name}.h5')
        if os.path.exists(ae_path):
            print(f'{ae_path} found. Skip pretraning.')
            self.autoencoder.load_weights(ae_path)
            return

        print(f'{ae_path} not found. Pretraning start...')
        cb = [callbacks.CSVLogger(os.path.join(save_dir, f'pretrain_log_{self.exp_name}.csv'))]

        self.autoencoder.compile(optimizer=optimizer, loss='mse')
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb, workers=4, use_multiprocessing=True)
        self.autoencoder.save_weights(ae_path)
    

    # TODO: return u itself when new BP is ready
    @staticmethod
    def auxiliary_target_distribution(u):
        w = u ** 2 / u.sum(0)
        return (w.T / w.sum(1)).T
    

    @staticmethod
    def fuzzy_cmean_loss(x, v, u):
        # with open('/Users/wangjianxi/Desktop/DEFCM-keras/test/x.npy', 'wb') as f:
        #     np.save(f, x)
        # with open('/Users/wangjianxi/Desktop/DEFCM-keras/test/v.npy', 'wb') as f:
        #     np.save(f, v)
        # with open('/Users/wangjianxi/Desktop/DEFCM-keras/test/u.npy', 'wb') as f:
        #     np.save(f, u)
        # exit()
        l = np.sum(np.square(np.expand_dims(x, axis=1) - v), axis=2)
        l = np.sum(u * l)
        return l
    

    @staticmethod
    def fuzzy_cmean_gradients(x, v, u):

        A = 2

        x_grad = []
        for i in range(len(x)):
            xi_grad = np.zeros(len(x[i]))
            for j in range(len(v)):
                xi_grad += (2 * A) * (u[i][j] * (x[i] - v[j]))
            x_grad.append(xi_grad)
        x_grad = np.array(x_grad)

        v_grad = []
        for j in range(len(v)):
            vj_grad = np.zeros(len(v[j]))
            for i in range(len(x)):
                vj_grad += (2 * A) * (u[i][j] * (x[i] - v[j]))
            v_grad.append(vj_grad)
        v_grad = np.array(v_grad)
        
        return x_grad, v_grad
    

    def fit(self, x, y=None, max_iter=2e4, batch_size=256, tol=1e-3, update_interval=140, save_dir='../data/exp_result_DEFCM'):

        save_dir = os.path.join(save_dir, self.exp_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            print(f'result for {save_dir} already exists. Skip.')
            return

        # Step 1: initialize cluster centers with FCM
        print(f'Initializing cluster centers with fcm, M = {self.fuzzifier}')
        x_10dim = self.encoder.predict(x, max_queue_size=1)
        [cntr, u_init] = fuzz.cmeans(x_10dim.T, self.n_clusters, self.fuzzifier, error=0.000005, maxiter=1000000)[:2]

        y_pred = np.argmax(u_init, axis=0)
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([cntr])

        # Step 2: start deep clustering
        iter_metrics_f = open(os.path.join(save_dir, 'metrics-per-iter.csv'), 'w')
        iter_metrics_writer = csv.DictWriter(iter_metrics_f, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss', 'time'])
        iter_metrics_writer.writeheader()

        [loss, index, index_array, start_time] = [0, 0, np.arange(x.shape[0]), time()]
        for ite in range(int(max_iter)):

            epoch = ite // update_interval

            if ite % update_interval == 0:

                # get auxiliary target distribution & predicted result
                u = self.model.predict(x, verbose=0)
                u = u[0] if self._idefcm else u
                p = self.auxiliary_target_distribution(u)
                
                y_pred = u.argmax(1)

                # save cluster result for selected epochs
                if epoch in self._cluster_saving_epochs:
                    if not os.path.exists(os.path.join(save_dir, 'epoch-clusters-10dim')):
                        os.makedirs(os.path.join(save_dir, 'epoch-clusters-10dim'))
                    csv_generator.x10dim_vs_yTrue(
                        epoch,
                        self.encoder.predict(x, max_queue_size=1, verbose=0),
                        y,
                        os.path.join(save_dir, 'epoch-clusters-10dim', f'epoch_{epoch}.csv')
                    )
                
                # save metrics result among iters
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if y is not None:
                    [acc, nmi, ari, loss, tick] = [
                        np.round(metrics.acc(y, y_pred), 5),
                        np.round(metrics.nmi(y, y_pred), 5),
                        np.round(metrics.ari(y, y_pred), 5),
                        np.round(loss, 5),
                        time() - start_time
                    ]
                    iter_metrics_writer.writerow(dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=loss, time=tick))
                    print(f'Iter {ite} at {tick}s: acc = {acc}, nmi = {nmi}, ari = {ari} | loss = {loss}, delta = {delta_label}')

                # stop when tolerance threshold reached
                if ite > 0 and delta_label < tol:
                    print('Reached tolerance threshold. Stopping training.')
                    break
        
            # train on batch
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            # print(u[idx][0])
            # x_grad, v_grad = self.fuzzy_cmean_gradients(x_10dim[idx], self.model.get_layer(name='clustering').clusters.numpy(), u[idx])
            # print(x_grad[0], x_grad[1], x_grad[2])
            # print(v_grad)
            # self.model.x_grad, self.model.v_grad = x_grad, v_grad
            # l = self.fuzzy_cmean_loss(x_10dim[idx], self.model.get_layer(name='clustering').clusters, u[idx])
            # print(l)
            # self.model.custom_loss = l
            # print(x_10dim[idx][0], self.model.get_layer(name='clustering').clusters.numpy(), u[idx][0])
            # exit()
            # try:
            loss = self.model.train_on_batch(x=x[idx], y=p[idx] if not self._idefcm else [p[idx], x[idx]])
            # except Exception as e:
            #     print(e)
            #     e0 = list(self.model.get_layer(name='encoder_0').get_weights())
            #     e1 = list(self.model.get_layer(name='encoder_1').get_weights())
            #     e2 = list(self.model.get_layer(name='encoder_2').get_weights())
            #     e3 = list(self.model.get_layer(name='encoder_3').get_weights())

            #     import itertools
                # print(all(list(np.concatenate(e0[0]))), all(list(np.concatenate(e0[1]))))
                # print(all(list(e1[0])), all(list(e1[1])))
                # print(all(list(e2[0])), all(list(e2[1])))
                # print(all(list(e3[0])), all(list(e3[1])))
                # print(all(list(self.model.get_layer(name='encoder_1').get_weights())))
                # print(all(list(self.model.get_layer(name='encoder_2').get_weights())))
                # print(all(list(self.model.get_layer(name='encoder_3').get_weights())))
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
            # exit()

            ite += 1

        iter_metrics_f.close()
        self.model.save_weights(os.path.join(save_dir, 'DEFCM_model_final.h5'))

        # store the final predict result
        q_final = self.model.predict(x, verbose=0)
        q_final = q_final[0] if self._idefcm else q_final
        y_pred_final = q_final.argmax(1)
        p_final = self.auxiliary_target_distribution(q_final)[range(len(y_pred_final)), y_pred_final]
        csv_generator.yTrue_vs_yPred_vs_p(
            y, y_pred_final, p_final, os.path.join(save_dir, 'predict-result.csv')
        )

        return y_pred_final
