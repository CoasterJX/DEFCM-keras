import numpy as np


def x10dim_vs_yTrue(epoch, x_10dim, y_true, save_path):

    np.savetxt(
        save_path,
        np.append(x_10dim, np.expand_dims(y_true, axis=1), axis=1),
        delimiter=',',
        header='x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,y_true',
        comments=''
    )


def yTrue_vs_yPred_vs_p(y_true, y_pred, p, save_path):

    np.savetxt(
        save_path,
        np.concatenate([
            np.expand_dims(y_true, axis=1).T,
            np.expand_dims(y_pred, axis=1).T,
            np.expand_dims(p, axis=1).T,
        ], axis=0).T,
        delimiter=',',
        header='y_true,y_pred,p',
        comments=''
    )