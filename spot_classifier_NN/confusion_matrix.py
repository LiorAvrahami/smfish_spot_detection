import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def get_confusion_matrix(ytrue, ypred):
    ytrue, ypred = np.array(ytrue), np.array(ypred)
    assert (len(ytrue.shape) == 1)
    assert (ytrue.shape == ypred.shape)

    res_pred = ypred > 0.5
    res_true = ytrue > 0.5

    tp = np.mean(res_true * res_pred)
    fp = np.mean((1 - res_true) * res_pred)
    fn = np.mean(res_true * (1 - res_pred))
    tn = np.mean((1 - res_true) * (1 - res_pred))

    return tp, fp, fn, tn
