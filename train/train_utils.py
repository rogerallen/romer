# code inspired by
#   https://github.com/metachi/fastaiv2keras/blob/master/fastai/utils.py
# and
#   https://github.com/bckenstler/CLR/blob/master/clr_callback.py
from keras.callbacks import *
from math import pi, cos, ceil

class ExponentialLR(Callback):
    """This callback implements a learning rate that grows exponentially.
    It is used to find a max learning rate.

    After you train this for one epoch, look at the
    self.history['loss'] vs self.history['lr'] to find approximately
    where the negative slope is greatest per the discussion in fast.ai
    course.  http://forums.fast.ai/t/wiki-lesson-1/9398

    Arguments:
        iterations: should be set to (dataset_size//batch_size)
        min_lr: initial learning rate
        max_lr: final learning rate

    """

    def __init__(self, iterations, min_lr=1e-9, max_lr=1.0):
        """iterations = dataset size / batch size"""
        super(ExponentialLR, self).__init__()
        self.lr          = min_lr
        self.max_lr      = max_lr
        self.growth_rate = (max_lr/min_lr)**(1/iterations)
        self.iterations  = 0.
        self.history     = {}

    def on_train_begin(self, logs={}):
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.lr)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.lr)

        self.lr         *= self.growth_rate
        self.lr          = min(self.max_lr,self.lr)
        self.iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class CyclicalCosineWithRestartsLR(Callback):
    """CyclicalCosineWithRestartsLR

    Based on Cosine Annealing from "Snapshot Ensembles: Train 1, get M
    for free." https://openreview.net/pdf?id=BJYwwY9ll

    Arguments:

        iterations_per_cycle: should be set to
                              (dataset_size//batch_size)*N. Where N is
                              the length in epochs for a full
                              iteration high-to-low.
        min_lr: lowest learning rate (0.0 per paper)
        max_lr: highest, starting learning rate (use ExponentialLR to discover)

    """
    def __init__(self, iterations_per_cycle, min_lr=0.0, max_lr=1.0):
        self.min_lr               = min_lr
        self.max_lr               = max_lr
        self.lr                   = max_lr
        self.iterations_per_cycle = iterations_per_cycle
        self.iterations           = 0.
        self.history              = {}

    def on_train_begin(self, logs={}):
        logs = logs or {}
        self.iteration2learning_rate()
        K.set_value(self.model.optimizer.lr, self.lr)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.iteration2learning_rate()
        K.set_value(self.model.optimizer.lr, self.lr)

        self.iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def iteration2learning_rate(self):
        alpha0  = self.max_lr - self.min_lr
        N       = self.iterations_per_cycle
        self.lr = (alpha0/2)*(cos(pi*(self.iterations % N)/N) + 1) + self.min_lr
