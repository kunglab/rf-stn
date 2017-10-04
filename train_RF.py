import matplotlib  # NOQA
matplotlib.use('Agg')  # NOQA

import argparse

from chainer import datasets
from chainer import iterators
from chainer import links as L
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer.training import Trainer
from chainercv.datasets import TransformDataset

import numpy as np
from skimage import transform
from scipy import fftpack
from scipy import signal

from extensions import SamplingGridVisualizer
from models import STCNN

import dataset

img_size = (129, 67)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--n-sample', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def transform_mnist_rts(in_data):
    img, label = in_data
    img = img[0]  # Remove channel axis for skimage manipulation

    # Rotate
    img = transform.rotate(img, angle=np.random.uniform(-45, 45),
                           resize=True, mode='constant')
    #  Scale
    img = transform.rescale(img, scale=np.random.uniform(0.7, 1.2),
                            mode='constant')

    # Translate
    h, w = img.shape
    if h >= img_size[0] or w >= img_size[1]:
        img = transform.resize(img, output_shape=img_size, mode='constant')
        img = img.astype(np.float32)
    else:
        img_canvas = np.zeros(img_size, dtype=np.float32)
        ymin = np.random.randint(0, img_size[0] - h)
        xmin = np.random.randint(0, img_size[1] - w)
        img_canvas[ymin:ymin+h, xmin:xmin+w] = img
        img = img_canvas

    img = img[np.newaxis, :]  # Add the bach channel back
    return img, label

global_max = -float('inf')
global_min = float('inf')

def normalize(x):
    # print x, global_max, global_min
    # print x
    # assert False
    x = (global_max - x)/(global_max - global_min)
    return x

def convertIQtoSpect(in_data, normalize_flag=False):
    sample, label = in_data

    # print "Hola todos: ", sample.shape

    sample = sample.reshape(2, 128)

    samp_rate = 128.

    sample = [t[0] + (1.j)*t[1] for t in sample.T]
    sample = np.array(sample)
    total_pad = 5.
    pad = np.random.uniform(0,total_pad)

    sample = np.hstack((np.zeros(int(pad*samp_rate)), sample, np.zeros(int((total_pad-pad)*samp_rate))))

    t = np.linspace(0, int(sample.shape[0]/samp_rate), sample.shape[0])

    Y = fftpack.fft(sample)/t.shape[0]
    Y = Y[:t.shape[0]/2]
    frq = np.arange(sample.shape[0])/(sample.shape[0]/samp_rate)
    frq = frq[:sample.shape[0]/2]

    t = np.linspace(0, int(sample.shape[0]/samp_rate), sample.shape[0])
    carrier_freq = np.random.randint(5,50) 

    fc = np.real(sample)*np.cos(2.*np.pi*carrier_freq*t) + \
                    np.imag(sample)*np.sin(2.*np.pi*carrier_freq*t)

    f,t_spec,Sxx = signal.spectrogram(fc, fs=samp_rate, nperseg=100, noverlap=90, nfft=256)

    # plt.figure()
    # plt.pcolormesh(t_spec, f, Sxx, cmap='jet')
    Sxx = Sxx.astype('float32')
    Sxx = Sxx.reshape(1,Sxx.shape[0],Sxx.shape[1])
    # print "Aqui: ", Sxx.shape

    if normalize_flag:
        Sxx = normalize(Sxx)

    return Sxx, label



if __name__ == '__main__':
    args = parse_args()

    # train, test = datasets.get_mnist(ndim=3)
    train = dataset.RFModLabeled(class_set=['8PSK', 'QPSK', 'GFSK'], noise_levels=range(6, 20, 2), test=False, snr=False)

    train = datasets.TupleDataset(train.xs, train.ys)


    '''
    for x, y in zip(train._datasets[0], train._datasets[1]):
        s, _ = convertIQtoSpect((x,y), normalize_flag=False)
        if np.max(s) > global_max:
            global_max = np.max(s)
        if np.min(s) < global_min:
            global_min = np.min(s)
    '''

    # print "SET GLOBALS: ", global_max, global_min
    # assert False
    # train = train.xs, train.ys
    train = TransformDataset(train, convertIQtoSpect)
    train_iter = iterators.SerialIterator(train, batch_size=args.batch_size)

    model = L.Classifier(predictor=STCNN())
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = Trainer(updater=updater,
                      stop_trigger=(args.max_iter, 'iteration'), out=args.out)
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/accuracy', 'elapsed_time']),
        trigger=(1, 'iteration'))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'main/accuracy'], trigger=(1, 'iteration')))
    # print args.n_sample
    # print img_size
    # assert False
    x_fixed = np.empty((args.n_sample, 1, img_size[0], img_size[1]), dtype=np.float32)
    for i in range(args.n_sample):
        x_fixed[i] = train[i][0]
    trainer.extend(SamplingGridVisualizer(x_fixed), trigger=(1, 'iteration'))
    trainer.run()
