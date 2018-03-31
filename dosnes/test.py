import numpy as np
from scipy.spatial.distance import squareform, pdist
import pandas as pd
from sklearn import datasets
from sinkhorn_knopp import sinkhorn_knopp as skp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris()

X = iris.data
y = iris.target

D = squareform(pdist(X, 'sqeuclidean'))
P = np.exp(-D)

sk = skp.SinkhornKnopp()
P = sk.fit(P)

no_dims = 3
n = X.shape[0]
min_gain = 0.01
momentum = 0.5
final_momentum = 0.8
epsilon = 500
mom_switch_iter = 250
max_iter = 1000

P[np.diag_indices_from(P)] = 0.

P = ( P + P.T )/2

P = np.max(P / np.sum(P), axis=0)

const = np.sum( P * np.log(P) )

ydata = 1e-4 * np.random.random(size=(n, no_dims))

y_incs  = np.zeros(shape=ydata.shape)
gains = np.ones(shape=ydata.shape)

for iter in range(max_iter):
    sum_ydata = np.sum(ydata**2, axis = 1)

    bsxfun_1 = sum_ydata.T + -2*np.dot(ydata, ydata.T)
    bsxfun_2 = sum_ydata + bsxfun_1
    num = 1. / ( 1 + bsxfun_2 )

    num[np.diag_indices_from(num)] = 0.

    Q = np.max(num / np.sum(num), axis=0)

    L = (P - Q) * num

    t =  np.diag( L.sum(axis=0) ) - L
    y_grads = 4 * np.dot( t , ydata )

    gains = (gains + 0.2) * ( np.sign(y_grads) != np.sign(y_incs) ) \
          + (gains * 0.8) * ( np.sign(y_grads) == np.sign(y_incs) )

    # gains[np.where(np.sign(y_grads) != np.sign(y_incs))] += 0.2
    # gains[np.where(np.sign(y_grads) == np.sign(y_incs))] *= 0.8

    gains = np.clip(gains, a_min = min_gain, a_max = None)

    y_incs = momentum * y_incs - epsilon * gains * y_grads
    ydata += y_incs

    ydata -= ydata.mean(axis=0)

    alpha = np.sqrt(np.sum(ydata ** 2, axis=1))
    r_mean = np.mean(alpha)
    ydata = ydata * (r_mean / alpha).reshape(-1, 1)

    if iter == mom_switch_iter:
        momentum = final_momentum

    if iter % 10 == 0:
        cost = const - np.sum( P * np.log(Q) )
        print( "Iteration {} : error is {}".format(iter, cost) )


Y = ydata
Y = Y / np.sqrt(np.sum(Y ** 2, axis=1)).reshape(-1,1)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=y, cmap=plt.cm.Set1)
plt.show()