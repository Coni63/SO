import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = pd.read_csv("mnist_data.csv").as_matrix()

np.random.shuffle(X)

X = X[:1000, :]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 3], cmap=plt.cm.Set1)
plt.show()

