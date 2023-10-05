import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


x_train = np.load("./data2/data.npy")


# GMM 생성
gmm = GaussianMixture(n_components=2)
gmm.fit(x_train)

print(gmm.means_, "\n\n", gmm.covariances_)

x, y = np.meshgrid(np.linspace(-1, 6), np.linspace(-1, 6))
xx = np.array([x.ravel(), y.ravel()]).T
z = gmm.score_samples(xx)
z = z.reshape((50, 50))

plt.contour(x, y, z)
plt.scatter(x_train[:, 0], x_train[:, 1])
plt.show()
