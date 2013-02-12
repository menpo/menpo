import numpy as np
from matplotlib import pyplot as plt
plt.interactive(True)

def two_norm(samples):
  return np.sqrt(np.sum(samples*samples, axis=1))

def produce_barycentric_samples(n_samples, plot=False):
  barycentric_samples = np.random.rand(3*n_samples).reshape([-1,3])
  barycentric_samples = (barycentric_samples/np.sum(barycentric_samples,
      axis=1)[..., np.newaxis])
  if plot:
    plt.subplot(1,3,0)
    plt.scatter(barycentric_samples[:,0], barycentric_samples[:,1])
    plt.subplot(1,3,1)
    plt.scatter(barycentric_samples[:,0], barycentric_samples[:,2])
    plt.subplot(1,3,2)
    plt.scatter(barycentric_samples[:,1], barycentric_samples[:,2])
  return barycentric_samples.reshape([-1,3,1])

A = np.array([-0.3,  0.5,-0.6])
B = np.array([ 0.2,  0.8, 0.1])
C = np.array([ 0.1,  0.7,-0.3])
abc = np.vstack([A,B,C])
coordinates = np.array([0,0,1,0,0,1]).reshape(3,2)

b_samples = produce_barycentric_samples(100000)
xy = np.sum(b_samples*coordinates, axis=1)
samples = np.sum(b_samples*abc, axis=1)
z = two_norm(samples)
tri_z = two_norm(abc)

