import matplotlib.pyplot as plt
import matplotlib.cm as cm


def MatplotLibImageViewer(image):
    if image.shape[2] == 1:
        return plt.imshow(image[..., 0], cmap=cm.Greys_r)
    else:
        return plt.imshow(image)


def MatplotLibPointCloudViewer2d(points, **kwargs):
    return plt.scatter(points[...,0], points[...,1])