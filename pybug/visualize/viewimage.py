import matplotlib.pyplot as plt
import matplotlib.cm as cm


def MatplotLibImageViewer(image):
    if image.shape[-1] == 1:
        return plt.imshow(image[..., 0], cmap=cm.Greys_r)
    else:
        return plt.imshow(image)


def MatplotLibPointCloudViewer2d(points, **kwargs):
    # note that we flip the y,x to account for the fact
    # that we always consider axis 0, axis 1
    return plt.scatter(points[...,1], points[...,0])