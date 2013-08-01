import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation


def MatplotLibImageViewer(image):
    if image.shape[-1] == 1:
        return plt.imshow(image[..., 0], cmap=cm.Greys_r)
    else:
        return plt.imshow(image)


def MatplotLibPointCloudViewer2d(points, **kwargs):
    # note that we flip the y,x to account for the fact
    # that we always consider axis 0, axis 1
    return plt.scatter(points[..., 1], points[..., 0])


def MatplotlibFittingViewer(images, points, **kwargs):
    """
    Designed to show an iterative fitting process. For example, given a list of
    images and points, we can show an animation of the fitting process for
    an active appearance model.

    In the case of something like a Lucas-Kanade fitting process the list of
    points can be empty, or simply less than the number of images, and it will
    just animate the images.

    Assumes the points are passed in (y, x) and so we index in to the points
    as (points[:, 1], points[:, 0]).

    :param images: list of numpy arrays representing images
    :type images: list of 2D ndarrays [M x N]
    :param points: list of numpy array representing points on the face
    :type points: list of 2D ndarrays [M x 2]
    :param kwargs: kwargs to pass through to ``animation.FuncAnimation``
    :return: An animation object
    :rtype: ``animation.FuncAnimation``
    """
    fig = plt.figure()
    im = plt.imshow(images[0], cmap=cm.Greys_r)
    ax = plt.axes()
    # Only animate points if we have any
    if len(points) > 0:
        sh, = ax.plot(points[0][:, 1], points[0][:, 0], 'bo')

    def animate(i):
        im.set_array(images[i])
        if i < len(points):
            sh.set_data(points[i][:, 1], points[i][:, 0])
        return im,

    return animation.FuncAnimation(fig, animate, frames=len(images), **kwargs)