.. _ug-visualization:

Visualizing Objects
===================
In Menpo, we take an opinionated stance that data exploration is a key part
of working with visual data. Therefore, we tried to make the mental overhead
of visualizing objects as low as possible. Therefore, we made visualization a
key concept directly on our data containers, rather than requiring extra imports
in order to view your data.

We also took a strong step towards simple visualization of data collections
by integrating some of our core types such as :map:`Image` with visualization
widgets for the IPython notebook.

Visualizing 2D Images
---------------------
Without further ado, a quick example of viewing a 2D image:

.. code-block:: python

    %matplotlib inline  # This is only needed if viewing in an IPython notebook
    import menpo.io as mio

    bb = mio.import_builtin_asset.breakingbad_jpg()
    bb.view()

Viewing the image landmarks:

.. code-block:: python

    %matplotlib inline  # This is only needed if viewing in an IPython notebook
    import menpo.io as mio

    bb = mio.import_builtin_asset.breakingbad_jpg()
    bb.view_landmarks()

Viewing the image with a native IPython widget:

.. code-block:: python

    %matplotlib inline  # This is only needed if viewing in an IPython notebook
    import menpo.io as mio

    bb = mio.import_builtin_asset.breakingbad_jpg()
    bb.view_widget()

Visualizing A List Of 2D Images
-------------------------------
Visualizing lists of images is also incredibly simple if you are using
the IPython notebook:

.. code-block:: python

    %matplotlib inline
    import menpo.io as mio
    from menpo.visualize import visualize_images

    # import_images is a generator, so we must exhaust the generator before
    # we can visualize the list. This is because the widget allows you to
    # jump arbitrarily around the list, which cannot be done with generators.
    images = list(mio.import_images('./path/to/images/*.jpg'))
    visualize_images(images)

Visualizing A 2D PointCloud
---------------------------
Visualizing :map:`PointCloud` objects and subclasses is a very familiar
experience:

.. code-block:: python

    %matplotlib inline
    from menpo.shape import PointCloud
    import numpy as np

    pcloud = PointCloud(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    pcloud.view()

Visualizing In 3D
-----------------
Menpo natively supports 3D objects, such as triangulated meshes, as our
base classes are n-dimensional. However, as viewing in 3D is a much more
complicated experience, we have segregated the 3D viewing package into one
of our sub-packages: Menpo3D.

If you try to view a 3D :map:`PointCloud` without having Menpo3D installed, you
will receive an exception asking you to install it.

Menpo3D also comes with many other complicated pieces of functionality for
3D meshes such as a rasterizer. We recommend you look at Menpo3D if you want
to use Menpo for 3D mesh manipulation.
