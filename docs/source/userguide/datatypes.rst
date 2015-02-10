.. _ug-datatypes:

Menpo's Data Types
==================
Menpo is a high level software package. It is not a replacement for scikit-image,
scikit-learn, or opencv - it ties all these types of packages together in to a
unified framework for building and fitting deformable models. As a result, most
of our algorithms take as input a higher level representation of data than
simple numpy arrays.


Why have data types - what's wrong with numpy arrays?
-----------------------------------------------------
Menpo's data types are thin wrappers around numpy arrays. They give semantic
meaning to the underlying array through providing clearly named and consistent
properties. As an example let's take a look at :map:`PointCloud`, Menpo's
workhorse for spatial data. Construction requires a numpy array::

    x = np.random.rand(3, 2)
    pc = PointCloud(x)

It's natural to ask the question:

    `Is this a collection of three 2D points, or two 3D points?`

In Menpo, you never do this - just look at the properties on the
pointcloud::

    pc.n_points  # 3
    pc.n_dims    # 2

If we take a look at the properties we can see they are trivial::

    @property
    def n_points(self):
        return self.points.shape[0]

    @property
    def n_dims(self):
        return self.points.shape[1]

Using these properties makes code much more readable in algorithms accepting
Menpo's types. Let's imagine a routine that does some operation on an image
and a related point cloud. If it accepted numpy arrays, we might see something
like this on the top line::


    def foo_arrays(x, img):
        # preallocate the result
        y = np.zeros(x.shape[1],
                     x.shape[2],
                     img.shape[-1])
        ...

On first glance it is not at all apparent what ``y``'s shape is semantically.
Now let's take a look at the equivalent code using Menpo's types::

    def foo_menpo(pc, img):
        # preallocate the result
        y = np.zeros(pc.n_dims,
                     pc.n_points,
                     img.n_channels)
        ...

This time it's immediately apparent what ``y``'s shape is. Although this is a
somewhat contrived example, you will find this pattern applied consistently
across Menpo, and it aids greatly in keeping the code readable.

Key points
----------
1. **Containers store the underlying numpy array in an easy to access
attribute.** For the :map:`PointCloud` family see the ``.points`` attribute. On
:map:`Image` and subclasses, the actual data array is stored at ``.pixels``.

2. **Importing assets though** :ref:`api-io-index` **will result in our data
containers, not numpy arrays**. This means in a lot of situations you never
need to remember the Menpo conventions for ordering of array data - just ask
for an image and you will get an :map:`Image` object.

3. **All containers copy data by default**. Look for the ``copy=False`` keyword
argument if you want to avoid copying a large numpy array for performance.

4. **Containers perform sanity checks**. This helps catch obvious bugs like
misshaping an array. You can sometimes suppress them for extra performance with
the ``skip_checks=True`` keyword argument.
