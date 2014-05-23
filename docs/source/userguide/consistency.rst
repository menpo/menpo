.. _ug-consistency:

Working with Images and PointClouds
===================================
Menpo takes an opinionated stance on certain issues - one of which is
establishing sensible rules for how to work with spatial data and image data
in the same framework.

Let's start with a quiz - which of the following is correct?

.. figure:: indexing.jpg

Most would answer **b** - images are indexed from the top left, with ``x`` going
across and ``y`` going down.

Now another question - how do I access that pixel in the pixels array? ::

    a: lenna[30, 50]
    b: lenna[50, 30]

The correct answer is **b** - pixels get stored in a `y, x` order so we have to
flip the points to access the array.

As Menpo blends together use of PointClouds and Images frequently this can
cause a lot of confusion. You might create a :map:`Translation` of ``5`` in the
``y`` direction as the following::

    t = menpo.transform.Translation([0, 5])

And then expect to use it to warp an image::

     img.warp_to(reference_shape, t)

and then some spatial data related to the image::

    t.apply(some_data)

By the above indexing 
The *worst* part about this is that once we go to voxel data (which
:map:`Image` largely supports, and will fully support in the future), adds a
z-axis. Now we drop all the swapping business - and the third axis of the spatial
data once more corresponds with the third axis of the image data.

eg

Menpo's approach
----------------
Menpo's solution to this problem is simple - **drop the insistence of calling
axes x, y, and z!** The zeroth axis of the pixel data is simply that - the
zeroth axis. It corresponds exactly with the zeroth axis on the point cloud.

It's natural to be concerned at this point that establishing such rules must
make it really difficult ingest data into Menpo -

