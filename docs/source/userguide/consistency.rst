.. _ug-consistency:

Indexing into Images
====================
Menpo takes an oppinionated stance on certian issues - one of which is
establishling sensible rules for how to work with spatial data and image data
in the same framework.

Let's start with a test - can you tell me which of the following is correct?


Most of course would answer C. Now another question - how do I access that
pixel in a numpy array?



The correct answer is B, we have to index into the first axis to go down Y,
the second axis being X. So what if we had an annotation at the above point in
the image? Now we have to start doing things like this:

The *worst* part about this is that once we go to voxel data (which
:map:`Image` largely supports, and will fully support in the future), adds a
z-axis. Now we drop all the swapping business - and the third axis of the spatial
data once more corresponds with the third axis of the image data.

eg

Menpo's appraoch
----------------
Menpo's solution to this problem is simple - **drop the insistance of calling
axes x, y, and z!** The zeroth axis of the pixel data is simply that - the
zeroth axis. It corresponds exactly with the zeroth axis on the point cloud.

It's natural to be concerned at this point that establishing such rules must
make it really difficult ingest data into Menpo -

Core Interfaces
---------------
Menpo is an object oriented framework built around a set of core abstract
interfaces, each one governing a single facet of Menpo's design. Menpo's key
interfaces are:

- :map:`Shape` - spatial data containers
- :map:`Vectorizable` - efficient bi-directional conversion of types to a vector representation
- :map:`Viewable` - :map:`view` method for easy interactive visualisation
- :map:`Targetable` - objects that generate some spatial data
- :map:`Transform` - flexible spatial transformations
- :map:`Landmarkable` - objects that can be annotated with spatial labelled landmarks
- :map:`DX`, :map:`DP` & :map:`DL` - derivatives in spatial, parameter, and landmark spaces

Data containers
---------------
Most numerical data in Menpo is passed around in one of our core data
containers. The features of each of the data containers is explained in great
detail in the notebooks - here we just list them to give you a feel for what
to expect:

- :map:`Image` - n-dimensional image with k-channels of data
- :map:`MaskedImage` - As :map:`Image`, but with a boolean mask
- :map:`PointCloud` - n-dimensional ordered point collection
- :map:`TriMesh` - As :map:`PointCloud`, but with a triangulation
