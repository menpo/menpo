.. _ug-vectorizing:

Vectorizing Objects
-------------------

Computer Vision algorithms are frequently formulated as linear algebra problems
in a high dimensional space, where each asset is stripped into a vector.
In this high dimensional space we may perform any number of operations,
but normally we can't stay in this space for the whole algorithm - we normally
have to recast the vector back into it's original domain in order to perform
other operations. An example of this might be seen with images, where the
gradient of the intensity values of an image needs to be taken. This is a
complex problem to solve in a vector space representation of the image, but
trivial to solve in the image domain.


Menpo bridges the gap by naively supporting bi-directional vectorisation of
it's types through the :map:`Vectorizable` interface. Through this, any type can be
safely and efficiently converted to a vector form and back again.
