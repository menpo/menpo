Welcome
=======
**Welcome to the Menpo documentation!**

Menpo is a Python package designed to make manipulating annotated data more
simple. In particular, sparse locations on either images or meshes, referred
to as **landmarks** within Menpo, are tightly coupled with their reference
objects. For areas such as Computer Vision that involve learning models
based on prior knowledge of object location (such as object detection
and landmark localisation), Menpo is a very powerful toolkit.

A short example is often more illustrative than a verbose explanation. Let's
assume that you want to load a set of images that have been annotated with
bounding boxes, and that these bounding box locations live in text files
next to the images. Here's how we would load the images and extract the
areas within the bounding boxes using Menpo:

.. code-block:: python

    import menpo.io as mio

    images = []
    for image in mio.import_images('./images_folder'):
        images.append(image.crop_to_landmarks())

Where :map:`import_images` returns a :map:`LazyList` to keep memory usage low.

Although the above is a very simple example, we believe that being able
to easily manipulate and couple landmarks with images *and* meshes, is an
important problem for building powerful models in areas such as facial
point localisation.

To get started, check out the User Guide for instructions on installation
and some of the core concepts within Menpo.

.. toctree::
  :maxdepth: 2
  :hidden:

  userguide/index
  api/index
