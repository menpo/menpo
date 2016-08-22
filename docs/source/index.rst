Menpo Documentation
===================

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

.. raw:: html

   <big><b><a href="http://www.menpo.org/installation/">Installation <i class="fa fa-external-link"></i></a></b></big><br>
   Please refer to our detailed <a href="http://www.menpo.org/installation/">installation instructions</a> in <tt><a href="http://www.menpo.org/">menpo.org</a></tt>.
   <br>
   <br>
   <big><b><a href="http://www.menpo.org/menpo/">User Guide <i class="fa fa-external-link"></i></a></b></big><br>
   To get started, check out the <a href="http://www.menpo.org/menpo/">user guide</a> in <code><a href="http://www.menpo.org/">menpo.org</a></code> for an explanation of some of the core concepts within Menpo.
   <br>
   <br>

Finally, please refer to Menpo's :ref:`changelog` for a list of changes per release.


Menpo API
~~~~~~~~~
This section attempts to provide a simple browsing experience for the Menpo
documentation. In Menpo, we use legible docstrings, and therefore, all
documentation should be easily accessible in any sensible IDE (or IPython)
via tab completion. However, this section should make most of the core
classes available for viewing online.

.. toctree::
  :maxdepth: 2

  api/base/index
  api/io/index
  api/image/index
  api/feature/index
  api/landmark/index
  api/math/index
  api/model/index
  api/shape/index
  api/transform/index
  api/visualize/index

.. toctree::
  :maxdepth: 2
  :hidden:

  changelog
