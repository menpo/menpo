.. _menpo-io-import_builtin_asset:

.. currentmodule:: menpo.io.input.base

import_builtin_asset
====================

.. function:: import_builtin_asset()

    This is a dynamically generated method. This method is designed to
    automatically generate import methods for each data file in the ``data``
    folder. This method it designed to be tab completed, so you do not need
    to call this method explicitly. It should be treated more like a property
    that will dynamically generate functions that will import the shipped
    data. For example:

    ::

        import menpo.io as mio
        bb_image = mio.import_builtin_asset.breakingbad_jpg()
