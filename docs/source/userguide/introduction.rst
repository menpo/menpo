.. _ug-introduction:

Introduction
============

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

Most numerical data in Menpo is passed around in one of our core data containers:

- :map:`Image` - n-dimensional image with k-channels of data
- :map:`MaskedImage` - As :map:`Image`, but with a boolean mask
- :map:`PointCloud` - n-dimensional ordered point collection
- :map:`TriMesh` - As :map:`PointCloud`, but with a triangulation

This user guide is a general introduction to Menpo, aiming to provide a
bird's eye of Menpo's design. After reading this guide you should be able to
go explore Menpo's extensive Notebooks and not be too suprised by what you see.
We'll start by covering Menpo's apprach to handling data.
