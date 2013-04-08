pybug
=====

A flexible python framework for working with 3D and 4D facial data.

DEPENDENCIES
============

numpy   - basic mathematical objects
scipy   - linear algebra operations
mayavi2 - visualization



STRUCTURE
=========

ibugMM - the full package
  |
  |- face.py : Face class definition
  |
  |- alignment # methods for aligning a set of source data points to a target shape
  |    |
  |    |- rigid.py : Procrustes alignment class
  |    |   
  |    |- nonrigid.py : TPS class
  |
  |- importer # classes which can import data required for 3DMM construction
       |
       |- models.py : classes for importing 3D models into Face instances
       |
       |- metadata.py : Imports for landmarks, emotion data etc





