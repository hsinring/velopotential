.. VeloPotential documentation master file, created by
   sphinx-quickstart on Fri Mar  6 16:54:56 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to VeloPotential
=========================
VeloPotential is a Gradient Flow Learning-based computational framework. By parameterizing a scalar potential function via a Multi-Layer Perceptron, VeloPotential directly constructs a continuous global potential landscape by aligning its negative gradient with observed single-cell velocities.


Installation
============
VeloPotential requires Python 3.11 or later. We recommend installing VeloPotential in a separate conda environment.

Create a fresh conda environment:

.. code-block:: bash

   conda create -n velopotential python=3.12 -y
   conda activate velopotential

Install VeloPotential from GitHub using:

.. code-block:: bash

   pip install "git+https://github.com/hsinring/velopotential.git"

Install VeloPotential from PyPI using:

.. code-block:: bash

   # velopotential is not available on PyPI yet, we will upload the package as soon as possible
   # pip install velopotential


.. toctree::
   :maxdepth: 2
   :caption: Main
   :titlesonly:
   :hidden:
   
   Tutorial<pancreas_tutorials>