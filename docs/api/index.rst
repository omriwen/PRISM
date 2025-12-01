PRISM API Documentation
=======================

Welcome to PRISM (Progressive Reconstruction from Imaging with Sparse Measurements) API documentation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules/core
   modules/models
   modules/utils
   modules/config

Installation
------------

.. code-block:: bash

   git clone <repository>
   cd PRISM
   uv sync

Quick Start
-----------

.. code-block:: python

   from prism.models.networks import ProgressiveDecoder
   from prism.core.instruments import Telescope, TelescopeConfig

   # Create telescope configuration
   config = TelescopeConfig(n_pixels=512, aperture_radius_pixels=20)
   telescope = Telescope(config)

   # Create model
   model = ProgressiveDecoder(input_size=512, output_size=256)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
