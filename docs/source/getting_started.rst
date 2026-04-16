===============
Getting started
===============

Installation
---------------

The code was written using Python (Version 3.12.8) using mostly standard library packages.

The packages and versions required are

.. code-block:: console

   numpy >= 1.26
   matplotlib >= 3.10
   scipy >= 1.16.2
   pandas >= 2.2.3

``xDAVE`` is currently not registered anywhere. Therefore, to install it, you must build it from source. 

First you must clone the repository

.. code-block:: console

    git clone https://github.com/hbellenbaum/xdave.git

Then you can use ``pip`` to build and install ``xDAVE`` locally: 

.. code-block:: console

   cd /path/to/xdave
   pip install -e .


Examples
------------

A test suite is included which can be used for building examples.

The ``examples`` folder contains a few more examples for a simple Be and CH test case.
These give an overview of how the code can be used to derive dynamic structure factors and a spectrum for a given SIF.
This also contains an interface to the x-ray tracing code HEART.
Note that the script ``hydrogen_test.py`` currently will not run because the relevant PIMC files have not been added to the repo.
The same goes for the comparison test cases against MCSS in the ``test`` folder.


A simple example on how to run the code in its default options is given here

.. code-block:: python3

   from xdave import xDave, ModelOptions
   import numpy as np

   kernel = xDave(
        mass_density=1.2,
        electron_temperature=30,
        ion_temperature=30,
        elements=np.array(["Be", "Be"]),
        charge_states=np.array([2.0, 3.0]),
        partial_densities=np.array([0.4, 0.6]),
        models=ModelOptions(),
        enforce_fsum=False,
        user_defined_inputs=None,
        verbose=True,
        save_to_json=True,
        output_file_name='./test_output.json'
    )

   w = np.linspace(-1000, 1000, 10000)
   bf_tot, ff_tot, dsf, rayleigh_weight, ff_i, bf_i = kernel.run(w=w, angle=75, beam_energy=9.0e3, mode="DYNAMIC")

   spec_energy, inelastic, elastic, spectrum = kernel.convolve_with_sif(
        omega=w,
        bf=bf_tot,
        ff=ff_tot,
        dsf=dsf,
        Wr=rayleigh_weight,
        beam_energy=9.0e3,
        type="GAUSSIAN",
        fwhm=10,
    )



.. toctree::
    :maxdepth: 2