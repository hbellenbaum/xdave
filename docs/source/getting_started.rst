===============
Getting started
===============

Installation
---------------

The code was written using Python (Version 3.12.8) using mostly standard library packages.

.. The packages and versions required are

.. .. code-block:: console

..    numpy >= 1.26
..    matplotlib >= 3.10
..    scipy >= 1.16.2
..    pandas >= 2.2.3

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
These include demonstrations of the individual code components and comparisons against ``MCSS``.


An example on how to run the code in its default options is given here

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

   w = np.linspace(-1000, 1000, 1000)
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


The ``xDAVE`` kernel contains information required for the setup and calculation of the dynamic strucutre factors.
When initialised, it calculates the mean and individual plasma states required to contain the different bound and free states.
Outputs are given in terms of the individual components: bound-free (bf), free-free (ff) and the rayleigh weight.
The DSF output contains the sum of the BF and FF components.
The variable ``bf_i`` is an array of the bound-freec components of each individual state.



Unit testing
-------------

Several unit tests to track changes in the code are included in a separate test suite.
These can be run with python using the command:

.. code-block:: console
    
    python -m pytest tests/

The version tests keep track of any changes made to the models.

.. toctree::
    :maxdepth: 2