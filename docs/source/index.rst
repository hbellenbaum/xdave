.. xDAVE documentation master file, created by
   sphinx-quickstart on Tue Mar 24 11:09:57 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================================================================
X-Ray Diagnostics, Analysis, Validation and Evaluation (xDAVE) Code
===================================================================

.. **xDAVE** (X-Ray Diagnostics, Analysis, Validation and Evaluation) 

xDave is an open-source Python code for analysing of x-ray Thomson scattering spectra and obtaining quick estimates of the dynamic structure factor in the chemical picture.
The code uses the Chihara decomposition [#f1]_ to split the elastic scattering into bound-free/free-bound and free-free scattering contributions.
It is based on the work by Glenzer et al. [#f2]_ and Gregori et al. [#f3]_, with the multi-component description taken from Wuensch et al. [#f4]_.

All suggestions for improvements or additional models are welcome. Submit a merge request or contact the author: h.bellenbaum@hzdr.de .

.. [#f1]  Chihara, J. (2000). "Interaction of photons with plasmas and liquid metals - photoabsorption and scattering." *J. Phys. Condens. Matter, 12*
.. [#f2]  Glenzer, S., and Redmer, R. (2009). "X-ray Thomson scattering in high energy density plasmas." *Reviews of Modern Physics, 81*
.. [#f3]  Gregori, G., et al. (2003). "Theoretical model of x-ray scattering as a dense matter probe." *Physical Review E, 67*
.. [#f4]  Wünsch, K., et al. (2008). "Structure of strongly coupled multicomponent plasmas." *Physical Review E, 77*

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.


.. Usage
.. ===================

.. Getting started
.. ---------------

.. The code was written using Python (Version 3.12.8) using mostly standard library packages.

.. The code can be installed by cloning this repo and using the command:

.. .. code-block:: console

..    cd /path/to/xdave
..    pip install -e .


.. Examples
.. ------------

.. A test suite is included which can be used for building examples.

.. The ``examples`` folder contains a few more examples for a simple Be and CH test case.
.. These give an overview of how the code can be used to derive dynamic structure factors and a spectrum for a given SIF.
.. This also contains an interface to the x-ray tracing code HEART.
.. Note that the script ``hydrogen_test.py`` currently will not run because the relevant PIMC files have not been added to the repo.
.. The same goes for the comparison test cases against MCSS in the ``test`` folder.



.. Contributions
.. --------------

.. All suggestions for improvements or additional models are welcome. Submit a merge request or contact the author: h.bellenbaum@hzdr.de .



.. Authors and Acknowledgements
.. ----------------------------

.. Version 1 was developed by Hannah Bellenbaum (`CASUS`_, `HZDR`_).

.. .. _CASUS: https://www.casus.science/
.. .. _HZDR: https://www.hzdr.de/


.. This work was partially supported by the Center for Advanced Systems Understanding ([CASUS]), financed by Germany’s Federal Ministry of Education and Research and the Saxon state government out of the State budget approved by the Saxon State Parliament. This work has received funding from the European Union's Just Transition Fund (JTF) within the project Röntgenlaser-Optimierung der Laserfusion (ROLF), contract number 5086999001, co-financed by the Saxon state government out of the State budget approved by the Saxon State Parliament. This work has received funding from the European Research Council (ERC) under the European Union’s Horizon 2022 research and innovation programme (Grant agreement No. 101076233, "PREXTREME"). 
.. Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.


.. Code structure
.. ==============

.. Components
.. ----------

.. Plasma parameters are stored in a class:

.. .. autoclass:: xdave.plasma_state.PlasmaState


.. The different components comprising the DSF are




.. Run functions
.. -------------


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   manual
   acknowledgements

