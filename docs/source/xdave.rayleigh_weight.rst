Rayleigh weight
---------------

.. autoclass:: xdave.rayleigh_weight.OCPRayleighWeight
.. autofunction:: xdave.rayleigh_weight.OCPRayleighWeight.get_rayleigh_weight

.. autoclass:: xdave.rayleigh_weight.MCPRayleighWeight
.. autofunction:: xdave.rayleigh_weight.MCPRayleighWeight.get_rayleigh_weight


Static structure factor
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: xdave.static_sf.OCPStaticStructureFactor
.. autofunction:: xdave.static_sf.OCPStaticStructureFactor.get_ii_static_structure_factor
.. autofunction:: xdave.static_sf.OCPStaticStructureFactor.mean_spherical_approximation_ocp_ii
.. autofunction:: xdave.static_sf.OCPStaticStructureFactor.hnc_ocp_ii
.. autofunction:: xdave.static_sf.OCPStaticStructureFactor.xhnc_ocp_ii
.. autofunction:: xdave.static_sf.OCPStaticStructureFactor._hnc_ii_pseudopotential
.. autofunction:: xdave.static_sf.OCPStaticStructureFactor._hnc_bridge_function

.. autoclass:: xdave.static_sf.MCPStaticStructureFactor
.. autofunction:: xdave.static_sf.MCPStaticStructureFactor.get_ab_static_structure_factor
.. autofunction:: xdave.static_sf.MCPStaticStructureFactor.hnc_ab_ss
.. autofunction:: xdave.static_sf.MCPStaticStructureFactor._hnc_pseudopotential


Screening cloud
^^^^^^^^^^^^^^^

.. autoclass:: xdave.screening_cloud.ScreeningCloud
.. autofunction:: xdave.screening_cloud.ScreeningCloud.get_screening_cloud
.. autofunction:: xdave.screening_cloud.ScreeningCloud._debye_huckel_screening_full
.. autofunction:: xdave.screening_cloud.ScreeningCloud._finite_wavelength_screening_full


Form factors
^^^^^^^^^^^^
.. autoclass:: xdave.ii_ff.ScreeningConstants
.. autofunction:: xdave.ii_ff.ScreeningConstants.c1s
.. autofunction:: xdave.ii_ff.ScreeningConstants.c2s
.. autofunction:: xdave.ii_ff.ScreeningConstants.c2p
.. autofunction:: xdave.ii_ff.ScreeningConstants.c3s
.. autofunction:: xdave.ii_ff.ScreeningConstants.c3p
.. autofunction:: xdave.ii_ff.ScreeningConstants.c4s
.. autofunction:: xdave.ii_ff.ScreeningConstants.c3d
.. autofunction:: xdave.ii_ff.ScreeningConstants.get_all_screening_constants



.. autoclass:: xdave.ii_ff.PaulingShermanIonicFormFactor
.. autofunction:: xdave.ii_ff.PaulingShermanIonicFormFactor.calculate_effective_charge_state
.. autofunction:: xdave.ii_ff.PaulingShermanIonicFormFactor.calculate_form_factor



.. toctree::
    :maxdepth: 2