from dataclasses import dataclass


@dataclass(slots=True)
class ModelOptions:

    ei_potential: str = "YUKAWA"
    ii_potential: str = "YUKAWA"
    ee_potential: str = "YUKAWA"
    polarisation_model: str = "DANDREA_FIT"
    sf_model: str = "HNC"
    lfc_model: str = "DORNHEIM_ESA"
    bf_model: str = "SCHUMACHER"
    ipd_model: str = "CROWLEY"
    bridge_function: str = "IYETOMI"
    screening_model: str = "FINITE_WAVELENGTH"
