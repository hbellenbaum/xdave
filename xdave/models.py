from dataclasses import dataclass


@dataclass(slots=True)
class ModelOptions:
    """
    Class containing all the different model options for the code.

    Attributes:
        ei_potential (str): electron-ion potential for the screening cloud, default is YUKAWA
        ii_potential (str): ion-ion potential for the HNC solver, default is YUKAWA
        ee_potential (str): electron-electron potential, default is COULOMB
        ei_collision_frequency (str): electron-ion collision frequency, default is BORN
        polarisation_model (str): electron polarisation model for the free-free dsf, default is DANDREA_FIT
        sf_model (str): model for the ion-ion static structure factor, default is HNC
        lfc_model (str): model for the local field correction, default is DORNEHIM_ESA
        bf_model (str): model for the bound-free dsf, default is SCHUMACHER
        ipd_model (str): ionization potential depression model, default is CROWLEY
        bridge_function (str): model for the bridge function used in the extended HNC, default and only current option is IYETOMI
        screening_model (str): model used in the screening cloud, default is FINITE_WAVELENGTH
    """

    ei_potential: str = "YUKAWA"
    ii_potential: str = "YUKAWA"
    ee_potential: str = "COULOMB"
    ei_collision_frequency: str = "BORN"
    polarisation_model: str = "DANDREA_FIT"
    sf_model: str = "HNC"
    lfc_model: str = "DORNHEIM_ESA"
    bf_model: str = "SCHUMACHER"
    ipd_model: str = "CROWLEY"
    bridge_function: str = "IYETOMI"
    screening_model: str = "FINITE_WAVELENGTH"

    def print_default_options(self):
        print(
            "\nDefault model options\n"
            "---------------\n"
            "free electron dielectric function: DANDREA_FIT\n"
            "bound-free dynamic structure factor: SCHUMACHER\n"
            "ion-ion static structure factor: HNC\n"
            "local field correction: DORNHEIM_ESA\n"
            "ionization potential depression: CROWLEY\n"
            "electron-ion collision frequency: BORN\n"
            "electron-ion potential: YUKAWA\n"
            "ion-ion potential: YUKAWA\n"
            "electron-electron potential: COULOMB\n"
            "screening cloud: FINITE_WAVELENGTH\n"
            "ionic form factor: PAULING_SHERMAN\n"
            "HNC bridge function: IYETOMI\n"
            "---------------\n"
        )

    def print_all_model_options(self):
        print(
            "\nModel options\n"
            "---------------\n"
            "free electron dielectric function: DANDREA_FIT, LINDHARD, NUMERICAL, MERMIN\n"
            "bound-free dynamic structure factor: SCHUMACHER, HR_CORRECTION, TRUNCATED_IA\n"
            "OCP ion-ion static structure factor: HNC, MSA, xHNC\n"
            "MCP ion-ion static structure factor: HNC\n"
            "local field correction: DORNHEIM_ESA, NONE, PADE_INTERP, UI, GV, FARID, USER_DEFINED\n"
            "ionization potential depression: CROWLEY, NONE, STEWART_PYATT, DEBYE_HUCKEL, ECKER_KROLL, ION_SPHERE, USER_DEFINED\n"
            "electron-ion collision frequency: BORN, ZIMAN\n"
            "electron-ion potential: YUKAWA, COULOMB, HARD_CORE, SOFT_CORE\n"
            "ion-ion potential: YUKAWA, DEBYE_HUCKEL, DEUTSCH\n"
            "electron-electron potential: COULOMB, KELBG, SRR, CSD\n"
            "screening cloud: FINITE_WAVELENGTH, DEBYE_HUCKEL\n"
            "ionic form factor: PAULING_SHERMAN\n"
            "HNC bridge function: IYETOMI\n"
            "---------------\n"
        )
