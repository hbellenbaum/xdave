class ModelOptions:

    def __init__(
        self,
        ei_potential="YUKAWA",
        ii_potential="YUKAWA",
        ee_potential="YUKAWA",
        polarisation_model="DANDREA_FIT",
        static_structure_factor_approximation="HNC",
        lfc_model="DORNHEIM_ESA",
        bf_model="SCHUMACHER",
        ipd_model="CROWLEY",
        bridge_function="IYETOMI",
        screening_model="FINITE_WAVELENGTH",
        # wr_model,
    ):
        self.ii_potential = ii_potential
        self.ee_potential = ee_potential
        self.ei_potential = ei_potential
        self.screening_model = screening_model
        self.polarisation_model = polarisation_model
        self.static_structure_factor_approximation = static_structure_factor_approximation
        self.lfc_model = lfc_model
        self.bf_model = bf_model
        self.ipd_model = ipd_model
        self.bridge_function = bridge_function
        self.screening_model = screening_model
        # self.wr_model = wr_model
