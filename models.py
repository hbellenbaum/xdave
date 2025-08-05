class ModelOptions:

    def __init__(
        self,
        screening_model="DH",
        ei_potential="COULOMB",
        # ii_potential,
        # ee_potential,
        polarisation_model="DANDREA_FIT",
        static_structure_factor_approximation="MSA",
        # lfc_model,
        bf_model="SCHUMACHER",
        # ipd_model,
        # wr_model,
    ):
        # self.ii_potential = ii_potential
        # self.ee_potential = ee_potential
        self.ei_potential = ei_potential
        self.screening_model = screening_model
        self.polarisation_model = polarisation_model
        self.static_structure_factor_approximation = static_structure_factor_approximation
        # self.lfc_model = lfc_model
        self.bf_model = bf_model
        # self.ipd_model = ipd_model
        # self.wr_model = wr_model
