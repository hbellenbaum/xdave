class ModelOptions:

    def __init__(
        self,
        # screening_model,
        # ei_potential,
        # ii_potential,
        # ee_potential,
        polarisation_model="DANDREA_FIT",
        static_structure_factor_approximation="MSA",
        # lfc_model,
        # bf_model,
        # ipd_model,
        # wr_model,
    ):
        # self.screening_model = screening_model
        # self.ei_potential = ei_potential
        # self.ii_potential = ii_potential
        # self.ee_potential = ee_potential
        self.polarisation_model = polarisation_model
        self.static_structure_factor_approximation = static_structure_factor_approximation
        # self.lfc_model = lfc_model
        # self.bf_model = bf_model
        # self.ipd_model = ipd_model
        # self.wr_model = wr_model
