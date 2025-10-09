from models import ModelOptions
from plasma_state import PlasmaState
from constants import ELEMENTARY_CHARGE
from potentials import coulomb_k
from freefree_dsf import FreeFreeDSF


class ScreeningCloud:

    def __init__(self, state: PlasmaState, models: ModelOptions):
        self.state = state
        self.screening_model = models.screening_model
        self.ei_potential = models.ei_potential
        self.kappa_e = 1 / self.state.debye_screening_length(
            ELEMENTARY_CHARGE, self.state.electron_number_density, self.state.electron_temperature
        )

    def get_screening_cloud(self, k, w, lfc=0.0):
        screening = 0.0
        Zi = self.state.ion_charge
        if self.screening_model == "DH":
            screening = self._debye_huckel_screening(self, k, w)
        elif self.screening_model == "FWS":
            screening = self._finite_wavelength_screening(self, k, w, lfc=lfc)
        else:
            raise NotImplementedError(f" Model for the screening cloud: {self.screening_model} not recognised.")

        Vee = coulomb_k(Qa=1, Qb=1, k=k)
        Vei = 0.0
        if self.ei_potential == "COULOMB":
            Vei = coulomb_k(Qa=Zi, Qb=1)
        else:
            raise NotImplementedError(f"Model ")
        G = lfc
        ratio = Vei / Vee
        screening_cloud = -ratio * screening / (k**2 + (1 - G) * screening)

        return screening_cloud

    def _debye_huckel_screening(self, k, w):
        """
        Small wavenumber limit of the screening cloud
        """

        screening_decay = 1 / self.kappa_e**2
        return screening_decay

    def _finite_wavelength_screening(self, k, w, lfc=0.0):
        # TODO(Hannah): figure out what these different potentials are...
        Qa = Qb = 1
        Ueek = coulomb_k(Qa=Qa, Qb=Qb, k=k)
        screening_decay = (
            -(k**2) * FreeFreeDSF(state=self.state).get_dsf(k=k, w=w, lfc=lfc, model="DANDREA_FIT") * Ueek
        )
        return screening_decay


def test():
    return


if __name__ == "__main__":
    test()
