
from constants import ELEMENTARY_CHARGE

def hnc_ab_structure(Za,
                     ne,
                     na,
                     Te,
                     Ti,
                     arr,
                     res,
                     number_of_fft_points,
                     mode_of_operation,
                     ii_potential_model,
                     srr_sigma_parameter,
                     srr_core_power,
                     ion_core_radius,
                     csd_gamma_parameter,
                     csd_core_charge,
                     hnc_mix_fraction):
    
    beta = 1. / (ELEMENTARY_CHARGE * Ti)
    # Raa = mean_sphere_radius()
