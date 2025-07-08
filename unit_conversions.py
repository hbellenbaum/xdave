# from numpy import PI
# from constants.physics import ELEMENTARY_CHARGE, BOLTZMANN_CONSTANT
from constants import (
    DIRAC_CONSTANT,
    ELEMENTARY_CHARGE,
    BOLTZMANN_CONSTANT,
    PI,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE_SQR,
    VACUUM_PERMITTIVITY,
    PLANCK_CONSTANT_SQR,
)


###############
# -- Angle -- #
###############

deg_TO_rad = PI / 180.0
rad_TO_deg = 180.0 / PI


################
# -- Energy -- #
################

J_TO_eV = 1.0 / ELEMENTARY_CHARGE
eV_TO_J = ELEMENTARY_CHARGE
RYDBERG_TO_eV = (
    ELECTRON_MASS
    * ELEMENTARY_CHARGE
    * ELEMENTARY_CHARGE_SQR
    / (8 * VACUUM_PERMITTIVITY * VACUUM_PERMITTIVITY * PLANCK_CONSTANT_SQR)
)


################
# -- Length -- #
################

m_TO_cm = 1e2
cm_TO_m = 1e-2

m_TO_mm = 1e3
mm_TO_m = 1e-3

m_TO_um = 1e6
um_TO_m = 1e-6

m_TO_nm = 1e9
nm_TO_m = 1e-9

m_TO_ang = 1e10
ang_TO_m = 1e-10

m_TO_pm = 1e12
pm_TO_m = 1e-12

m_TO_fm = 1e15
fm_TO_m = 1e-15

aB_TO_A = 0.529177249


######################
# -- Mass density -- #
######################

kg_per_m3_TO_g_per_cm3 = 1e-3
g_per_cm3_TO_kg_per_m3 = 1e3

amu_TO_kg = 1.66053906660e-27
kg_TO_amu = 1.0 / amu_TO_kg


########################
# -- Number density -- #
########################

per_m3_TO_per_cm3 = 1e-6
per_cm3_TO_per_m3 = 1e6

per_m3_TO_per_A3 = 1e-30
per_A3_TO_per_m3 = 1e30


###############
# -- Speed -- #
###############

m_per_s_TO_km_per_s = 1e3
km_per_s_TO_m_per_s = 1e-3

m_per_s_TO_cm_per_us = 1e-4
cm_per_us_TO_m_per_s = 1e4


#####################
# -- Temperature -- #
#####################

eV_TO_K = ELEMENTARY_CHARGE / BOLTZMANN_CONSTANT
K_TO_eV = BOLTZMANN_CONSTANT / ELEMENTARY_CHARGE


##############
# -- Time -- #
##############

s_TO_ms = 1e3
ms_TO_s = 1e-3

s_TO_us = 1e6
us_TO_s = 1e-6

s_TO_ns = 1e9
ns_TO_s = 1e-9

s_TO_ps = 1e12
ps_TO_s = 1e-12


################
# -- Volume -- #
################

cm3_TO_m3 = 1e-6
m3_TO_cm3 = 1e6

nm3_TO_m3 = 1e-27
m3_TO_nm3 = 1e27

A3_TO_m3 = 1e-30
m3_TO_A3 = 1e30


################
# -- Energy -- #
################


Hz_TO_eV = DIRAC_CONSTANT / ELEMENTARY_CHARGE
