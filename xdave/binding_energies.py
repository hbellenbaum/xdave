import numpy as np

# K 1s, L1 2s, L2 2p1/2, L3 2p3/2, M1 3s, M2 3p1/2, M3 3p3/2, M4 3d3/2, M5 3d5/2, N1 4s, N2 4p1/2, N3 4p3/2

fully_ionized = np.array([0.000000000000])

# Format is {AN: {Z: {Binding energies}}}

binding_energies = {
    1 : {
        0 : np.array([1.35984345E+01]),
        1 : fully_ionized,
    },

    2 : {
        0 : np.array([2.46911318E+01]),
        1 : np.array([5.44180236E+01]),
        2 : fully_ionized,
    },

    3 : {
        0 : np.array([6.33949235E+01, 5.17085728E+00]),
        1 : np.array([7.57027598E+01]),
        2 : np.array([1.22454941E+02]),
        3 : fully_ionized,
    },

    4 : {
        0 : np.array([1.23150575E+02, 9.04647007E+00]),
        1 : np.array([1.35764868E+02, 1.79860733E+01]),
        2 : np.array([1.53935051E+02]),
        3 : np.array([2.17719462E+02]),
        4 : fully_ionized,
    },

    5 : {
        0 : np.array([2.00320590E+02, 1.21600444E+01, 7.42389640E+00]),
        1 : np.array([2.17226497E+02, 2.48309323E+01]),
        2 : np.array([2.35432099E+02, 3.76963555E+01]),
        3 : np.array([2.59396448E+02]),
        4 : np.array([3.40227258E+02]),
        5 : fully_ionized,
    },

    6 : {
        0 : np.array([2.95896497E+02, 1.55406052E+01, 1.02088752E+01]),
        1 : np.array([3.14950539E+02, 2.98986488E+01, 2.34059588E+01]),
        2 : np.array([3.38727754E+02, 4.75212314E+01]),
        3 : np.array([3.62365890E+02, 6.42519248E+01]),
        4 : np.array([3.92098824E+02]),
        5 : np.array([4.89994897E+02]),
        6 : fully_ionized,
    },
}