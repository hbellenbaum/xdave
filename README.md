# xDaveC

X-Ray Diagnostics, Analysis, Verification and Exploration Code

xDave is an open-source Python code for analysing of x-ray Thomson scattering spectra and obtaining quick estimates of the dynamic structure factor in the chemical picture.

The code uses the Chihara decomposition [1] to split the elastic scattering into bound-free/free-bound and free-free scattering contributions.

It is based on the work by Glenzer et al.[2] and Gregori et al.[3], with the multi-component description taken from Wuensch et al.[4].

## Getting started

More detail to be added.

The code was written using Python (Version 3.12.8) using mostly standard library packages.

The code can be installed by cloning this repo and using the command:

```
cd /path/to/xdave
pip install -e .
```

## Usage

A test suite is included which can be used for building examples.

The `examples` folder contains a few more examples for a simple Be and CH test case.
These give an overview of how the code can be used to derive dynamic structure factors and a spectrum for a given SIF.
This also contains an interface to the x-ray tracing code HEART.
Note that the script `hydrogen_test.py` currently will not run because the relevant PIMC files have not been added to the repo.
The same goes for the comparison test cases against MCSS in the `test` folder.
Sorry for the inconvenience :)

## Build docs locally

To build the documentation locally, one need to start with installation of the respective
dependencies:

```bash
pip install -e .[docs]
```

Afterwards, the docs need to be build. For this one can use `sphinx-build`:

```bash
cd docs
sphinx-build . _build
```

However, it is recommended to use `make` instead:

```bash
cd docs

make html
```

In either way, the generated documentation is placed in the `docs/_build/` folder and can
be previewed by opening `docs/_build/index.html` in the browser of your choice.

## Citation and Attribution

TBA.

## Contributions

All suggestions for improvements or additional models are welcome. Submit a merge request or contact the author: h.bellenbaum@hzdr.de .

## Authors and Acknowledgements

Version 1 was developed by Hannah Bellenbaum ([CASUS], [HZDR]).

_This work was partially supported by the Center for Advanced Systems Understanding ([CASUS]), financed by Germany’s Federal Ministry of Education and Research and the Saxon state government out of the State budget approved by the Saxon State Parliament. This work has received funding from the European Union's Just Transition Fund (JTF) within the project Röntgenlaser-Optimierung der Laserfusion (ROLF), contract number 5086999001, co-financed by the Saxon state government out of the State budget approved by the Saxon State Parliament. This work has received funding from the European Research Council (ERC) under the European Union’s Horizon 2022 research and innovation programme (Grant agreement No. 101076233, "PREXTREME").
Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them._

## References

[1]: Chihara, J. (2000). "Interaction of photons with plasmas and liquid metals - photoabsorption and scattering." _J. Phys. Condens. Matter, 12_

[2]: Glenzer, S., and Redmer, R. (2009). "X-ray Thomson scattering in high energy density plasmas." _Reviews of Modern Physics, 81_

[3]: Gregori, G., et al. (2003). "Theoretical model of x-ray scattering as a dense matter probe." _Physical Review E, 67_

[4]: Wünsch, K., et al. (2008). "Structure of strongly coupled multicomponent plasmas." _Physical Review E, 77_

[CASUS]: https://www.casus.science
[HZDR]: https://www.hzdr.de
