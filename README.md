# PhDdata
Scripts and data for reproducing the results obtained during my PhD

- The data analyzed in this thesis comes from a large number of high-resolution simulations and would require hundreds of terabytes of storage. For this reason, we only keep track of the code and retain the initial conditions of the simulations instead of the actual data.

- The astrophysical code `swift`, which was used to run the simulations used in all chapters of the thesis, is publicly available at: [link](https://swift.strw.leidenuniv.nl/)

- The different neighbour selection algorithms presented in Chapter 3 and the thermal-kinetic model presented in Chapter 4 are publicly available as part of `swift` (see [link](https://github.com/SWIFTSIM/SWIFT/tree/master/src/feedback/EAGLE_thermal) and [link](https://github.com/SWIFTSIM/SWIFT/tree/master/src/feedback/EAGLE_kinetic), respectively)

- The `emulator` package used in Chapter 5 is also publicly available and documented at: [link](https://github.com/SWIFTSIM/emulator)

- All software used to analyze the simulations is publicly available:

  - `VELOCIraptor`: [link](https://velociraptor-stf.readthedocs.io/en/latest/) (used to produce halo catalogues from snapshots of the simulations)
  - `velociraptor-python`: [link](https://github.com/SWIFTSIM/velociraptor-python) (a python library used to read halo catalogues in an easy-to-use way)
  - `swiftsimio`: [link](https://github.com/SWIFTSIM/swiftsimio) (a Python library used to read snapshots in an easy-to-use way)
  - `pipeline-configs`: [link](https://github.com/SWIFTSIM/pipeline-configs) (a repository with relevant scripts that can be used to produce plots for Chapter 5)
  - `velociraptor-comparison-data`: [link](https://github.com/SWIFTSIM/velociraptor-comparison-data) (a repository with observational data used in Chapter 3 and Chapter 5)

- We note, however, that some parts of the COLIBRE model that have been implemented in Swift and were used in this thesis, are not yet publically available. This includes the gravitational instability star formation criterion (Nobels et al., 2023), the subgrid model for the formation and evolution of dust (Trayford et al, in preparation), the prescriptions for stellar early feedback (Ploeckinger et al., in preparation), the prescription for metal diffusion (Correa et al., in preparation). These are expected to become available after the release of the COLIBRE simulations.

