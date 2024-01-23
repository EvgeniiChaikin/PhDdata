# PhDdata
Scripts and data for reproducing the results obtained during my PhD research

- The data analyzed in my thesis comes from a large number of high-resolution simulations and would require hundreds of terabytes of storage. For this reason, we only keep track of the code and retain the initial conditions of the simulations instead of the actual data. Together with my published papers, and publicly available code, this should be sufficient to be able to reproduce the results of the thesis.

- The astrophysical code `swift`, which was used to run *all* simulations presented in the thesis, is publicly available at: [link](https://swift.strw.leidenuniv.nl/) (including very extensive documentation)

- The numerical implementation of the different neighbour selection algorithms presented in Chapter 3 is part of `swift` and is publicly available at: [link](https://github.com/SWIFTSIM/SWIFT/tree/master/src/feedback/EAGLE_thermal)

- The numerical implementation of the thermal and kinetic channels of energy injection, in the supernova feedback model presented in Chapter 4, is publicly available at: [link](https://github.com/SWIFTSIM/SWIFT/tree/master/src/feedback/EAGLE_thermal) and [link](https://github.com/SWIFTSIM/SWIFT/tree/master/src/feedback/EAGLE_kinetic), respectively

- The `emulator` package used in Chapter 5 is publicly available and documented at: [link](https://github.com/SWIFTSIM/emulator)

- *All* software used to analyze the simulations is publicly available:

  - `VELOCIraptor`: [link](https://velociraptor-stf.readthedocs.io/en/latest/) (used to produce halo catalogues from snapshots of the simulations)
  - `velociraptor-python`: [link](https://github.com/SWIFTSIM/velociraptor-python) (a python library used to read halo catalogues in an easy-to-use way)
  - `swiftsimio`: [link](https://github.com/SWIFTSIM/swiftsimio) (a Python library used to read snapshots in an easy-to-use way)
  - `pipeline-configs`: [link](https://github.com/SWIFTSIM/pipeline-configs) (a repository with relevant scripts that can be used to produce plots for Chapter 5)
  - `velociraptor-comparison-data`: [link](https://github.com/SWIFTSIM/velociraptor-comparison-data) (a repository with observational data used in Chapter 3 and Chapter 5)


We note, however, that some parts of the COLIBRE model that have been implemented in `swift` and were used in this thesis, are not yet publically available. This includes (i) the gravitational instability star formation criterion (Nobels et al., 2023), (ii) the subgrid model for the formation and evolution of dust (Trayford et al, in preparation), (iii) the prescription for stellar early feedback (Ploeckinger et al., in preparation), (iv) the prescription for metal diffusion (Correa et al., in preparation). These parts of the model are expected to become available after the release of the COLIBRE simulations.

