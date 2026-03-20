[![codecov](https://codecov.io/gh/GFAyM/pyppm/branch/main/graph/badge.svg)](https://codecov.io/gh/GFAyM/pyppm)
[![DOI](https://zenodo.org/badge/654602334.svg)](https://doi.org/10.5281/zenodo.15066679)


# PyPPM: Python-based Polarization Propagator Methods

PyPPM is an open-source Python package built on PySCF for calculating response properties using the Polarization Propagator formalism at various levels of approximation and frameworks. It currently supports RPA, Higher-RPA (HRPA) and SOPPA level of approach in calculations of non-relativistic spin-spin couplings. In the relativistic framework, the coupling, NMR shielding and polarizabilities are implemented at TDA and RPA level of approach.

In the non-relativistic framework, PyPPM can perform calculations using localized molecular orbitals for both occupied and virtual sets, and analyze individual coupling pathway contributions. Additionally, this software can compute quantum entanglement between virtual excitations based on the Principal Propagator, a key element of the Polarization Propagator formalism.

## Motivation

Molecular properties within the PP formalism are calculated as a product of two components: Perturbators and the Principal Propagator. Perturbators describe how a perturbation is applied yo the unperturbed system, generating a disturbance. The Principal Propagator matrix, dependent on the overall system, represents how those disturbance interact throughout the molecular system.

## Features

PyPPM performs the explicit inversion of the Principal Propagator for a response property and calculates the response as a product of Perturbators and the Principal Propagator. While most quantum chemistry software uses the more efficient CPHF method, which avoids inverse calculations, PyPPM's approach allows for the observation and analysis of Principal Propagator matrix elements, revealing insights into how virtual excitations communicate within the molecular system.

## Documentation
https://pyppm-quantum.readthedocs.io/en/latest/index.html

## Installation

To install PyPPM, use pip directly from the GitHub repository:

```bash
pip install git+https://github.com/GFAyM/pyppm.git
```

## Future Features

* 4C-SOPPA: Implementation of the Second-Order Polarization Propagator Approach for including relativistic effects.
* DFT in polarization propagator calculations: also known as DFTTDF, both for 1C and 4C frameworks.

## Authors and Acknowledgment

Daniel F. E. Bajac, under the guidance of Professor Gustavo A. Aucar. 
