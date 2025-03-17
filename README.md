[![codecov](https://codecov.io/gh/GFAyM/pyppm/branch/main/graph/badge.svg)](https://codecov.io/gh/GFAyM/pyppm)

# Python based Polarization Propagator Methods

PyPPM is an open-source Python package based on PySCF for the calculation of response properties using the Polarization Propagator formalism at different levels of approach and frameworks. Currently, it supports RPA and Higher-RPA calculations of non-relativistic spin-spin couplings. In the next version, we plan to extend it to the Second Order Polarization Propagator Approach (SOPPA) and include relativistic calculations of J-coupling and shielding. 

In the non-relativistic framework, PyPPM can perform calculations using localized molecular orbitals for both occupied and virtual sets, and analyze every coupling pathway contribution. Additionally, this software can calculate quantum entanglement between virtual excitations based on the Principal Propagator, one of the elements of the Polarization Propagator formalism.

## Motivation

Molecular properties can be calculated in the PP formalism as a product of two elements: Perturbators and the Principal Propagator. Perturbators represent how the perturbation is performed in the surrounding of each nucleus, generating virtual excitations. The Principal Propagator depends on the system as a whole and represents how those virtual excitations communicate with each other throughout the molecule.

## Features

PyPPM performs the explicit inverse of the Principal Propagator of a response property and calculates the response as a product of Perturbators and the Principal Propagator. These calculations have been performed in most quantum chemistry software using the CPHF method, which avoids the use of inverse calculations and is more efficient. However, the CPHF method does not allow for the observation and analysis of Principal Propagator matrix elements, which carry an interesting physical meaning: they represent how the virtual excitations communicate with each other throughout the molecular system.


## Authors and Acknowledgment

Daniel F. E. Bajac, under the guidance of Professor Gustavo A. Aucar.  
