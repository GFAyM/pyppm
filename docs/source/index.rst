.. pyppm documentation master file, created by
   sphinx-quickstart on Mon Mar 17 11:47:47 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyPPM documentation!
===============================
PyPPM is an open-source Python package built on PySCF for calculating response properties using the Polarization Propagator formalism at various levels of approximation and frameworks. It currently supports RPA and Higher-RPA calculations of non-relativistic spin-spin couplings. Future versions aim to extend this to the Second-Order Polarization Propagator Approach (SOPPA) and include relativistic calculations of J-coupling and shielding.

In the non-relativistic framework, PyPPM can perform calculations using localized molecular orbitals for both occupied and virtual sets, and analyze individual coupling pathway contributions. Additionally, this software can compute quantum entanglement between virtual excitations based on the Principal Propagator, a key element of the Polarization Propagator formalism.

**Motivation**

Molecular properties within the PP formalism are calculated as a product of two components: Perturbators and the Principal Propagator. Perturbators describe how a perturbation is applied around each nucleus, generating virtual excitations. The Principal Propagator, dependent on the overall system, represents how these virtual excitations interact throughout the molecule.

**Features**

PyPPM performs the explicit inversion of the Principal Propagator for a response property and calculates the response as a product of Perturbators and the Principal Propagator. While most quantum chemistry software uses the more efficient CPHF method, which avoids inverse calculations, PyPPM's approach allows for the observation and analysis of Principal Propagator matrix elements, revealing insights into how virtual excitations communicate within the molecular system.

**Installation**

To install PyPPM, use pip directly from the GitHub repository:

```
pip install git+https://github.com/GFAyM/pyppm.git
```

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pyppm
   modules

