# PyPPM
PyPPM is an open source python package based on PySCF for calculation of response properties using the Polarization Propagator formalism using canonical and previosly localized molecular orbitals through CLOPPA model. So far, only contain RPA calculations of Non-relativistic Spin-Spin Couplings, but our idea is to extend it to Shieldings and EPR, also improve the electronic corrrelation with the second order Polarization Propagator Approach SOPPA, and include Relativistic calculations. Besides, this software is able to perform calculation of quantum entanglement between virtual excitations, based in Principal Propagator, one of the elements of Polarization Propagator formalism,   

# Motivation
Molecular Properties can be calculated, in the PP formalism, as a product of two elements: Perturbators and Principal Propagator. Perturbators represent how the perturbation is perform in the surrounding of each nuclei, generating virtual excitations, and the Principal Propagator depends of the system as a whole and represents how those virtual excitations communicate each other for over all the molecule. 

# Features
Perform the explicit inverse of Principal Propagator of a response property and calculates the response as a product of Perturbators and Principal Propagator. This calculations have been performed using the CPHF method, which avoid use the inverse calculations and is more efficient, but does not allow to observe and analyze Principal Propagator matrix elements, which carry an interesting physical meaning: represents how the virtual excitations communicate each other.

Besides, is the only software, as far we know, that perform calculations of properties using LMOs both for occupied and virtual molecular orbitals. It can be used any set of LMOs. 

