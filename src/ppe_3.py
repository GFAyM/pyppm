import numpy as np
from pyscf import gto, ao2mo, scf
from os import remove, path
from scipy.linalg import expm
import attr


@attr.s
class M_matrix:
	"""[summary]
	Clase para calcular la matriz inversa del propagador principal utilizando
	orbitales moleculares localizados previamente. No se tiene en cuenta
	el término A(0) (que tiene que ver con las energías). La ecuación
	es M = (A(1) +- B(1)). Donde A y B pueden ser las matrices triplete o
	singlete.
	Ref: Aucar G.A., Concepts in Magnetic Resonance,2008,doi:10.1002/cmr.a.20108

	o1 y o2 son los números de "orden" de los LMOs ocupados, a uno y otro lado
	de la molécula.
	v1 y v2 son los números de "orden" de los LMOs desocupados.
	Los LMOs van a estar dados por los coeficientes de constracción mo_coeff,
	donde i,j son los LMOs ocupados y a,b los LMOs desocupados.

	Todos los elementos matriciales son calculados cuando llamamos a la clase,
	luego, debemos llamar a las distintas funciones para obtener las distintas
	funcionalidades.

	La diferencia con la clase Inverse_principal_propagator es que aquí calcula
	solamente el elemento matricial correspondiente a M_{ia,jb}
	Returns
	-------
	[type]
		[description]
	"""

	occ = attr.ib(default=None, type=list)
	vir = attr.ib(default=None, type=list)
	mo_coeff = attr.ib(default=None, type=np.ndarray)
	mol = attr.ib(default=None, type=gto.Mole)
	triplet = attr.ib(default=True, type=bool)
	mo_occ = attr.ib(default=None)
	classical = attr.ib(default=False, type=bool)
	mo_occ = attr.ib(default=None)


	@property
	def fock_matrix_canonical(self):
		self.fock_canonical = self.mf.get_fock()
		return self.fock_canonical

	def __attrs_post_init__(self):
		self.occidx = np.where(self.mo_occ>0)[0]
		self.viridx = np.where(self.mo_occ==0)[0]

		self.orbv = self.mo_coeff[:,self.viridx]
		self.orbo = self.mo_coeff[:,self.occidx]
		self.nocc = self.orbo.shape[1]        
		self.nvir = self.orbv.shape[1]
		#self.mf = scf.RHF(self.mol).run()
		
		self.mo = np.hstack((self.orbo,self.orbv))
		
		self.nmo = self.nocc + self.nvir
		self.m_full = np.zeros((self.nocc,self.nvir,self.nocc,self.nvir))

		eri_mo = ao2mo.general(self.mol, 
				[self.mo,self.mo,self.mo,self.mo], compact=False)
		eri_mo = eri_mo.reshape(self.nmo,self.nmo,self.nmo,self.nmo)
		self.m_full -= np.einsum('ijba->iajb', eri_mo[:self.nocc,:self.nocc,self.nocc:,self.nocc:])
		if self.triplet:
			self.m_full -= np.einsum('jaib->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])
		elif not self.triplet:
			self.m_full += np.einsum('jaib->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])

		self.m_full = self.m_full.reshape((self.nocc*self.nvir,self.nocc*self.nvir))
		#exp_ = np.exp(self.m_full)
		
		exp_ = expm(self.m_full)
		
		self.Z = np.trace(exp_)
		
		self.rho_ = exp_/self.Z
		
	@property
	def entropy_iaia(self):
		"""Entanglement of the M_{ia,jb} matrix:
		M = (M_{ia,ia}  )
			
		Returns
		-------
		[real]
			[value of entanglement]
		"""
		#self.m_iaia = m[:m.shape[0]//4, :m.shape[0]//4] 
		nocc = self.nocc
		orb_a = self.vir[:(len(self.vir)//2)]
		orb_i = self.occ[:(len(self.occ)//2)]
		
		rho = self.rho_.reshape(self.nocc,self.nvir,self.nocc,self.nvir)
		for a in orb_a:
			rho[orb_i,a-nocc,orb_i,a-nocc] = 0
		#self.rho_iaia = np.zeros([len(orb_i),len(orb_a),len(orb_i),len(orb_a)])
		rho_ia = []
		for a in orb_a:
			rho_ia.append(rho[orb_i,a-nocc,:,:].sum())

		print(rho_ia)
		#print(m_ia)
		#eigenvalues = np.linalg.eigvals(rho_iaia)
		#print(eigenvalues)
		#print(rho_ia)
		ent_ia = 0
		for i in rho_ia:
			ent_ia += -i*np.log(i)
		return ent_ia
	
	@property
	def entropy_iajb(self):
		"""Entanglement of the M_{ia,jb} matrix:
		M = (M_{ia,ia}  )
			
		Returns
		-------
		[real]
			[value of entanglement]
		"""
		nocc = self.nocc
		orb_a = self.vir[:(len(self.vir)//2)]
		orb_i = self.occ[:(len(self.occ)//2)]
		orb_b = self.vir[(len(self.vir)//2):]
		orb_j = self.occ[(len(self.occ)//2):]
		rho = self.rho_.reshape(self.nocc,self.nvir,self.nocc,self.nvir)
		m = self.m_full.reshape(self.nocc,self.nvir,self.nocc,self.nvir)
		rho_iajb = np.zeros([len(orb_i),len(orb_a) ,len(orb_i),len(orb_a)])
		rho_jbia = np.zeros([len(orb_i),len(orb_a) ,len(orb_i),len(orb_a)])
		rho_iaia = np.zeros([len(orb_i),len(orb_a) ,len(orb_i),len(orb_a)])
		for a_ord, a in enumerate(orb_a):
			for b_ord, b in enumerate(orb_b):
				#print(m[orb_i[0],a-nocc,orb_j[0],b-nocc],rho[orb_i[0],a-nocc,orb_j[0],b-nocc],a,b)
				#print(rho[orb_i[0],a-nocc,orb_j[0],b-nocc])
				rho_iajb[0,a_ord,0,b_ord] += rho[orb_i[0],a-nocc,orb_j[0],b-nocc]
				rho_jbia[0,b_ord,0,a_ord] += rho[orb_j[0],b-nocc,orb_i[0],a-nocc]
		rho_iajb = rho_iajb.reshape(len(orb_i)*len(orb_a) ,len(orb_i)*len(orb_a))
		rho_jbia = rho_jbia.reshape(len(orb_i)*len(orb_a) ,len(orb_i)*len(orb_a))
		rho_iaia = rho_iaia.reshape(len(orb_i)*len(orb_a) ,len(orb_i)*len(orb_a))

		#print(rho_iajb, rho_jbia, rho_iaia)
		r1 = np.concatenate((rho_iaia,rho_iajb),axis=1)
		r2 = np.concatenate((rho_jbia,rho_iaia),axis=1)
		self.rho_iajb = np.concatenate((r1,r2),axis=0)
		#print(self.rho_iajb)
		eigenvalues = np.linalg.eigvals(self.rho_iajb)
		#print(eigenvalues)
		ent_iajb = 0
		for eig in eigenvalues:
			if eig > 0:    
				ent_iajb += -eig*np.log(eig)
		return ent_iajb


	@property
	def entropy_jbjb(self):
		"""Entanglement of the M_{ia,jb} matrix:
		M = (M_{ia,ia}  )
			
		Returns
		-------
		[real]
			[value of entanglement]
		"""
		nocc = self.nocc
		orb_b = self.vir[(len(self.vir)//2):]
		orb_j = self.occ[(len(self.occ)//2):]
		rho = self.rho_.reshape(self.nocc,self.nvir,self.nocc,self.nvir)
		
		self.rho_jbjb = np.zeros([len(orb_j),len(orb_b),len(orb_j),len(orb_b)])
		for b_ord, b in enumerate(orb_b):
			for b_ordd, bb in enumerate(orb_b):
				self.rho_jbjb[0,b_ord,0,b_ordd] += rho[orb_j[0],b-nocc,orb_j[0],bb-nocc]

		rho_jbjb = self.rho_jbjb.reshape(len(orb_j)*len(orb_b),len(orb_j)*len(orb_b))
		eigenvalues = np.linalg.eigvals(rho_jbjb)
		ent_ia = 0
		for i in eigenvalues:
			if i > 0:
				ent_ia += -i*np.log(i)
		return ent_ia

	@property
	def entropy_iajb_2(self):
		"""Entanglement of the M_{ia,jb} matrix:
		M = (M_{ia,ia}  )
			
		Returns
		-------
		[real]
			[value of entanglement]
		"""
		nocc = self.nocc
		orb_a = self.vir[:(len(self.vir)//2)]
		orb_i = self.occ[:(len(self.occ)//2)]
		orb_b = self.vir[(len(self.vir)//2):]
		orb_j = self.occ[(len(self.occ)//2):]
		rho = self.rho_.reshape(self.nocc,self.nvir,self.nocc,self.nvir)
		
		self.rho_iajb = np.zeros([len(orb_i),len(orb_a) ,len(orb_i),len(orb_a)])
		for a_ord, a in enumerate(orb_a):
			for b_ord, b in enumerate(orb_b):
				print(rho[orb_i[0],a-nocc,orb_j[0],b-nocc])
				self.rho_iajb[0,b_ord,0,a_ord] += rho[orb_i[0],a-nocc,orb_j[0],b-nocc]
		self.rho_iajb = self.rho_iajb.reshape(len(orb_i)*len(orb_a) ,len(orb_i)*len(orb_a))
		print(self.rho_iajb)
		eigenvalues = abs(np.linalg.eigvals(self.rho_iajb))
		print(eigenvalues)
		ent_iajb = 0
		for eig in eigenvalues:    
			ent_iajb += -eig*np.log(eig)
		return ent_iajb
