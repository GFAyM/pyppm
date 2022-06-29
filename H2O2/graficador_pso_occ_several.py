import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_pso_occ_H2O2_631G.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'pso_ab', 'a', 'b', 'pso']

#####PARA ANALIZAR
#data_J[(data_J.LMOS.str.contains('F3') == True) & (data_J.pp > 1)]
#data_J.LMOS +'*'+ data_J.LMOS
#orb = ['F3_2pz']
#for a in []

occ_lmo = ['O1_1s', 'O2_1s',  'O1_2p1','O2_2p1','O1_2p2', 'O2_2p2','H4_1s','H3_1s','C_C']

for orb1 in occ_lmo:
	for orb2 in occ_lmo:
		df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
								
		if df.pso_ab[abs(df.pso_ab) > 0.5].any():
			#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

			ang = df.ang
			#df_F_C.pso_ab += df.reset_index().pso_ab
#        pso_ab = pso_F_C + pso_ab_reit
			a = df.a
			b = df.b
			pso = df.pso
			pso_ab = df.pso_ab
			
			print(orb1,orb2)
			#plt.plot(ang, DSO, 'ro', label='DSO')
			plt.figure(figsize=(10,8))
			plt.plot(ang, df.pso_ab, 'b^-', label='$^{PSO}J_{ia,jb}$ ' )#f'a={orb1} b={orb2}')
			#plt.plot(ang, DSO, 'm--', label='DSO')
			#plt.plot(ang, pso, 'go-', label='$^{pso}J_{ij}$')
			#plt.plot(ang, psoSD+pso+PSO, 'm--', label='Total')
			#plt.plot(ang, psoSD, 'r+-', label='pso+SD')

			plt.legend()
			plt.ylabel('Hz')
			plt.xlabel('Ángulo diedro')
			plt.suptitle('PSO contribution to $^3J(H-H)_{ia,jb}$ en H$_2$O$_2$, 6-31G**')
			plt.title(f'i={orb1}  j={orb2}')# f'a={orb1}, b={orb2}')
			#plt.set_size_inches(6.5, 6.5)
			plt.show()
			#plt.savefig(f'pso_sum_C2H4F2_ccpvdz.png', dpi=200)


			#fig = px.line(df, x="ang", y="pp")#, animation_frame='LMOS')
			#fig.update_layout(    yaxis_title=r'SSC Coupling [Hz]' )

				#fig.write_html("pso_coupling_pathways_C2H4F2.html", include_mathjax='cdn')
