import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_virt_C2H4F2_ccpvdz_ab.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'pp', 'a', 'b']

#####PARA ANALIZAR
#data_J[(data_J.LMOS.str.contains('F3') == True) & (data_J.pp > 1)]
#data_J.LMOS +'*'+ data_J.LMOS
df = data_J[(data_J.LMOS.str.contains('F3_2pz') == True) & (data_J.LMOS.str.contains('F7_2pz') == True)]



#df = pd.DataFrame(data_J, columns=['angulo', 'PP', 'Occupied'])

fig = px.line(data_J, x="ang", y="pp")#, animation_frame='LMOS')
fig.update_layout(    yaxis_title=r'SSC Coupling [Hz]' )

fig.write_html("fc_coupling_pathways_C2H4F2.html", include_mathjax='cdn')
