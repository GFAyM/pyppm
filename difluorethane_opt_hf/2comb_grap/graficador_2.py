import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

data_J = pd.read_csv('cloppa_pso_C2H4F2_ccpvdz.txt', sep='\s+', header=None)

data_J.columns = ['ang', 'pp', 'LMOS']


#df = pd.DataFrame(data_J, columns=['angulo', 'PP', 'Occupied'])
fig = px.line(data_J, x="ang", y="pp", animation_frame='LMOS')
fig.update_layout(    yaxis_title=r'SSC Coupling [Hz]' )

fig.write_html("pso_coupling_pathways_C2H4F2.html", include_mathjax='cdn')
