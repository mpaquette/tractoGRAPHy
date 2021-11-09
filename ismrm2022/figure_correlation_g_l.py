import numpy as np
import pylab as pl 
from scipy.stats import spearmanr


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
# Optionally, you can import packages
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}'



mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'


graphcone_g = np.load(mainpath+'graph_mat_cone60_com2com_geom.npy')
graphcone_l = np.load(mainpath+'graph_mat_cone60_com2com_l.npy')

N = graphcone_g.shape[0]

idx_diag = np.diag_indices(N)
idx_triu = np.triu_indices(N, 1)
idx_tril = np.tril_indices(N, -1)
idx_nodiag = (np.concatenate((idx_triu[0], idx_tril[0])),
			  np.concatenate((idx_triu[1], idx_tril[1])))




geom_graph = graphcone_g[idx_nodiag]
len_graph = graphcone_l[idx_nodiag]


pl.figure()
pl.scatter(geom_graph.ravel(), len_graph.ravel(), 
	       alpha=0.3, 
	       color='blue',
	       edgecolors='none')


pl.xlabel(r'$$\sqrt[n]{{\prod_{{i\in\text{{path}}}}^{{n = |\text{{path}}|}} p_i}}$$', fontsize=16)
pl.ylabel(r'path length (\# vertex)', fontsize=16)
pl.title('Geometric Mean of Path Probability vs Length', fontsize=18)



R = np.corrcoef(geom_graph.ravel(), len_graph.ravel())[0,1]
rho, _ = spearmanr(geom_graph.ravel(), len_graph.ravel())


x_pos_text = 0.15
y_pos_text = 110
offset_text = 7

pl.text(x_pos_text, y_pos_text, 
	    r'''Pearson's $R = {:.2f}$'''.format(R), fontsize=16)

pl.text(x_pos_text, y_pos_text+offset_text, 
	    r'''Spearman's $\rho = {:.2f}$'''.format(rho), fontsize=16)



pl.savefig('/data/hu_paquette/work/tractoGRAPHy/ismrm2022/images/'+'geom_len_correlation.png',
			dpi=300,
			pad_inches=0.25,
			transparent=True,
			bbox_inches='tight')

pl.close()

# pl.show()












