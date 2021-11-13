import numpy as np
import pylab as pl 
from scipy.stats import spearmanr


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
# Optionally, you can import packages
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}'



mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'


graphcone_p = np.load(mainpath+'graph_mat_cone60_com2com_prob.npy')
graphcone_l = np.load(mainpath+'graph_mat_cone60_com2com_l.npy')

N = graphcone_p.shape[0]

idx_diag = np.diag_indices(N)
idx_triu = np.triu_indices(N, 1)
idx_tril = np.tril_indices(N, -1)
idx_nodiag = (np.concatenate((idx_triu[0], idx_tril[0])),
			  np.concatenate((idx_triu[1], idx_tril[1])))




log10_prob_graph = np.log10(graphcone_p)[idx_nodiag]
len_graph = graphcone_l[idx_nodiag]


pl.figure()
pl.scatter(log10_prob_graph.ravel(), len_graph.ravel(), 
	       alpha=0.3, 
	       color='blue',
	       edgecolors='none')


pl.xlabel(r'$\log_{{10}}{{(\mathbb{{P}}_{{path}})}}$', fontsize=16)
pl.ylabel(r'path length (\# vertex)', fontsize=16)
pl.title('Shortest Path Probability vs Length', fontsize=18)



R = np.corrcoef(log10_prob_graph.ravel(), len_graph.ravel())[0,1]
rho, _ = spearmanr(log10_prob_graph.ravel(), len_graph.ravel())


x_pos_text = -38
y_pos_text = 10
offset_text = 9

pl.text(x_pos_text, y_pos_text, 
	    r'''Pearson's $R = {:.2f}$'''.format(R), fontsize=16)

pl.text(x_pos_text, y_pos_text+offset_text, 
	    r'''Spearman's $\rho = {:.2f}$'''.format(rho), fontsize=16)



pl.savefig('/data/hu_paquette/work/tractoGRAPHy/ismrm2022/images/'+'prob_len_correlation.png',
			dpi=300,
			pad_inches=0.25,
			transparent=True,
			bbox_inches='tight')

pl.close()

# pl.show()





from scipy.stats import gaussian_kde

x = log10_prob_graph.ravel()
y = len_graph.ravel()
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]








pl.figure()
pl.scatter(x, y, c=z, s=50, 
	       alpha=0.5,
	       edgecolors='none')


pl.xlabel(r'$\log_{{10}}{{(\mathbb{{P}}_{{path}})}}$', fontsize=16)
pl.ylabel(r'path length (\# vertex)', fontsize=16)
pl.title('Shortest Path Probability vs Length', fontsize=18)



R = np.corrcoef(log10_prob_graph.ravel(), len_graph.ravel())[0,1]
rho, _ = spearmanr(log10_prob_graph.ravel(), len_graph.ravel())


x_pos_text = -38
y_pos_text = 10
offset_text = 9

pl.text(x_pos_text, y_pos_text, 
	    r'''Pearson's $R = {:.2f}$'''.format(R), fontsize=16)

pl.text(x_pos_text, y_pos_text+offset_text, 
	    r'''Spearman's $\rho = {:.2f}$'''.format(rho), fontsize=16)



pl.savefig('/data/hu_paquette/work/tractoGRAPHy/ismrm2022/images/'+'prob_len_correlation_density.png',
			dpi=300,
			pad_inches=0.25,
			transparent=True,
			bbox_inches='tight')

pl.close()

# pl.show()














