import numpy as np
import pylab as pl 
from scipy.stats import spearmanr


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
# Optionally, you can import packages
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}'



mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'


graphcone_p = np.load(mainpath+'graph_mat_cone60_com2com_prob.npy')
# graphcone_p = np.load(mainpath+'graph_mat_cone60_comroi2comroi_prob.npy')
graph_p = np.load(mainpath+'graph_mat_com2com_prob.npy')
# graph_p = np.load(mainpath+'graph_mat_roicom2roicom_prob.npy')

N = graphcone_p.shape[0]

idx_diag = np.diag_indices(N)
idx_triu = np.triu_indices(N, 1)
idx_tril = np.tril_indices(N, -1)
idx_nodiag = (np.concatenate((idx_triu[0], idx_tril[0])),
			  np.concatenate((idx_triu[1], idx_tril[1])))




log10_prob_graphcone = np.log10(graphcone_p)[idx_nodiag]
log10_prob_graph = np.log10(graph_p)[idx_nodiag]







pl.figure()
pl.scatter(log10_prob_graphcone.ravel(), log10_prob_graph.ravel(), 
	       alpha=0.3, 
	       color='blue',
	       edgecolors='none')


pl.xlabel(r'$\log_{{10}}{{(\mathbb{{P}}_{{path}}^{{\text{{oriented}}}})}}$', fontsize=16)
pl.ylabel(r'$\log_{{10}}{{(\mathbb{{P}}_{{path}}^{{\text{{plain}}}})}}$', fontsize=16)
pl.title('Plain Graph vs Oriented Graph ', fontsize=18)



R = np.corrcoef(log10_prob_graphcone.ravel(), log10_prob_graph.ravel())[0,1]
rho, _ = spearmanr(log10_prob_graphcone.ravel(), log10_prob_graph.ravel())


x_pos_text = -37
y_pos_text = -10
offset_text = 5

pl.text(x_pos_text, y_pos_text, 
	    r'''Pearson's $R = {:.2f}$'''.format(R), fontsize=16)

pl.text(x_pos_text, y_pos_text+offset_text, 
	    r'''Spearman's $\rho = {:.2f}$'''.format(rho), fontsize=16)



pl.savefig('/data/hu_paquette/work/tractoGRAPHy/ismrm2022/images/'+'cone_vs_plain.png',
			dpi=300,
			pad_inches=0.25,
			transparent=True,
			bbox_inches='tight')

pl.close()

# pl.show()












