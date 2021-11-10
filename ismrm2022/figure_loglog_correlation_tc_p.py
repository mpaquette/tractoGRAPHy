import numpy as np
import pylab as pl 
from scipy.stats import spearmanr


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
# Optionally, you can import packages
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}'

# r'$\mathbf{{f}}_a = \mathbf{{f}}_{}$'.format(a)



# load DIPY theta=45deg, step=0.5vox, th = 20% tracto
# load COM to COM cone graph matrix
mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'
# prob_int_tc = np.concatenate([np.genfromtxt(mainpath+'connectome{}.csv'.format(i), delimiter=',')[:,:,None] for i in range(10)], axis=2).sum(axis=2)
prob_int_tc = np.load(mainpath+'tc_step_0p5_theta_90_th_0p20_npv_50_seed_int.npy')
graphcone_p = np.load(mainpath+'graph_mat_cone60_com2com_prob.npy')

N = prob_int_tc.shape[0]

idx_diag = np.diag_indices(N)
idx_triu = np.triu_indices(N, 1)
idx_tril = np.tril_indices(N, -1)
idx_nodiag = (np.concatenate((idx_triu[0], idx_tril[0])),
			  np.concatenate((idx_triu[1], idx_tril[1])))



log_tc_plus_1 = np.log10(prob_int_tc[idx_nodiag]+1)

graphcone_p_row_norm = graphcone_p / ((graphcone_p - np.eye(N)).sum(axis=1))[:, None]
log10_prob_graph_row_norm = np.log10(graphcone_p_row_norm)[idx_nodiag]


pl.figure()
pl.scatter(log10_prob_graph_row_norm.ravel(), log_tc_plus_1.ravel(), 
	       alpha=0.3, 
	       color='blue',
	       edgecolors='none')


pl.xlabel(r'$\log_{{10}}{{(\mathbb{{P}}_{{path}})}}$', fontsize=16)
pl.ylabel(r'$\log_{{10}}{{(\text{{Streamline}}_{{\text{{count}}}} + 1)}}$', fontsize=16)
pl.title('Connectome weights', fontsize=18)
# pl.title('Connectome weights: Streamline Count vs Shortest Path Probability', fontsize=18)



R = np.corrcoef(log10_prob_graph_row_norm.ravel(), log_tc_plus_1.ravel())[0,1]
rho, _ = spearmanr(log10_prob_graph_row_norm.ravel(), log_tc_plus_1.ravel())


x_pos_text = -32
y_pos_text = 3.5
offset_text = 0.4
# pl.text(x_pos_text, y_pos_text, 'R^2 = {:.2f}'.format(R2))
pl.text(x_pos_text, y_pos_text, 
	    r'''Pearson's $R = {:.2f}$'''.format(R), fontsize=16)
# pl.text(x_pos_text, y_pos_text+offset_text, 'rho = {:.2f}'.format(rho))
pl.text(x_pos_text, y_pos_text+offset_text, 
	    r'''Spearman's $\rho = {:.2f}$'''.format(rho), fontsize=16)



pl.savefig('/data/hu_paquette/work/tractoGRAPHy/ismrm2022/images/'+'loglog_correlation_tc_p.png',
			dpi=300,
			pad_inches=0.25,
			transparent=True,
			bbox_inches='tight')

pl.close()

# pl.show()






from scipy.stats import gaussian_kde

x = log10_prob_graph_row_norm.ravel()
y = log_tc_plus_1.ravel()
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
pl.ylabel(r'$\log_{{10}}{{(\text{{Streamline}}_{{\text{{count}}}} + 1)}}$', fontsize=16)
pl.title('Connectome weights', fontsize=18)
# pl.title('Connectome weights: Streamline Count vs Shortest Path Probability', fontsize=18)



R = np.corrcoef(log10_prob_graph_row_norm.ravel(), log_tc_plus_1.ravel())[0,1]
rho, _ = spearmanr(log10_prob_graph_row_norm.ravel(), log_tc_plus_1.ravel())


x_pos_text = -32
y_pos_text = 3.5
offset_text = 0.4
# pl.text(x_pos_text, y_pos_text, 'R^2 = {:.2f}'.format(R2))
pl.text(x_pos_text, y_pos_text, 
	    r'''Pearson's $R = {:.2f}$'''.format(R), fontsize=16)
# pl.text(x_pos_text, y_pos_text+offset_text, 'rho = {:.2f}'.format(rho))
pl.text(x_pos_text, y_pos_text+offset_text, 
	    r'''Spearman's $\rho = {:.2f}$'''.format(rho), fontsize=16)



pl.savefig('/data/hu_paquette/work/tractoGRAPHy/ismrm2022/images/'+'loglog_correlation_tc_p_density.png',
			dpi=300,
			pad_inches=0.25,
			transparent=True,
			bbox_inches='tight')

pl.close()

# pl.show()






