import numpy as np
import pylab as pl 
from scipy.stats import spearmanr


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
# Optionally, you can import packages
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}'



# load DIPY theta=45deg, step=0.5vox, th = 20% tracto
# load COM to COM cone graph matrix
mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'
# prob_int_tc = np.concatenate([np.genfromtxt(mainpath+'connectome{}.csv'.format(i), delimiter=',')[:,:,None] for i in range(10)], axis=2).sum(axis=2)
prob_int_tc = np.load(mainpath+'tc_step_0p5_theta_90_th_0p20_npv_50_seed_int.npy')
graphcone_p = np.load(mainpath+'graph_mat_cone60_com2com_prob.npy')



N = prob_int_tc.shape[0]


tmp_x = np.zeros(N, dtype=np.int)
tmp_y = np.zeros(N, dtype=np.int)

# compute index of the 2 "half diagonal" from the 2 secondary quadrant
tmp_x[:N//2] = np.arange(0, N//2)
tmp_y[:N//2] = np.arange(N//2, N)
tmp_x[N//2:] = np.arange(N//2, N)
tmp_y[N//2:] = np.arange(0, N//2)

idx_off_diag = (tmp_x, tmp_y)



log_tc_plus_1 = np.log10(prob_int_tc[idx_off_diag]+1)

graphcone_p_row_norm = graphcone_p / ((graphcone_p - np.eye(N)).sum(axis=1))[:, None]
log10_prob_graph_row_norm = np.log10(graphcone_p_row_norm)[idx_off_diag]
# log10_prob_graph_row_norm = np.log10(graphcone_p)[idx_off_diag]


pl.figure()
pl.scatter(log10_prob_graph_row_norm.ravel(), log_tc_plus_1.ravel(), 
	       alpha=0.3, 
	       color='blue',
	       edgecolors='none')


pl.xlabel(r'$\log_{{10}}{{(\mathbb{{P}}_{{path}})}}$', fontsize=16)
pl.ylabel(r'$\log_{{10}}{{(\text{{Streamline}}_{{\text{{count}}}} + 1)}}$', fontsize=16)
pl.title('Connectome weights: Transcallosal pairs', fontsize=18)
# pl.title('Connectome weights: Streamline Count vs Shortest Path Probability', fontsize=18)



R = np.corrcoef(log10_prob_graph_row_norm.ravel(), log_tc_plus_1.ravel())[0,1]
rho, _ = spearmanr(log10_prob_graph_row_norm.ravel(), log_tc_plus_1.ravel())


x_pos_text = -28
y_pos_text = 1.5
offset_text = 0.15
# pl.text(x_pos_text, y_pos_text, 'R^2 = {:.2f}'.format(R2))
pl.text(x_pos_text, y_pos_text, 
	    r'''Pearson's $R = {:.2f}$'''.format(R), fontsize=16)
# pl.text(x_pos_text, y_pos_text+offset_text, 'rho = {:.2f}'.format(rho))
pl.text(x_pos_text, y_pos_text+offset_text, 
	    r'''Spearman's $\rho = {:.2f}$'''.format(rho), fontsize=16)



pl.savefig('/data/hu_paquette/work/tractoGRAPHy/ismrm2022/images/'+'loglog_correlation_tc_p_transcallosal.png',
			dpi=300,
			pad_inches=0.25,
			transparent=True,
			bbox_inches='tight')

pl.close()

# pl.show()








