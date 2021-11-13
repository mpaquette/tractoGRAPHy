import numpy as np
import pylab as pl 



import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
# Optionally, you can import packages
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}'





mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'

prob_int_tc = np.load(mainpath+'tc_step_0p5_theta_90_th_0p20_npv_50_seed_int.npy')
graphcone_p = np.load(mainpath+'graph_mat_cone60_com2com_prob.npy')

N = prob_int_tc.shape[0]




graphcone_p_row_norm = graphcone_p / ((graphcone_p - np.eye(N)).sum(axis=1))[:, None]
log10_prob_graph_row_norm = np.log10(graphcone_p_row_norm)

log_tc_plus_1 = np.log10(prob_int_tc+1)



# pl.figure()
# pl.imshow(log10_prob_graph_row_norm)
# pl.xticks([])
# pl.yticks([])
# pl.title(r'$\log_{{10}}{{(\mathbb{{P}}_{{path}})}}$', fontsize=18)
# # pl.show()


pl.figure()
pl.imshow(np.log10(graphcone_p), 'viridis')
pl.xticks([])
pl.yticks([])
pl.title(r'$\log_{{10}}{{(\mathbb{{P}}_{{path}})}}$', fontsize=18)
pl.colorbar()
# pl.show()

pl.savefig('/data/hu_paquette/work/tractoGRAPHy/ismrm2022/images/'+'matrix_log_p.png',
			dpi=300,
			pad_inches=0.25,
			transparent=True,
			bbox_inches='tight')

pl.close()


pl.figure()
pl.imshow(log_tc_plus_1, 'viridis')
pl.xticks([])
pl.yticks([])
pl.title(r'$\log_{{10}}{{(\text{{Streamline}}_{{\text{{count}}}} + 1)}}$', fontsize=18)
pl.colorbar()
# pl.show()

pl.savefig('/data/hu_paquette/work/tractoGRAPHy/ismrm2022/images/'+'matrix_log_tc.png',
			dpi=300,
			pad_inches=0.25,
			transparent=True,
			bbox_inches='tight')

pl.close()
