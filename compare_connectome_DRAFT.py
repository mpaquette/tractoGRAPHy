import numpy as np
import pylab as pl 
from scipy.stats import spearmanr


def scattercorr(dataA, dataB, title='', nonzero=False):

	if nonzero:
		zeroA = dataA==0
		zeroB = dataB==0
		keepAB = np.logical_not(np.logical_or(zeroA, zeroB))
		dataA = dataA[keepAB]
		dataB = dataB[keepAB]

	R2 = np.corrcoef(dataA, dataB)[0,1]**2
	rho, _ = spearmanr(dataA, dataB)

	title += ' R^2 = {:.2f} rho = {:.2f}'.format(R2, rho)
	pl.figure()
	pl.scatter(dataA, dataB)
	pl.title(title)







mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'



# probabilistic interface trackcount
prob_int_tc = np.load(mainpath+'tc_step_0p5_theta_90_th_0p20_npv_50_seed_int.npy')
prob_int_vol = np.load(mainpath+'vol_step_0p5_theta_90_th_0p20_npv_50_seed_int.npy')
prob_int_len = np.load(mainpath+'len_step_0p5_theta_90_th_0p20_npv_50_seed_int.npy')



# ROICOM2ROICOM graph
graph_w_roicom = np.load(mainpath+'graph_mat_roicom2roicom_w.npy')
graph_l_roicom = np.load(mainpath+'graph_mat_roicom2roicom_l.npy')
graph_p_roicom = np.load(mainpath+'graph_mat_roicom2roicom_prob.npy')
graph_g_roicom = np.load(mainpath+'graph_mat_roicom2roicom_geom.npy')


# ROICOM2ROICOM cone-graph
graphcone_w_roicom = np.load(mainpath+'graph_mat_cone60_comroi2comroi_w.npy')
graphcone_l_roicom = np.load(mainpath+'graph_mat_cone60_comroi2comroi_l.npy')
graphcone_p_roicom = np.load(mainpath+'graph_mat_cone60_comroi2comroi_prob.npy')
graphcone_g_roicom = np.load(mainpath+'graph_mat_cone60_comroi2comroi_geom.npy')


# COM2COM cone-graph
graphcone_w_com = np.load(mainpath+'graph_mat_cone60_com2com_w.npy')
graphcone_l_com = np.load(mainpath+'graph_mat_cone60_com2com_l.npy')
graphcone_p_com = np.load(mainpath+'graph_mat_cone60_com2com_prob.npy')
graphcone_g_com = np.load(mainpath+'graph_mat_cone60_com2com_geom.npy')


# COM2COM graph
graph_w_com = np.load(mainpath+'graph_mat_com2com_w.npy')
graph_l_com = np.load(mainpath+'graph_mat_com2com_l.npy')
graph_p_com = np.load(mainpath+'graph_mat_com2com_prob.npy')
graph_g_com = np.load(mainpath+'graph_mat_com2com_geom.npy')


# COM2ALL mean graph
graph_w_comall = np.load(mainpath+'graph_mat_com2all_mean_w.npy')
graph_l_comall = np.load(mainpath+'graph_mat_com2all_mean_l.npy')
graph_p_comall = np.load(mainpath+'graph_mat_com2all_mean_prob.npy')
graph_g_comall = np.load(mainpath+'graph_mat_com2all_mean_geom.npy')







scattercorr(prob_int_tc.ravel(), prob_int_len.ravel(), title='prob-tc-vs-len', nonzero=False)
scattercorr(prob_int_tc.ravel(), prob_int_len.ravel(), title='prob-tc-vs-len (nz)', nonzero=True)
pl.show()



scattercorr(graph_p_com.ravel(), graph_l_com.ravel(), title='com-p-vs-l', nonzero=False)
pl.show()


scattercorr(graph_w_com.ravel(), graph_l_com.ravel(), title='com-w-vs-l', nonzero=False)
pl.show()




scattercorr(graph_g_com.ravel(), graph_l_com.ravel(), title='com-g-vs-l', nonzero=False)
pl.show()





allg = [graph_g_com, graph_g_roicom, graphcone_g_com, graphcone_g_roicom, graph_g_comall]
allname = ['COM', 'COMROI', 'CONECOM', 'CONECOMROI', 'COMALL']

for i in range(len(allg)-1):
	for j in range(i+1, len(allg)):
		scattercorr(allg[i].ravel(), allg[j].ravel(), title='g: {}-{}'.format(allname[i], allname[j]), nonzero=False)

pl.show()







allg = [graph_w_com, graph_w_roicom, graphcone_w_com, graphcone_w_roicom, graph_w_comall]
allname = ['COM', 'COMROI', 'CONECOM', 'CONECOMROI', 'COMALL']

for i in range(len(allg)-1):
	for j in range(i+1, len(allg)):
		scattercorr(allg[i].ravel(), allg[j].ravel(), title='w: {}-{}'.format(allname[i], allname[j]), nonzero=False)

pl.show()





allg = [graph_l_com, graph_l_roicom, graphcone_l_com, graphcone_l_roicom, graph_l_comall]
allname = ['COM', 'COMROI', 'CONECOM', 'CONECOMROI', 'COMALL']

for i in range(len(allg)-1):
	for j in range(i+1, len(allg)):
		scattercorr(allg[i].ravel(), allg[j].ravel(), title='l: {}-{}'.format(allname[i], allname[j]), nonzero=False)

pl.show()







scattercorr(graphcone_p_com.ravel(), prob_int_tc.ravel(), title='conecom-p--vs--prob-tc', nonzero=False)
scattercorr(graphcone_p_com.ravel(), prob_int_tc.ravel(), title='(nz)conecom-p--vs--prob-tc', nonzero=True)
pl.show()





roi_id = 0

observed_tc = prob_int_tc[roi_id]
model_pi = graphcone_p_com[roi_id] # this might actually be wrong and it should be [:,id] /shrug
# remove self node
notself = np.array(range(observed_tc.shape[0]))!=roi_id
observed_tc = observed_tc[notself]
model_pi = model_pi[notself]
# normalize model
model_pi_norm = model_pi / model_pi.sum()
# get event count
N = int(observed_tc.sum())
# get maximum likelihood estimte
obs_maxlik = observed_tc / N
# compute model mean and std
model_mean = N*model_pi_norm
model_std = np.sqrt(N*model_pi_norm*(1-model_pi_norm))
# sort by model
sortidx = np.argsort(model_pi_norm)
model_pi_norm_sorted = model_pi_norm[sortidx]
observed_tc_sorted = observed_tc[sortidx]
obs_maxlik_sorted = obs_maxlik[sortidx]
model_mean_sorted = model_mean[sortidx]
model_std_sorted = model_std[sortidx]



pl.figure()
pl.errorbar(range(observed_tc.shape[0]), y=model_mean_sorted, yerr=(3*model_std_sorted, 3*model_std_sorted), errorevery=1, fmt='x', markersize=4)
pl.plot(range(observed_tc.shape[0]),observed_tc_sorted, '.')
pl.show()



# -2 * np.sum(observed_tc_sorted * np.log(model_pi_norm_sorted / obs_maxlik_sorted))
nz = np.nonzero(observed_tc_sorted)
ratio_stat_nz = -2 * np.sum(observed_tc_sorted[nz] * np.log(model_pi_norm_sorted[nz] / obs_maxlik_sorted[nz]))

ratio_stat_nz_corr = -2 * np.sum(observed_tc_sorted[nz] * np.log((model_pi_norm_sorted[nz]/model_pi_norm_sorted[nz].sum()) / obs_maxlik_sorted[nz]))




# only looking at top of distribution
# defined by 
# model_mean_sorted - 2*model_std_sorted > 1
idx_top = np.where(model_mean_sorted - 2*model_std_sorted > 1)[0][0]

# reweight tops
observed_tc_sorted_top = observed_tc_sorted[idx_top:]
model_pi_norm_sorted_top = model_pi_norm_sorted[idx_top:] / np.sum(model_pi_norm_sorted[idx_top:])
obs_maxlik_sorted_top = obs_maxlik_sorted[idx_top:] / np.sum(obs_maxlik_sorted[idx_top:])

ratio_stat_top = -2 * np.sum(observed_tc_sorted_top * np.log(model_pi_norm_sorted_top / obs_maxlik_sorted_top))


chi2.pdf(ratio_stat_nz, np.sum(nz)-1)
chi2.pdf(ratio_stat_nz_corr, np.sum(nz)-1)
chi2.pdf(ratio_stat_top, observed_tc.shape[0]-idx_top-1)


# correlate prob and l
# correlate cone and no cone
# correlate graph and nograph with subsets
# compute bounds from graph and test on nograph










