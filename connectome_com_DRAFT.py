import numpy as np
import nibabel as nib 

from time import time

import igraph as ig

# from dipy.io.streamline import save_tck, save_trk
# from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram

from utils import mask2vertex, mask_COM, load_graph




mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'
g, vertex2vox, vox2vertex = load_graph(mainpath + 'graph_th_0p20.pkl')


mask_fname = mainpath + 'wmmask.nii'
mask_img = nib.load(mask_fname)
affine = mask_img.affine
mask = mask_img.get_fdata().astype(np.bool)




# load roi mask and clip to WM mask
roipath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/rois/'
from os import listdir
from os.path import isfile, join
roifiles = [f for f in listdir(roipath) if isfile(join(roipath, f))]
rois_fname = [roipath + roifiles[i] for i in range(len(roifiles))]


rois_mask_tmp = [nib.load(fname).get_fdata().astype(np.bool) for fname in rois_fname]
rois_mask = [np.logical_and(roi_mask, mask) for roi_mask in rois_mask_tmp]


rois_vertex = [mask2vertex(roi_mask, vox2vertex) for roi_mask in rois_mask]

# del rois_mask
# del rois_mask_tmp


# compute center-of-mass -ish voxel for each roi
rois_center_vertex = []
for roi_mask in rois_mask:
	rois_center_vertex.append(vox2vertex[mask_COM(roi_mask)])



all_dest = [vert for roi_vertex in rois_vertex for vert in roi_vertex]
all_count = [len(roi_vertex) for roi_vertex in rois_vertex]



# import pickle

# a = {'hello': 'world'}

# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)





# connectome = np.zeros((len(rois_fname), len(rois_fname)))
# connectome_geom = np.zeros((len(rois_fname), len(rois_fname)))



# start_time = time()

# # shortest_paths = []
# shortest_lenghts = []
# shortest_geom = []


# for i_source in range(len(rois_fname)):
# 	print('source is {}'.format(rois_fname[i_source].split('/')[-1][:-7]))
# 	source = rois_center_vertex[i_source]


# 	shortest_path = g.get_shortest_paths(source, 
# 	                                    to=all_dest, 
# 	                                    weights='neg_log', 
# 	                                    mode='out', 
# 	                                    output='vpath')
# 	# shortest_paths.append(shortest_path)
# 	# [len(s) for s in shortest_path]


# 	shortest_lenght = g.shortest_paths(source=source, 
# 	                                    target=all_dest, 
# 	                                    weights='neg_log', 
# 	                                    mode='out')

# 	shortest_lenghts.append(shortest_lenght)


# 	shortest_geom.append(np.exp(-np.array(shortest_lenght[0]) / np.array([len(s) for s in shortest_path])))


# end_time = time()
# print('Elapsed time (all path) = {:.2f} s'.format(end_time - start_time))




# # shortest_lenghts is sum(-log(prob))
# # we want geometric_mean(prob) = -shortest_lenghts / len


# all_count_cum = np.concatenate(([0], np.cumsum(all_count)))


# start_time = time()

# for ii in range(len(rois_fname)):
# 	for jj in range(len(rois_fname)):
# 		# bundle-mean of streamline-cummulative-probability
# 		connectome[ii, jj] = np.mean(np.exp(-np.array(shortest_lenghts[ii][0][all_count_cum[jj]:all_count_cum[jj+1]])))
		
# 		# bundle-mean of streamline-geometric-mean-probability
# 		connectome_geom[ii, jj] = np.mean(shortest_geom[ii][all_count_cum[jj]:all_count_cum[jj+1]])

# end_time = time()
# print('Elapsed time (all path) = {:.2f} s'.format(end_time - start_time))


# import pylab as pl

# pl.figure()
# pl.subplot(2,2,1)
# pl.imshow(connectome)
# pl.title('bundle-mean of streamline-cummulative-probability')
# pl.subplot(2,2,2)
# pl.imshow(np.log(connectome))
# pl.title('LOG-OF bundle-mean of streamline-cummulative-probability')
# pl.subplot(2,2,3)
# pl.imshow(connectome_geom)
# pl.title('bundle-mean of streamline-geometric-mean-probability')
# pl.show()




# connectome_geom_alpha_tmp = np.zeros_like(connectome_geom)
# connectome_geom_alpha = np.zeros_like(connectome_geom)

# connectome_alpha_tmp = np.zeros_like(connectome)
# connectome_alpha = np.zeros_like(connectome)

# order = np.argsort(rois_fname)


# for i in range(connectome_geom.shape[0]):
# 	idx = order[i]
# 	connectome_geom_alpha_tmp[i,:] = connectome_geom[idx,:]
# 	connectome_alpha_tmp[i,:] = connectome[idx,:]


# for j in range(connectome_geom.shape[1]):
# 	idx = order[j]
# 	connectome_geom_alpha[:, j] = connectome_geom_alpha_tmp[:, idx]
# 	connectome_alpha[:, j] = connectome_alpha_tmp[:, idx]


# pl.figure()
# pl.subplot(2,2,1)
# pl.imshow(connectome_alpha)
# pl.title('bundle-mean of streamline-cummulative-probability')
# pl.subplot(2,2,2)
# pl.imshow(np.log(connectome_alpha))
# pl.title('LOG-OF bundle-mean of streamline-cummulative-probability')
# pl.subplot(2,2,3)
# pl.imshow(connectome_geom_alpha)
# pl.title('bundle-mean of streamline-geometric-mean-probability')
# pl.show()








# # start_time = time()

# # all_count_cum = np.concatenate(([0], np.cumsum(all_count)))

# # connectome_mean_log = np.zeros((len(rois_fname), len(rois_fname)))
# # connectome_mean_prob = np.zeros((len(rois_fname), len(rois_fname)))

# # connectome_median_log = np.zeros((len(rois_fname), len(rois_fname)))
# # connectome_median_prob = np.zeros((len(rois_fname), len(rois_fname)))


# # for ii in range(len(rois_fname)):
# # 	for jj in range(len(rois_fname)):
# # 		connectome_mean_log[ii, jj] = np.mean(shortest_lenghts[ii][0][all_count_cum[jj]:all_count_cum[jj+1]])
# # 		connectome_median_log[ii, jj] = np.median(shortest_lenghts[ii][0][all_count_cum[jj]:all_count_cum[jj+1]])
# # 		connectome_mean_prob[ii, jj] = np.mean(np.exp(-np.array(shortest_lenghts[ii][0][all_count_cum[jj]:all_count_cum[jj+1]])))
# # 		connectome_median_prob[ii, jj] = np.median(np.exp(-np.array(shortest_lenghts[ii][0][all_count_cum[jj]:all_count_cum[jj+1]])))



# # end_time = time()
# # print('Elapsed time (all path) = {:.2f} s'.format(end_time - start_time))




# # pl.figure()
# # pl.subplot(2,2,1)
# # pl.imshow(connectome_mean_log)
# # pl.title('connectome_mean_log')
# # pl.subplot(2,2,2)
# # pl.imshow(connectome_median_log)
# # pl.title('connectome_median_log')
# # pl.subplot(2,2,3)
# # # pl.imshow(connectome_mean_prob)
# # pl.imshow(np.log(connectome_mean_prob))
# # pl.title('connectome_mean_prob')
# # pl.subplot(2,2,4)
# # # pl.imshow(connectome_median_prob)
# # pl.imshow(np.log(connectome_median_prob))
# # pl.title('connectome_median_prob')
# # pl.show()




# # # example 
# # qq_path = shortest_paths[0][666]
# # qq_len = shortest_lenghts[0][0][666]
# # np.sum([g.es[g.get_eid(qq_path[i], qq_path[i+1])]['neg_log'] for i in range(len(qq_path)-1)]) / qq_len
# # np.prod([g.es[g.get_eid(qq_path[i], qq_path[i+1])]['prob'] for i in range(len(qq_path)-1)]) / np.exp(-qq_len)









# # start_time = time()

# # streamlines = []
# # for i_dest in range(len(all_dest)):
# # 	streamlines.append(np.array([vertex2vox[v] for v in shortest_paths[0][i_dest]]))

# # end_time = time()
# # print('Elapsed time (convert path to streamlines) = {:.2f} s'.format(end_time - start_time))



# # tgm = StatefulTractogram(
# #                     streamlines=streamlines,
# #                     reference=prob_img,
# #                     space=Space.VOX,
# # 					origin=Origin.NIFTI)



# # fname = mainpath + 'one_source_to_all.tck'
# # save_tck(tgm, fname, bbox_valid_check=False)














# start_time = time()

# shortest_paths_com = []
# # shortest_lenghts = []


# for i_source in range(len(rois_fname)):
# 	print('source is {}'.format(rois_fname[i_source].split('/')[-1][:-7]))
# 	source = rois_center_vertex[i_source]
# 	dest = rois_center_vertex


# 	shortest_path = g.get_shortest_paths(source, 
# 	                                    to=dest, 
# 	                                    weights='neg_log', 
# 	                                    mode='out', 
# 	                                    output='vpath')
# 	shortest_paths_com.append(shortest_path)



# 	# shortest_lenght = g.shortest_paths(source=source, 
# 	#                                     target=dest, 
# 	#                                     weights='neg_log', 
# 	#                                     mode='out')

# 	# shortest_lenghts.append(shortest_lenght)

# end_time = time()
# print('Elapsed time (all path) = {:.2f} s'.format(end_time - start_time))







# [g.es[g.get_eid(shortest_paths_com[0][3][i], shortest_paths_com[0][3][i+1])]['neg_log'] for i in range(len(shortest_paths_com[0][3])-1)]





# [g.es[g.get_eid(qq_path[i], qq_path[i+1])]['neg_log'] for i in range(len(qq_path)-1)]








