import numpy as np
import nibabel as nib 

from time import time

import igraph as ig

from utils import mask2vertex, \
                  mask_COM, \
                  load_graph, \
                  compute_shortest_paths_COM2ALL_w, \
                  compute_shortest_paths_1COM2ALL
                  # save_COM2ALL_path_as_streamlines




mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'
g, vertex2vox, vox2vertex = load_graph(mainpath + 'graph_th_0p20.pkl')


mask_fname = mainpath + 'wmmask.nii'
mask_img = nib.load(mask_fname)
affine = mask_img.affine
mask = mask_img.get_fdata().astype(np.bool)




# load label mask (already intersected with wm mask)
roipath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/all_label.nii.gz'
label_map = nib.load(roipath).get_fdata().astype(np.int)



# this is for list of mask
# rois_vertex = [mask2vertex(roi_mask, vox2vertex) for roi_mask in rois_mask]
# this is for label map
rois_vertex = [mask2vertex(label_map==i, vox2vertex) for i in range(1, label_map.max()+1)]



# compute center-of-mass -ish voxel for each roi
rois_center_vertex = []
# for roi_mask in rois_mask:
#   rois_center_vertex.append(vox2vertex[mask_COM(roi_mask)])
for i in range(1, label_map.max()+1):
    roi_mask = (label_map==i)
    rois_center_vertex.append(vox2vertex[mask_COM(roi_mask)])






# memory buster
# start_time = time()
# paths, paths_length, weights = compute_shortest_paths_COM2ALL(g, 
#                                                               rois_center_vertex, 
#                                                               rois_vertex, 
#                                                               w='neg_log')
# end_time = time()
# print('Shortest path (COM2COM) = {:.2f} s'.format(end_time - start_time))


# weight only but for all
start_time = time()
weights = compute_shortest_paths_COM2ALL_w(g, 
                                           rois_center_vertex, 
                                           rois_vertex, 
                                           w='neg_log')

end_time = time()
print('Shortest path (COM2ALL W) = {:.2f} s'.format(end_time - start_time))



# streamline but for each source separately so we can save and stuff in between
start_time = time()
paths_length = []
for i_source in range(len(rois_center_vertex)):
    print('ROI {}: total runtime = {:.1f} s'.format(i_source, time()-start_time))
    _paths, _paths_length = compute_shortest_paths_1COM2ALL(g, 
                                                          rois_center_vertex[0], 
                                                          rois_vertex, 
                                                          w='neg_log')
    paths_length.append(_paths_length)
    # fuck the _paths

end_time = time()
print('Shortest path (COM2ALL path) = {:.2f} s'.format(end_time - start_time))





import pickle

fname = mainpath + 'shortest_COM2ALL.pkl'
mydict = {'neg_log':weights, 'length':paths_length}
with open(fname, 'wb') as f:
    pickle.dump(mydict, f, protocol=pickle.HIGHEST_PROTOCOL)







## save MEAN connectome matrix

# for each (source, dest) roi pair, we take the mean(neg_log)
matrix_weight = np.zeros((len(rois_center_vertex), len(rois_vertex)))
# for each (source, dest) roi pair, we take the mean(length)
matrix_length = np.zeros((len(rois_center_vertex), len(rois_vertex)))
# for each (source, dest) roi pair, we take the mean(total_prob)
matrix_prob = np.zeros((len(rois_center_vertex), len(rois_vertex)))
# for each (source, dest) roi pair, we take the mean(geom mean prob)
matrix_geom = np.zeros((len(rois_center_vertex), len(rois_vertex)))
for i_source in range(len(rois_center_vertex)):
    for i_dest in range(len(rois_vertex)):
        matrix_weight[i_source][i_dest] = np.mean(weights[i_source][i_dest])
        matrix_length[i_source][i_dest] = np.mean(paths_length[i_source][i_dest])
        matrix_prob[i_source][i_dest] = np.mean(np.exp(-np.array(weights[i_source][i_dest])))
        matrix_geom[i_source][i_dest] = np.mean(np.exp(-np.array(weights[i_source][i_dest])/np.array(paths_length[i_source][i_dest])))


np.save(mainpath+'graph_mat_com2all_mean_w.npy', matrix_weight)
np.save(mainpath+'graph_mat_com2all_mean_l.npy', matrix_length)
np.save(mainpath+'graph_mat_com2all_mean_prob.npy', matrix_prob)
np.save(mainpath+'graph_mat_com2all_mean_geom.npy', matrix_geom)






# start_time = time()
# fname_stl = mainpath + 'shortest_COM2COM.tck'
# save_COM2COM_path_as_streamlines(paths, 
#                                  vertex2vox, 
#                                  ref_img=mask_img, 
#                                  fname=fname_stl)
# end_time = time()
# print('Elapsed time (save path as streamlines) = {:.2f} s'.format(end_time - start_time))













# from scipy.interpolate import splprep, splev
# import pylab as pl


# start_time = time()

# interp_streamlines = []
# for i_stl in range(len(streamlines)):
#     stl = streamlines[i_stl]
#     if len(stl) > 3:
#         tck, u = splprep(x=[stl[:,i] for i in range(stl.shape[1])],
#                          k=3,
#                          s=0)

#         new_points = np.array(splev(np.linspace(0, 1, 100), tck)).T 
#     else:
#         new_points = stl.copy()

#     interp_streamlines.append(new_points)

# end_time = time()
# print('Interp streamlines = {:.2f} s'.format(end_time - start_time))



# tgm = StatefulTractogram(
#                     streamlines=interp_streamlines,
#                     reference=mask_img,
#                     space=Space.VOX,
#                     origin=Origin.NIFTI)



# fname = mainpath + 'shortest_COM2COM_smooth_s_0p0.tck'
# save_tck(tgm, fname, bbox_valid_check=False)



# fig = pl.figure()
# fig.add_subplot(projection='3d')
# pl.plot(stl[:,0], stl[:,1], stl[:,2], label='Streamline')
# pl.plot(new_points[:,0], new_points[:,1], new_points[:,2], label='Interpolated')
# pl.legend()
# pl.show()

