import numpy as np
import nibabel as nib 

from time import time

import igraph as ig

from dipy.io.streamline import save_tck
from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram

from utils import mask2vertex, 
                  mask_COM, 
                  load_graph, 
                  compute_shortest_paths_COM2COM, 
                  save_COM2COM_path_as_streamlines




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




start_time = time()
paths, paths_length, weights = compute_shortest_paths_COM2COM(g,
                                                              rois_center_vertex,
                                                              w='neg_log')

end_time = time()
print('Shortest path (COM2COM) = {:.2f} s'.format(end_time - start_time))





## save connectome matrix
matrix_weight = np.array(weights)
matrix_length = np.array(paths_length)
matrix_prob = np.exp(-matrix_weight)
matrix_geom = np.exp(-matrix_weight / matrix_length)


np.save(mainpath+'graph_mat_w.npy', matrix_weight)
np.save(mainpath+'graph_mat_l.npy', matrix_length)
np.save(mainpath+'graph_mat_prob.npy', matrix_prob)
np.save(mainpath+'graph_mat_geom.npy', matrix_geom)




start_time = time()
fname_stl = mainpath + 'shortest_COM2COM.tck'
save_COM2COM_path_as_streamlines(paths, 
                                 vertex2vox, 
                                 ref_img=mask_img, 
                                 fname=fname_stl)
end_time = time()
print('Elapsed time (save path as streamlines) = {:.2f} s'.format(end_time - start_time))








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


















# # start_time = time()

# # streamlines = []
# # for i_dest in range(len(all_dest)):
# #     streamlines.append(np.array([vertex2vox[v] for v in shortest_paths[0][i_dest]]))

# # end_time = time()
# # print('Elapsed time (convert path to streamlines) = {:.2f} s'.format(end_time - start_time))



# # tgm = StatefulTractogram(
# #                     streamlines=streamlines,
# #                     reference=prob_img,
# #                     space=Space.VOX,
# #                     origin=Origin.NIFTI)



# # fname = mainpath + 'one_source_to_all.tck'
# # save_tck(tgm, fname, bbox_valid_check=False)







# all_dest = [vert for roi_vertex in rois_vertex for vert in roi_vertex]
# all_count = [len(roi_vertex) for roi_vertex in rois_vertex]







######## save shortest streamline (all)
######## save smoothed all streamline
######## need to experiment with smoothing param
######## need? to enforce that smoothed still have true pts 







# # # example 
# # qq_path = shortest_paths[0][666]
# # qq_len = shortest_lenghts[0][0][666]
# # np.sum([g.es[g.get_eid(qq_path[i], qq_path[i+1])]['neg_log'] for i in range(len(qq_path)-1)]) / qq_len
# # np.prod([g.es[g.get_eid(qq_path[i], qq_path[i+1])]['prob'] for i in range(len(qq_path)-1)]) / np.exp(-qq_len)





# [g.es[g.get_eid(shortest_paths_com[0][3][i], shortest_paths_com[0][3][i+1])]['neg_log'] for i in range(len(shortest_paths_com[0][3])-1)]

# [g.es[g.get_eid(qq_path[i], qq_path[i+1])]['neg_log'] for i in range(len(qq_path)-1)]



