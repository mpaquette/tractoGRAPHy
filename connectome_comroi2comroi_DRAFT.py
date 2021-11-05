import numpy as np
import nibabel as nib 

from time import time

import igraph as ig

from utils import mask2vertex, \
                  mask_COM, \
                  load_graph, \
                  compute_shortest_paths_COM2COM, \
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





start_time = time()
# compute center-of-mass -ish voxel for each roi
# add nodes
g.add_vertices(['ROI_{}'.format(i) for i in range(1, label_map.max()+1)])

rois_center_vertex = []
edges_to_add = []
for i in range(1, label_map.max()+1):
    # ROI_vertex_cone = [vert for voxvert in rois_vertex_cone[i-1] for vert in voxvert]
    ROI_vertex = rois_vertex[i-1]
    # add new vertex to converge them all there
    new_vert_id = g.vs['name'].index('ROI_{}'.format(i))
    rois_center_vertex.append(new_vert_id)
    # create IN and OUT edge for all node at COM
    edges_to_add += [(new_vert_id, i_vert) for i_vert in ROI_vertex]
    edges_to_add += [(i_vert, new_vert_id) for i_vert in ROI_vertex]


merge_w = 10000 # remove twice from computed path weight

# edge of zero could give loops
# instead we put very very expensive nodes, and we can remove it when counting
g.add_edges(edges_to_add, 
            {'neg_log':[merge_w]*len(edges_to_add)})
end_time = time()
print('Make super-ROI = {:.2f} s'.format(end_time - start_time))











start_time = time()
paths_uncorr, paths_length_uncorr, path_weights_uncorr = compute_shortest_paths_COM2COM(g,
                                                              rois_center_vertex,
                                                              w='neg_log')

end_time = time()
print('Shortest path (COM2COM) = {:.2f} s'.format(end_time - start_time))















## correct values and
## save connectome matrix
matrix_weight = np.array(path_weights_uncorr)
matrix_weight[np.triu_indices(matrix_weight.shape[0],1)] -= 2*merge_w
matrix_weight[np.tril_indices(matrix_weight.shape[0],-1)] -= 2*merge_w


matrix_length = np.array(paths_length_uncorr)
matrix_length[np.triu_indices(matrix_weight.shape[0],1)] -= 2
matrix_length[np.tril_indices(matrix_weight.shape[0],-1)] -= 2


matrix_prob = np.exp(-matrix_weight)
matrix_geom = np.exp(-matrix_weight / matrix_length)


np.save(mainpath+'graph_mat_roicom2roicom_w.npy', matrix_weight)
np.save(mainpath+'graph_mat_roicom2roicom_l.npy', matrix_length)
np.save(mainpath+'graph_mat_roicom2roicom_prob.npy', matrix_prob)
np.save(mainpath+'graph_mat_roicom2roicom_geom.npy', matrix_geom)




start_time = time()
fname_stl = mainpath + 'shortest_ROICOM2ROICOM.tck'
save_COM2COM_path_as_streamlines(paths_uncorr, 
                                 vertex2vox, 
                                 ref_img=mask_img, 
                                 fname=fname_stl,
                                 exclude_endpoints=True)
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



