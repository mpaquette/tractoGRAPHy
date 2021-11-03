import numpy as np
import nibabel as nib 

from time import time

import igraph as ig

from utils import mask2vertex_cone, \
                  mask_COM, \
                  load_graph, \
                  compute_shortest_paths_COM2COM, \
                  save_COM2COM_path_as_streamlines




mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'
g, vertex2vox, vox2vertex = load_graph(mainpath + 'graph_cone_th_0p20_ang_60.pkl')


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
rois_vertex_cone = [mask2vertex_cone(label_map==i, vox2vertex) for i in range(1, label_map.max()+1)]



for i in range(1, label_map.max()+1):
	new_vert_id = g.vs['name'].index('COM_{}'.format(i))
	g.vs[new_vert_id]['name'] = 'ROI_{}'.format(i)


start_time = time()
# compute center-of-mass -ish voxel for each roi
# add nodes
g.add_vertices(['ROI_{}'.format(i) for i in range(1, label_map.max()+1)])

rois_center_vertex_cone = []
edges_to_add = []
for i in range(1, label_map.max()+1):
    ROI_vertex_cone = [vert for voxvert in rois_vertex_cone[i-1] for vert in voxvert]
    # add new vertex to converge them all there
    new_vert_id = g.vs['name'].index('ROI_{}'.format(i))
    rois_center_vertex_cone.append(new_vert_id)
    # create IN and OUT edge for all node at COM
    edges_to_add += [(new_vert_id, i_vert) for i_vert in ROI_vertex_cone]
    edges_to_add += [(i_vert, new_vert_id) for i_vert in ROI_vertex_cone]


merge_w = 10000 # remove twice from computed path weight

# edge of zero could give loops
# instead we put very very expensive nodes, and we can remove it when counting
g.add_edges(edges_to_add, 
            {'neg_log':[merge_w]*len(edges_to_add)})
end_time = time()
print('Make super-ROI = {:.2f} s'.format(end_time - start_time))




start_time = time()
paths_uncorr, paths_length_uncorr, path_weights_uncorr = compute_shortest_paths_COM2COM(g, rois_center_vertex_cone, w='neg_log')
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


np.save(mainpath+'graph_mat_cone60_comroi2comroi_w.npy', matrix_weight)
np.save(mainpath+'graph_mat_cone60_comroi2comroi_l.npy', matrix_length)
np.save(mainpath+'graph_mat_cone60_comroi2comroi_prob.npy', matrix_prob)
np.save(mainpath+'graph_mat_cone60_comroi2comroi_geom.npy', matrix_geom)




start_time = time()
fname_stl = mainpath + 'shortest_cone60_COMROI2COMROI.tck'
save_COM2COM_path_as_streamlines(paths_uncorr, 
                                 vertex2vox, 
                                 ref_img=mask_img, 
                                 fname=fname_stl,
                                 exclude_endpoints=True)
end_time = time()
print('Elapsed time (save path as streamlines) = {:.2f} s'.format(end_time - start_time))



