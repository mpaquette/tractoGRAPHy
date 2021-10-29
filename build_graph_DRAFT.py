import numpy as np
import nibabel as nib 
import igraph as ig
from time import time
from dipy.data import get_sphere

from utils import build_assign_mat, save_graph


# load data
mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'

mask_fname = mainpath + 'wmmask.nii'
prob_fname = mainpath + 'probability_th_0p20.nii.gz'

prob_img = nib.load(prob_fname)
prob = prob_img.get_fdata()
affine = prob_img.affine

mask = nib.load(mask_fname).get_fdata().astype(np.bool)


# get assignation matrix between sphere and neighboor
sphere = get_sphere('repulsion724').subdivide(1)
assign_mat, vec = build_assign_mat(sphere.vertices, 3)




g = ig.Graph(directed=True)
# add a vertex for each voxel in mask
N = mask.sum()
g.add_vertices(N)

# attribute a voxel to vertex
vox2vertex = {}
vertex2vox = {}
i = 0
for xyz in np.ndindex(prob.shape[:3]):
    if mask[xyz]:
        vox2vertex[xyz] = i
        vertex2vox[i] = xyz
        g.vs[i]['name'] = xyz
        i += 1



start_time = time()


edges_to_add_all = []
new_prob_all = []
neg_log_prob_all = []



for xyz in np.ndindex(prob.shape[:3]):
    if mask[xyz]:
        # print(xyz)
        current_vertex = vox2vertex[xyz]
        # idx of neighboor
        neigh_idx = vec + xyz
        neigh_mask = np.logical_and(mask[(neigh_idx[:,0], neigh_idx[:,1], neigh_idx[:,2])], prob[xyz] > 0)
        
        new_prob = prob[xyz][neigh_mask] / prob[xyz][neigh_mask].sum()
        new_prob_all += new_prob.tolist()

        neg_log_prob = -np.log(new_prob)
        neg_log_prob_all += neg_log_prob.tolist()

        edges_to_add = [(current_vertex, vox2vertex[tuple(n_idx)]) for n_idx in neigh_idx[neigh_mask]]
        edges_to_add_all += edges_to_add




g.add_edges(edges_to_add_all, 
            {'prob':new_prob_all, 'neg_log':neg_log_prob_all})


end_time = time()
print('Elapsed time (edges) = {:.2f} s'.format(end_time - start_time))


save_graph(g, mainpath + 'graph_th_0p20.pkl')


# xs, ys = zip(*[(left, count) for left, _, count in g.degree_distribution().bins()])

# import pylab as pl

# pl.figure()
# pl.bar(xs, ys)
# pl.show()



