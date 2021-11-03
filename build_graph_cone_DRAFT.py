import numpy as np
import nibabel as nib 
import igraph as ig
from time import time
from dipy.data import get_sphere

from utils import build_assign_mat_cone, save_graph



ang_th = 60

# load data
mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'

mask_fname = mainpath + 'wmmask.nii'
prob_fname = mainpath + 'probability_cone_th_0p20_ang_60.nii.gz'

prob_img = nib.load(prob_fname)
prob = prob_img.get_fdata(dtype=np.float32)
affine = prob_img.affine

mask = nib.load(mask_fname).get_fdata().astype(np.bool)

# get assignation matrix between sphere and neighboor
sphere = get_sphere('repulsion724').subdivide(1)
# get assignation matrix between sphere and neighboor for each angle
_, vec = build_assign_mat_cone(sphere.vertices, ang_th, 3)






g = ig.Graph(directed=True)
# add a vertex for each voxel in mask
N = mask.sum()*vec.shape[0]
g.add_vertices(N)

# attribute a voxel to vertex
vox2vertex = {}
vertex2vox = {}
i = 0
for xyz in np.ndindex(prob.shape[:3]):
    if mask[xyz]:
        for i_inc in range(vec.shape[0]):
            vox2vertex[xyz+(i_inc,)] = i
            vertex2vox[i] = xyz+(i_inc,)
            g.vs[i]['name'] = xyz+(i_inc,)
            i += 1




# for 232021 voxel with odf rel th 20% and 60deg radius cone
# we have ~6M vertex and ~33.6M edges
# built in 408 seconds!!!

start_time = time()


edges_to_add_all = []
# new_prob_all = [] # never used it
neg_log_prob_all = []



for xyz in np.ndindex(prob.shape[:3]):
    if mask[xyz]:
        # print(xyz)
        for i_inc in range(vec.shape[0]):

            current_vertex = vox2vertex[xyz+(i_inc,)]
            # idx of neighboor
            neigh_idx = vec + xyz
            neigh_mask = np.logical_and(mask[(neigh_idx[:,0], neigh_idx[:,1], neigh_idx[:,2])], prob[xyz][:, i_inc] > 0)
            neigh_mask_nonzero = np.where(neigh_mask)[0]

            new_prob = prob[xyz][:, i_inc][neigh_mask] / prob[xyz][:, i_inc][neigh_mask].sum()
            # new_prob_all += new_prob.tolist() # never used it

            neg_log_prob = -np.log(new_prob)
            neg_log_prob_all += neg_log_prob.tolist()

            edges_to_add = [(current_vertex, vox2vertex[tuple(neigh_idx[neigh_mask][i_neigh])+(neigh_mask_nonzero[i_neigh],)]) for i_neigh in range(len(neigh_mask_nonzero))]
            edges_to_add_all += edges_to_add



# never used raw prob
# g.add_edges(edges_to_add_all, 
#             {'prob':new_prob_all, 'neg_log':neg_log_prob_all})

g.add_edges(edges_to_add_all, 
            {'neg_log':neg_log_prob_all})


end_time = time()
print('Elapsed time (edges) = {:.2f} s'.format(end_time - start_time))



# for memory we delete the prob
del prob


save_graph(g, mainpath + 'graph_cone_th_0p20_ang_60.pkl')




# xs, ys = zip(*[(left, count) for left, _, count in g.degree_distribution().bins()])

# import pylab as pl

# pl.figure()
# pl.bar(xs, ys)
# pl.show()



