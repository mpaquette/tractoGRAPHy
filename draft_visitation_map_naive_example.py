import argparse

import numpy as np
import nibabel as nib 
from time import time
import igraph as ig

from utils import mask2vertex, \
                  mask_COM, \
                  load_graph, \
                  compute_shortest_paths_COM2COM, \
                  save_COM2COM_path_as_streamlines


graph_fname = 'test_data/results/naive/graph_0p2.pkl'
source_roipath = 'test_data/label_map_connected.nii.gz'
target_roipath = 'test_data/label_map_connected.nii.gz'
mask_fname = 'test_data/wm_mask.nii.gz'

g, vertex2vox, vox2vertex = load_graph(graph_fname)
mask_img = nib.load(mask_fname)
affine = mask_img.affine
mask = mask_img.get_fdata().astype(np.bool)
source_label_map = nib.load(source_roipath).get_fdata().astype(np.int)
tmp_source_label_max = np.max(source_label_map)
source_label_map[np.logical_not(mask)] = 0
for i in range(1, tmp_source_label_max+1):
    if (source_label_map==i).sum() == 0:
        print('Source label map has no voxel inside mask for label = {:}'.format(i))

target_label_map = source_label_map.copy()









COM_vertex = []
for i in range(1, source_label_map.max()+1):
    # get roi
    roi_mask = (source_label_map==i)
    # get COM vox
    COM = mask_COM(roi_mask)
    # get vertex id of all 26 node at that vox
    COM_vertex.append(vox2vertex[mask_COM(roi_mask)])





# this gives the paths, not the weight
path_vertex = []
for v in COM_vertex:
	tmp = g.get_all_shortest_paths(v, to=None, weights='neg_log', mode='out')
	path_vertex.append(tmp)

# numbers of COM node by number of vertex (minus out-of-mask)
path_length = np.inf*np.ones((len(COM_vertex), g.vcount()-1))
for i_COM in range(len(COM_vertex)):
	for path in path_vertex[i_COM]:
		path_length[i_COM, path[-1]-1] = len(path)-1 # N nodes mean N-1 "jumps"



# source can be None for all
path_weight = g.shortest_paths(source=COM_vertex, target=None, weights='neg_log', mode='out')

# visitation map Naive
heatmap_log_COM = np.zeros(mask.shape + (len(COM_vertex), ), dtype=np.float32) 
heatmap_geom_COM = np.zeros(mask.shape + (len(COM_vertex), ), dtype=np.float32) 
for i_COM in range(len(COM_vertex)):
	for i_vertex in range(1, g.vcount()):
		# skip out-of-node vertex 0
		prob = -path_weight[i_COM][i_vertex]
		heatmap_log_COM[vertex2vox[i_vertex]][i_COM] = prob
		l = path_length[i_COM, i_vertex-1]
		geom = np.exp(prob)**(1.0/l)
		if np.isinf(l):
			geom = 0.0
		heatmap_geom_COM[vertex2vox[i_vertex]][i_COM] = geom
	# remove self connection (prob 1 -> 0)
	heatmap_geom_COM[vertex2vox[COM_vertex[i_COM]]][i_COM] = 0.0	

nib.Nifti1Image(heatmap_log_COM, affine).to_filename('test_data/results/naive/visitation_map_log_COM.nii.gz')
nib.Nifti1Image(heatmap_geom_COM, affine).to_filename('test_data/results/naive/visitation_map_geom_COM.nii.gz')





