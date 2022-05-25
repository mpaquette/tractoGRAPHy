import argparse

import numpy as np
import nibabel as nib 
from time import time
import igraph as ig

from utils import mask2vertex_cone, \
                  mask_COM, \
                  load_graph, \
                  compute_shortest_paths_COM2COM, \
                  save_COM2COM_path_as_streamlines


graph_fname = 'test_data/results/oriented/graph_0p2_60deg.pkl'
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









# For each voxel, we create an IN node to all 26 and an OUT node from all 26

mask_idx = [a for a in zip(*np.where(mask))]

name_IN = ['{:} IN'.format(idx) for idx in mask_idx]
g.add_vertices(name_IN)
name_OUT = ['{:} OUT'.format(idx) for idx in mask_idx]
g.add_vertices(name_OUT)


vertex_id_IN = [g.vs['name'].index(n) for n in name_IN]
vertex_id_OUT = [g.vs['name'].index(n) for n in name_OUT]

vertex2vox_IN = {}
vox2vertex_IN = {}
vertex2vox_OUT = {}
vox2vertex_OUT = {}

for i in range(len(mask_idx)):
	vertex2vox_IN[vertex_id_IN[i]] = mask_idx[i]
	vertex2vox_OUT[vertex_id_OUT[i]] = mask_idx[i]
	vox2vertex_IN[mask_idx[i]] = vertex_id_IN[i]
	vox2vertex_OUT[mask_idx[i]] = vertex_id_OUT[i]


# For each voxel, fid the 26 corresponding and connect
edges_to_add = []
for i in range(len(mask_idx)):
	# voxel in mask
	vox_idx = mask_idx[i]
	OUT_vert = vox2vertex_OUT[vox_idx]
	IN_vert = vox2vertex_IN[vox_idx]
	# 26 vertex in graph
	corresponding_vertex = [vox2vertex[vox_idx+(i_inc,)] for i_inc in range(26)]
	# add edge from 26 to OUT
	edges_to_add += [(i_vert, OUT_vert) for i_vert in corresponding_vertex]
	# add edge from IN to 26 
	edges_to_add += [(IN_vert, i_vert) for i_vert in corresponding_vertex]


g.add_edges(edges_to_add, 
            {'neg_log':[0]*len(edges_to_add)})



# list of all COM vertex from IN
COM_vertex = []
for i in range(1, source_label_map.max()+1):
    # get roi
    roi_mask = (source_label_map==i)
    # get COM vox
    COM = mask_COM(roi_mask)
    # get vertex id of all 26 node at that vox
    COM_vertex.append(vox2vertex_IN[mask_COM(roi_mask)])



# list of all voxel from OUT
target_vertex = list(vertex2vox_OUT.keys())







# this gives the paths, not the weight
path_vertex = []
for v in COM_vertex:
	tmp = g.get_all_shortest_paths(v, to=target_vertex, weights='neg_log', mode='out')
	path_vertex.append(tmp)

# vertex2vox_IN[path_vertex[i_COM][i_target][0]]
# vertex2vox_OUT[path_vertex[i_COM][i_target][-1]]


# numbers of COM node by number of mask voxel
path_length = np.inf*np.ones((len(COM_vertex), len(target_vertex)))
for i_COM in range(len(COM_vertex)):
	for path in path_vertex[i_COM]:
		# i_source = target_vertex.index(vox2vertex_OUT[vertex2vox_IN[path[0]]])
		i_target = target_vertex.index(path[-1])
		path_length[i_COM, i_target] = len(path)-3 # N nodes mean N-3 "jumps" (N-1-IN-OUT)



# source can be None for all
path_weight = g.shortest_paths(source=COM_vertex, target=target_vertex, weights='neg_log', mode='out')

# visitation map oriented
heatmap_log_COM = np.zeros(mask.shape + (len(COM_vertex), ), dtype=np.float32) 
heatmap_geom_COM = np.zeros(mask.shape + (len(COM_vertex), ), dtype=np.float32) 
for i_COM in range(len(COM_vertex)):
	for i_vertex in range(len(target_vertex)):
		vertex = target_vertex[i_vertex]
		prob = -path_weight[i_COM][i_vertex]
		heatmap_log_COM[vertex2vox_OUT[vertex]][i_COM] = prob
		l = path_length[i_COM, i_vertex]
		geom = np.exp(prob)**(1.0/l)
		if np.isinf(l):
			geom = 0.0
		heatmap_geom_COM[vertex2vox_OUT[vertex]][i_COM] = geom
	# remove self connection (prob 1 -> 0)
	heatmap_geom_COM[vertex2vox_IN[COM_vertex[i_COM]]][i_COM] = 0.0	

nib.Nifti1Image(heatmap_log_COM, affine).to_filename('test_data/results/oriented/visitation_map_log_COM.nii.gz')
nib.Nifti1Image(heatmap_geom_COM, affine).to_filename('test_data/results/oriented/visitation_map_geom_COM.nii.gz')





