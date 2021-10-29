import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import center_of_mass
import igraph as ig


def build_assign_mat(vertices, cube_size=3):
	# Build a matrix assigning each vertices
	# to a neighboor inside a cube of side cube_size

	if vertices.shape[1] != 3:
		vertices = vertices.T 
	dirs = vertices / np.linalg.norm(vertices, axis=1)[:, None]

	# get coordinate of neighboor in cube
	coords_1D = np.arange(-(cube_size//2), (cube_size//2) + 1)
	X,Y,Z = np.meshgrid(coords_1D, coords_1D, coords_1D, indexing='ij')
	vec = np.concatenate((X.ravel()[:,None], Y.ravel()[:,None], Z.ravel()[:,None]), axis=1)
	vecmask = np.logical_not(np.all(vec==0, axis=1))
	vec = vec[vecmask]
	vec_norm = vec / np.linalg.norm(vec, axis=1)[:, None]

	# since all normalised, min(euclidean) == min(angle)
	dist_mat = cdist(dirs, vec_norm, metric='euclidean')
	closest = np.argmin(dist_mat, axis=1)

	assign_mat = np.zeros((dirs.shape[0], vec.shape[0]), dtype=np.int)
	assign_mat[(range(dirs.shape[0]), closest)] = 1

	return assign_mat, vec


# vertices = np.random.randn(1000000, 3)
# assign_mat, vec = build_assign_mat(vertices, cube_size=3)

# # for i in range(vec.shape[0]):
# # 	print(vec[i], '{:.2f} % of points'.format(100*assign_mat[:,i].sum()/assign_mat.shape[0]))

# dist = np.unique(np.abs(vec).sum(axis=1))
# prop = []
# for d in dist:
# 	prop.append(assign_mat[:, np.abs(vec).sum(axis=1)==d].sum() / (np.abs(vec).sum(axis=1)==d).sum() / assign_mat.shape[0])
# 	print(d, '{:.2f} %'.format(100*prop[-1]))


def load_graph(fname_pickle):
	# load a pickle igraph
	# reconstruct vox2vertex and vertex2vox dictionaries
	g = ig.Graph.Read_Pickle(fname_pickle)

	vertex2vox = {}
	vox2vertex = {}
	for i, vertex in enumerate(g.vs):
		vox = vertex['name']
		vertex2vox[i] = vox
		vox2vertex[vox] = i

	return g, vertex2vox, vox2vertex


def save_graph(g, fname_pickle):
	# save graph as pickle
	# this function does nothing except normalizing file extension
	if fname_pickle[-4:] != '.pkl':
		fname_pickle += '.pkl'
	print('Saving graph to {}'.format(fname_pickle))
	g.write_pickle(fname=fname_pickle)


def mask2vertex(mask, vox2vertex):
	# return the graph vertex of all voxels in a mask
	_vox = np.array(np.where(mask))
	_xyz = [tuple(_vox[:,i]) for i in range(_vox.shape[1])]
	_vertex = [vox2vertex[xyz] for xyz in _xyz]
	return _vertex


def mask_COM(mask):
	# compute Center-of-Mass of mask
	# return voxel in mask closest (euclidean) to COM
	com = np.array(center_of_mass(mask))[None, :]
	vox = np.array(np.where(mask)).T
	closest_idx = np.argmin(cdist(com, vox)[0])
	return tuple(vox[closest_idx])












