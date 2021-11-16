import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import center_of_mass
import igraph as ig
from dipy.io.streamline import save_tck
from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram


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


def build_assign_mat_cone(vertices, angle_thr, cube_size=3):
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

    # build assign mat with cone
    assign_mat_nocone = np.zeros((dirs.shape[0], vec.shape[0]), dtype=np.int)
    assign_mat_nocone[(range(dirs.shape[0]), closest)] = 1

    assign_mat = np.zeros((dirs.shape[0], vec.shape[0], vec.shape[0]), dtype=np.int)

    # with each neighboor vector as incoming direction, 
    # clamp to 0 (unassign) all points outside of cone 
    eucl_dist_cone = np.sqrt(2-2*np.cos((np.pi/180)*angle_thr))
    # build cones
    for i_inc in range(vec.shape[0]):
        cone_center = vec_norm[i_inc] # oups
        dist_mat_inc = cdist(dirs, cone_center[None,:], metric='euclidean')
        conemask = dist_mat_inc[:,0] <= eucl_dist_cone
        # ((4*np.pi*np.sin(((np.pi/180)*angle_thr)/2)**2) / (4*np.pi)) # expected volume chunk
        # np.sum(dist_mat_inc <= eucl_dist_cone) / vertices.shape[0] # point approx
        assign_mat[conemask, :, i_inc] = assign_mat_nocone[conemask]

    return assign_mat, vec


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


def path_stat(g, vertices, w='neg_log', func=np.sum):
    # return func of edge_weight w along path defined by vertices
    edge_weights = np.array([g.es(g.get_eid(vertices[i], vertices[i+1]))[w] for i in range(len(vertices)-1)])[:,0]
    return func(edge_weights)


def mask2vertex(mask, vox2vertex):
    # return the graph vertex of all voxels in a mask
    _vox = np.array(np.where(mask))
    _xyz = [tuple(_vox[:,i]) for i in range(_vox.shape[1])]
    _vertex = [vox2vertex[xyz] for xyz in _xyz]
    return _vertex


def mask2vertex_cone(mask, vox2vertex):
    # return the graph vertex of all voxels in a mask
    _vox = np.array(np.where(mask))
    _xyz = [tuple(_vox[:,i]) for i in range(_vox.shape[1])]
    _vertex = [[vox2vertex[xyz+(i_inc,)] for i_inc in range(26)] for xyz in _xyz]
    return _vertex


def mask_COM(mask):
    # compute Center-of-Mass of mask
    # return voxel in mask closest (euclidean) to COM
    com = np.array(center_of_mass(mask))[None, :]
    vox = np.array(np.where(mask)).T
    closest_idx = np.argmin(cdist(com, vox)[0])
    return tuple(vox[closest_idx])



def compute_shortest_paths_COM2COM(g, source_COMs, target_COMs, w='neg_log'):
    # compute shortest paths between all pairs of center-of-mass voxel

    # compute graph weight of shortest path
    path_weights = g.shortest_paths(source=source_COMs, 
                                    target=target_COMs, 
                                    weights=w, 
                                    mode='out')

    # compute the graph shortest path
    paths = [] # paths[i_source][i_dest]
    for i_source in range(len(source_COMs)):
        source = source_COMs[i_source]

        path = g.get_shortest_paths(source, 
                                    to=target_COMs, 
                                    weights=w, 
                                    mode='out', 
                                    output='vpath')
        paths.append(path)

    # compute the lenght in term of vertex of the shortest path
    paths_length = []
    for i_source in range(len(source_COMs)):
        path = paths[i_source]
        paths_length.append([len(s) for s in path])

    return paths, paths_length, path_weights


def save_COM2COM_path_as_streamlines(paths, vertex2vox, ref_img, fname, exclude_endpoints=False):

    # loop into paths of vertex to create list of array of voxel
    streamlines = []
    for i_source in range(len(paths)):
        for i_dest in range(len(paths[i_source])):
            p = paths[i_source][i_dest]
            if len(p) > 1:
                # ignore endpoints COM_i nodes
                # ignore i_inc index for strealine
                if exclude_endpoints:
                    p = p[1:-1]
                streamlines.append(np.array([vertex2vox[v] for v in p])[:,:3])

    tgm = StatefulTractogram(
                        streamlines=streamlines,
                        reference=ref_img,
                        space=Space.VOX,
                        origin=Origin.NIFTI)

    save_tck(tgm, fname, bbox_valid_check=False)





def compute_shortest_paths_COM2ALL_w(g, COMs, rois_vertex, w='neg_log'):

    all_dest = [vert for roi_vertex in rois_vertex for vert in roi_vertex]
    all_count = [len(roi_vertex) for roi_vertex in rois_vertex]

    # compute graph weight of shortest path
    path_weights_grouped = g.shortest_paths(source=COMs, 
                                            target=all_dest, 
                                            weights=w, 
                                            mode='out')

    # degroup
    path_weights = []
    for i_source in range(len(path_weights_grouped)):
        tmp_source = []
        for i_dest in range(len(all_count)):
            a = int(np.sum(all_count[:i_dest]))
            b = int(np.sum(all_count[:i_dest])+all_count[i_dest])
            tmp_source.append(path_weights_grouped[i_source][a:b])
        path_weights.append(tmp_source)

    return path_weights


# works but its a memory buster, need to a function saving for each source
def compute_shortest_paths_1COM2ALL(g, COM, rois_vertex, w='neg_log'):

    all_dest = [vert for roi_vertex in rois_vertex for vert in roi_vertex]
    all_count = [len(roi_vertex) for roi_vertex in rois_vertex]


    # compute the graph shortest path
    paths_grouped = g.get_shortest_paths(COM, 
                                to=all_dest, 
                                weights=w, 
                                mode='out', 
                                output='vpath')

    # degroup
    paths = []
    for i_dest in range(len(all_count)):
        a = int(np.sum(all_count[:i_dest]))
        b = int(np.sum(all_count[:i_dest])+all_count[i_dest])
        paths.append(paths_grouped[a:b])


    # compute the lenght in term of vertex of the shortest path
    paths_length = []
    for i_dest in range(len(all_count)):
        path = paths[i_dest]
        paths_length.append([len(s) for s in path])

    return paths, paths_length

