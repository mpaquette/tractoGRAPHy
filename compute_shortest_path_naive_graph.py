#!/usr/bin/env python3

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


DESCRIPTION = """
Compute shortest paths and connectivity matrices for naive graph.
Needs a naive graph, a mask, a label map and a target_type.

The label map is expected to have integer value above 0 for each source/target.
We intersect the label map with the mask and discard what's outside.

For target_type 'COM', we compute the center-of-mass of each label (what remains after the intersection) and we find the closest voxel inside the mask to that center-of-mass.
The shortest paths are computed for all pairs of COM voxels only.

For target_type 'ROI', we create new nodes for each label and attach them to every voxel of that label.
The shortest paths are computed between these ROI-nodes. 
This is equivalent to computing the shortest path between every pair of voxels of 2 ROIs and picking the shortest of the shortest.

Using the provided basename, we save 4 matrix: 
    - "w" with the graph weights (-ln(probability)).
    - "l" with the path lengths in number of nodes.
    - "prob" with the path probability.
    - "geom" with the geometric mean of the steps probabilities along the paths

If the savepath flag is enabled, we save all the shortest paths as streamlines defined by the voel coordinates.
"""

EPILOG = """
Michael Paquette, MPI CBS, 2021.
"""

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass

def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION,
                                epilog=EPILOG,
                                formatter_class=CustomFormatter)

    p.add_argument('graph', type=str, default=[],
                   help='Path of the naive graph (pickled).')
    p.add_argument('label', type=str, default=[],
                   help='Path of the label map.')
    p.add_argument('mask', type=str, default=[],
                   help='Path of the mask file.')
    p.add_argument('target', choices=('COM', 'ROI'))
    p.add_argument('output', type=str, default=[],
                   help='Base path of the output matrix.')
    p.add_argument('--savepath', type=str, default=None,
                   help='Output the paths as tck file if file name is given.')
    return p




def main():

    parser = buildArgsParser()
    args = parser.parse_args()

    graph_fname = args.graph
    roipath = args.label
    mask_fname = args.mask
    out_basefname = args.output


    g, vertex2vox, vox2vertex = load_graph(graph_fname)
    print('Loaded graph with {:} nodes and {:} edges'.format(len(g.vs), len(g.es)))

    mask_img = nib.load(mask_fname)
    affine = mask_img.affine
    mask = mask_img.get_fdata().astype(np.bool)

    label_map = nib.load(roipath).get_fdata().astype(np.int)
    print('Label map has {:} greater than zero voxel'.format((label_map>0).sum()))
    tmp_label_max = np.max(label_map)

    label_map[np.logical_not(mask)] = 0
    print('Label map has {:} greater than zero voxel inside mask'.format((label_map>0).sum()))

    for i in range(1, tmp_label_max+1):
        if (label_map==i).sum() == 0:
            print('Label map has no voxel inside mask for label = {:}'.format(i))


    if args.target == 'COM':
        print('Using Center-of-Mass as sources/targets')

        # compute center-of-mass -ish voxel for each roi
        target_vertex = []
        for i in range(1, label_map.max()+1):
            roi_mask = (label_map==i)
            target_vertex.append(vox2vertex[mask_COM(roi_mask)])


    elif args.target == 'ROI':
        print('Using ROI nodes as sources/targets')

        rois_vertex = [mask2vertex(label_map==i, vox2vertex) for i in range(1, label_map.max()+1)]


        start_time = time()
        # compute center-of-mass -ish voxel for each roi
        # add nodes
        g.add_vertices(['ROI_{}'.format(i) for i in range(1, label_map.max()+1)])

        target_vertex = []
        edges_to_add = []
        for i in range(1, label_map.max()+1):
            ROI_vertex = rois_vertex[i-1]
            # add new vertex to converge them all there
            new_vert_id = g.vs['name'].index('ROI_{}'.format(i))
            target_vertex.append(new_vert_id)
            # create IN and OUT edge for all node at COM
            edges_to_add += [(new_vert_id, i_vert) for i_vert in ROI_vertex]
            edges_to_add += [(i_vert, new_vert_id) for i_vert in ROI_vertex]

        # TODO replace the big weight hack with 2 ROI nodes, a source and a target with unidirectional free edges
        merge_w = 10000 # remove twice from computed path weight

        # edge of zero could give loops
        # instead we put very very expensive nodes, and we can remove it when counting
        g.add_edges(edges_to_add, 
                    {'neg_log':[merge_w]*len(edges_to_add)})
        end_time = time()

        print('Making ROI-nodes = {:.2f} s'.format(end_time - start_time))
        print('Temporarily added {:} nodes to graph'.format(len(target_vertex)))
        print('Temporarily added {:} edges to graph'.format(len(edges_to_add)))



    print('Computing shortest paths')
    start_time = time()
    paths, paths_length, weights = compute_shortest_paths_COM2COM(g,
                                                                  target_vertex,
                                                                  w='neg_log')

    end_time = time()
    print('Elapsed time = {:.2f} s'.format(end_time - start_time))


    if args.target == 'COM':
        # build matrices from shortest path
        matrix_weight = np.array(weights)
        matrix_length = np.array(paths_length)
        matrix_prob = np.exp(-matrix_weight)
        matrix_geom = np.exp(-matrix_weight / matrix_length)

    elif args.target == 'ROI':
        ## correct values to account for big ROI edges
        matrix_weight = np.array(weights)
        matrix_weight[np.triu_indices(matrix_weight.shape[0],1)] -= 2*merge_w
        matrix_weight[np.tril_indices(matrix_weight.shape[0],-1)] -= 2*merge_w

        matrix_length = np.array(paths_length)
        matrix_length[np.triu_indices(matrix_weight.shape[0],1)] -= 2
        matrix_length[np.tril_indices(matrix_weight.shape[0],-1)] -= 2

        matrix_prob = np.exp(-matrix_weight)
        matrix_geom = np.exp(-matrix_weight / matrix_length)


    # save matrice
    print('Saving matrices as {:}_(w,l,prob,geom).npy'.format(out_basefname))
    np.save(out_basefname + '_w.npy', matrix_weight)
    np.save(out_basefname + '_l.npy', matrix_length)
    np.save(out_basefname + '_prob.npy', matrix_prob)
    np.save(out_basefname + '_geom.npy', matrix_geom)



    if args.savepath is not None:
        print('Saving paths as tck')
        start_time = time()
        if args.target == 'COM':
            tmp_exclude_endpoints = False
        elif args.target == 'ROI':
            tmp_exclude_endpoints = True

        save_COM2COM_path_as_streamlines(paths, 
                                         vertex2vox, 
                                         ref_img=mask_img, 
                                         fname=args.savepath,
                                         exclude_endpoints=tmp_exclude_endpoints)


        end_time = time()
        print('Elapsed time = {:.2f} s'.format(end_time - start_time))


if __name__ == "__main__":
    main()

