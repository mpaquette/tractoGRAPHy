import argparse

import numpy as np
import nibabel as nib 
import igraph as ig
from time import time
from dipy.data import get_sphere

from utils import build_assign_mat, save_graph


DESCRIPTION = """
Compute and save naive graph construction.
Needs a mask and a probability map (from compute_probability_naive_graph.py).
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

    p.add_argument('prob', type=str, default=[],
                   help='Path of the probability file.')
    p.add_argument('mask', type=str, default=[],
                   help='Path of the mask file.')
    p.add_argument('output', type=str, default=[],
                   help='Path of the output graph.')
    return p




def main():

    parser = buildArgsParser()
    args = parser.parse_args()


    mask_fname = args.mask
    prob_fname = args.prob
    out_fname = args.output




    # load data
    prob_img = nib.load(prob_fname)
    prob = prob_img.get_fdata()
    affine = prob_img.affine

    mask = nib.load(mask_fname).get_fdata().astype(np.bool)

    print('Computing graph with probabilities from {:}'.format(prob_fname))
    print('inside mask {:}'.format(mask_fname))
    print('mask has {:} voxels'.format(mask.sum()))


    # TODO: save and load this from the probability creation
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


    print('Begin building naive graph.')
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
    print('Elapsed time = {:.2f} s'.format(end_time - start_time))


    print('Graph has {:} nodes'.format(len(g.vs)))
    print('Graph has {:} edges'.format(len(g.es)))


    save_graph(g, out_fname)


if __name__ == "__main__":
    main()

