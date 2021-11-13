import argparse

import numpy as np
import nibabel as nib 
import igraph as ig
from time import time
from dipy.data import get_sphere

from utils import build_assign_mat_cone, save_graph


DESCRIPTION = """
Compute and save oriented graph construction.
Needs a mask, a probability map (from compute_probability_oriented_graph.py) and the cone half-angle that was used.
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
    p.add_argument('cone', type=float, default=[],
                   help='Half-angle of cone (degrees).')
    p.add_argument('output', type=str, default=[],
                   help='Path of the output graph.')
    return p



def main():

    parser = buildArgsParser()
    args = parser.parse_args()


    mask_fname = args.mask
    prob_fname = args.prob
    ang_th = args.cone
    out_fname = args.output



    # load data
    prob_img = nib.load(prob_fname)
    prob = prob_img.get_fdata(dtype=np.float32)
    affine = prob_img.affine

    mask = nib.load(mask_fname).get_fdata().astype(np.bool)
    print('Computing graph with probabilities from {:}'.format(prob_fname))
    print('inside mask {:}'.format(fmask_fname))
    print('mask has {:} voxels'.format(mask.sum()))
    print('Using cone of half-angle {:.1f} degrees'.format(ang_th))


    # TODO: save and load this from the probability creation
    # get assignation matrix between sphere and neighboor
    sphere = get_sphere('repulsion724').subdivide(1)
    # get assignation matrix between sphere and neighboor for each angle
    _, vec = build_assign_mat_cone(sphere.vertices, ang_th, 3)





    print('Begin building naive graph.')
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



    g.add_edges(edges_to_add_all, 
                {'neg_log':neg_log_prob_all})


    end_time = time()
    print('Elapsed time = {:.2f} s'.format(end_time - start_time))


    # for memory we delete the prob
    del prob


    save_graph(g, out_fname)


if __name__ == "__main__":
    main()
