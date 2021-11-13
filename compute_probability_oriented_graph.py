import argparse

import numpy as np
import nibabel as nib
from time import time

from dipy.data import get_sphere
from dipy.reconst.shm import real_sh_tournier
from dipy.reconst.shm import calculate_max_order

from utils import build_assign_mat_cone



DESCRIPTION = """
Compute and save the neighboor connection probability for oriented graph construction.
Needs a mask, an fODF field (in tournier07/Mrtrix3 format), a relative ODF threshold and a cone half-angle.
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

    p.add_argument('fodf', type=str, default=[],
                   help='Path of the fODF file.')
    p.add_argument('mask', type=str, default=[],
                   help='Path of the mask file.')
    p.add_argument('output', type=str, default=[],
                   help='Path of the output probability map.')
    p.add_argument('--th', type=float, default=0.2,
                   help='Relative threshold for fODF (i.e. 0.2*max(fODF)).')
    p.add_argument('--cone', type=float, default=60,
                   help='Half-angle of cone (degrees).')
    return p



def main():

    parser = buildArgsParser()
    args = parser.parse_args()


    mask_fname = args.mask
    fod_fname = args.fodf
    ODF_TH = args.th
    ang_th = args.cone
    out_fname = args.output

    sh_img = nib.load(fod_fname)
    sh = sh_img.get_fdata()
    affine = sh_img.affine

    mask = nib.load(mask_fname).get_fdata().astype(np.bool)
    print('Computing probability from fODF {:}'.format(fod_fname))
    print('inside mask {:}'.format(mask_fname))
    print('mask has {:} voxels'.format(mask.sum()))
    print('Using fODF threshold of {:.1f}% of max(fODF)'.format(100*ODF_TH))
    print('Using cone of half-angle {:.1f} degrees'.format(ang_th))

    # setup sh matrix
    lmax = calculate_max_order(sh.shape[3], False)
    print('fODF SH lmax detected = {:}'.format(lmax))
    sphere = get_sphere('repulsion724').subdivide(1)
    print('Using discrete sphere with {:} points'.format(sphere.vertices.shape[0]))
    B, m, n = real_sh_tournier(lmax, sphere.theta, sphere.phi)

    # get assignation matrix between sphere and neighboor
    assign_mat, vec = build_assign_mat_cone(sphere.vertices, ang_th, 3)


    print('Begin computing probability slice by slice')
    print('Computed map will be saved to {:}'.format(out_fname))
    prob_mat = np.zeros(sh.shape[:3]+(assign_mat.shape[1], assign_mat.shape[2]), dtype=np.float32)

    start_time = time()
    for Z in range(sh.shape[2]):
        print('Processing Z slice {:} / {:}    current-total {:.1f} s'.format(Z, sh.shape[2], time()-start_time))
        # skip slice if nothing in mask
        if np.any(mask[:,:,Z]):

            tmp = np.clip(sh[:,:,Z,:].dot(B.T), 0, np.inf)

            # thresholding at the source
            tmp_max = np.max(tmp, axis=2)
            tmp[(tmp < ODF_TH*tmp_max[:,:,None])] = 0

            for i_inc in range(assign_mat.shape[2]):
                prob_mat[:,:,Z,:, i_inc] = tmp.dot(assign_mat[:, :, i_inc])

    end_time = time()
    print('Elapsed time = {:.2f} s'.format(end_time - start_time))

    print('Cleanup')
    # memory gain before normalization
    del sh 
    del sh_img


    # normalization per neighboor for memory
    for i_inc in range(assign_mat.shape[2]):
        # make prob sum to 1
        prob_mat[..., i_inc] = prob_mat[..., i_inc] / prob_mat[..., i_inc].sum(axis=3)[:, :, :, None]

    # clean outside mask
    # TODO check is inf or Nan checking is necc.
    prob_mat[np.logical_not(mask)] = 0

    nib.Nifti1Image(prob_mat, affine).to_filename(out_fname)


if __name__ == "__main__":
    main()

