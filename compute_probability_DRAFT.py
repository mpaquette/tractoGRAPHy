import numpy as np
import nibabel as nib
from time import time

from dipy.data import get_sphere
from dipy.reconst.shm import real_sh_tournier
from dipy.reconst.shm import calculate_max_order

from utils import build_assign_mat



# relative odf threshold
# points on odf smaller than ODF_TH*max(odf) are set to 0
ODF_TH = 0.2


# load data
mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'

mask_fname = mainpath + 'wmmask.nii'
fod_fname = mainpath + 'fod.nii'

sh_img = nib.load(fod_fname)
sh = sh_img.get_fdata()
affine = sh_img.affine

mask = nib.load(mask_fname).get_fdata().astype(np.bool)


# setup sh matrix
lmax = calculate_max_order(sh.shape[3], False)
# sphere = get_sphere('repulsion100').subdivide(0)
sphere = get_sphere('repulsion724').subdivide(1)
B, m, n = real_sh_tournier(lmax, sphere.theta, sphere.phi)

# get assignation matrix between sphere and neighboor
assign_mat, vec = build_assign_mat(sphere.vertices, 3)



prob_mat = np.zeros(sh.shape[:3]+(vec.shape[0],))

start_time = time()
for Z in range(sh.shape[2]):
	print('Processing Z slice {:} / {:}'.format(Z, sh.shape[2]))
	# skip slice if nothing in mask
	if np.any(mask[:,:,Z]):

		tmp = np.clip(sh[:,:,Z,:].dot(B.T), 0, np.inf)

		# thresholding at the source
		tmp_max = np.max(tmp, axis=2)
		tmp[(tmp < ODF_TH*tmp_max[:,:,None])] = 0

		prob_mat[:,:,Z,:] = tmp.dot(assign_mat)

end_time = time()
print('Elapsed time = {:.2f} s'.format(end_time - start_time))




# make prob sum to 1
prob_mat = prob_mat / prob_mat.sum(axis=3)[..., None]
# clean outside mask
# TODO check is inf or Nan checking is necc.
prob_mat[np.logical_not(mask)] = 0


out_fname = mainpath + 'probability_th_0p20.nii.gz'
nib.Nifti1Image(prob_mat, affine).to_filename(out_fname)





