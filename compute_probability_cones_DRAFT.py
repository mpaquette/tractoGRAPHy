import numpy as np
import nibabel as nib
from time import time

from dipy.data import get_sphere
from dipy.reconst.shm import real_sh_tournier
from dipy.reconst.shm import calculate_max_order

from utils import build_assign_mat_cone



# relative odf threshold
# points on odf smaller than ODF_TH*max(odf) are set to 0
ODF_TH = 0.2

# cone angle th
# th is from center, total "opening" is double
ang_th = 60



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

# get assignation matrix between sphere and neighboor for each angle
assign_mat, vec = build_assign_mat_cone(sphere.vertices, ang_th, 3)






# float32 for memory
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



# ODF_TH = 0.2
# ang_th = 60
out_fname = mainpath + 'probability_cone_th_0p20_ang_60.nii.gz'
nib.Nifti1Image(prob_mat, affine).to_filename(out_fname)





