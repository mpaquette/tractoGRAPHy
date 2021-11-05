# numpy, nibabel, utils

# assign label to each maskroi, match collision to nearest COM

import numpy as np 
import nibabel as nib
from utils import mask2vertex


# load roi mask and clip to WM mask
roipath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/rois/'
from os import listdir
from os.path import isfile, join
roifiles = [f for f in listdir(roipath) if isfile(join(roipath, f))]
rois_fname = [roipath + roifiles[i] for i in range(len(roifiles))]


mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'
mask_fname = mainpath + 'wmmask.nii'
mask_img = nib.load(mask_fname)
affine = mask_img.affine
mask = mask_img.get_fdata().astype(np.bool)


rois_mask_tmp = [nib.load(fname).get_fdata().astype(np.bool) for fname in rois_fname]
rois_mask = [np.logical_and(roi_mask, mask) for roi_mask in rois_mask_tmp]







names = [rois_fname[i_source].split('/')[-1][:-7][3:-13] for i_source in range(len(rois_fname))]
order = np.argsort(names)


label_fname = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'

with open(label_fname + 'labels.txt', 'w') as f:
	for idx in order:
		f.write(names[idx]+'\n')




data_label_init = np.empty(mask.shape, dtype=tuple)
for xyz in np.ndindex(data_label_init.shape):
	data_label_init[xyz] = ()


for ilabel, idx in enumerate(order):
	X,Y,Z = np.where(rois_mask[idx])
	for i in range(len(X)):
		# data_label_init[X[i], Y[i], Z[i]] += (idx+1,)
		data_label_init[X[i], Y[i], Z[i]] += (ilabel+1,)



data_label = np.zeros(mask.shape, dtype=np.int)
data_label_contested = np.zeros(mask.shape, dtype=np.bool)
for xyz in np.ndindex(data_label_init.shape):
	s = len(data_label_init[xyz])
	if s == 0:
		data_label[xyz] = 0
	elif s == 1:
		data_label[xyz] = data_label_init[xyz][0]
	else:
		data_label_contested[xyz] = True






# from scipy.ndimage.measurements import label
# from scipy.ndimage import generate_binary_structure

# # structure = np.zeros((3,3,3))
# # structure[0] = np.array([[0, 0, 0],
# # 						 [0, 1, 0],
# # 						 [0, 0, 0]])

# # structure[1] = np.array([[0, 1, 0],
# # 						 [1, 1, 1],
# # 						 [0, 1, 0]])

# # structure[2] = np.array([[0, 0, 0],
# # 						 [0, 1, 0],
# # 						 [0, 0, 0]])
# # structure = generate_binary_structure(3, 1)
# structure = generate_binary_structure(3, 3)

# tmp_label, n_label = label(rois_mask[90], structure)


from scipy.ndimage import center_of_mass

rois_center_voxel = []
for idx in order:
	roi_mask = rois_mask[idx]
	com = np.array(center_of_mass(roi_mask))
	rois_center_voxel.append(com)

rois_center_voxel = np.array(rois_center_voxel)



for xyz in np.ndindex(data_label_init.shape):
	if data_label_contested[xyz]:
		data_label[xyz] = np.argmin(cdist(np.array(xyz)[None, :], rois_center_voxel)[0]) + 1



mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'

out_fname = mainpath + 'all_label.nii.gz'
nib.Nifti1Image(data_label, affine).to_filename(out_fname)

out_fname = mainpath + 'all_label_bin.nii.gz'
nib.Nifti1Image((data_label>0).astype(np.int), affine).to_filename(out_fname)

