import numpy as np

import nibabel as nib

import sklearn.cluster as clu


n_parcel_each = 50


mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'


# load label mask (already intersected with wm mask)
roipath = mainpath + 'all_label.nii.gz'
roiimg = nib.load(roipath)
affine = roiimg.affine
label_map = roiimg.get_fdata().astype(np.int)

outpath = mainpath + 'random_parc_hemi_{}.nii.gz'.format(n_parcel_each)


offset = 180
hemi_1_mask = np.logical_and(label_map>0, label_map<=offset)
hemi_2_mask = np.logical_and(label_map>offset, label_map<=2*offset)


print('nb label',(label_map>0).sum())
print('nb label 1',hemi_1_mask.sum())
print('nb label 2',hemi_2_mask.sum())
print('nb label 1+2',np.logical_or(hemi_1_mask, hemi_2_mask).sum())
print('np label 1-2',np.logical_and(hemi_1_mask, hemi_2_mask).sum())


pos_1 = np.where(hemi_1_mask)
pts_1 = np.zeros((np.array(pos_1[0]).shape[0], 3))
pts_1[:,0] = np.array(pos_1[0])
pts_1[:,1] = np.array(pos_1[1])
pts_1[:,2] = np.array(pos_1[2])


pos_2 = np.where(hemi_2_mask)
pts_2 = np.zeros((np.array(pos_2[0]).shape[0], 3))
pts_2[:,0] = np.array(pos_2[0])
pts_2[:,1] = np.array(pos_2[1])
pts_2[:,2] = np.array(pos_2[2])


n_init = 3
clusterer1 = clu.MiniBatchKMeans(n_clusters=n_parcel_each, n_init=n_init, init="k-means++", compute_labels=True)
fitted_cluster_1 = clusterer1.fit(pts_1)


clusterer2 = clu.MiniBatchKMeans(n_clusters=n_parcel_each, n_init=n_init, init="k-means++", compute_labels=True)
fitted_cluster_2 = clusterer2.fit(pts_2)


parcelation = np.zeros_like(label_map).astype(np.float32)
for ii in range(pts_1.shape[0]):
    parcelation[int(pts_1[ii,0]), int(pts_1[ii,1]), int(pts_1[ii,2])] = fitted_cluster_1.labels_[ii] + 1
for ii in range(pts_2.shape[0]):
    parcelation[int(pts_2[ii,0]), int(pts_2[ii,1]), int(pts_2[ii,2])] = fitted_cluster_2.labels_[ii] + n_parcel_each + 1


output = nib.Nifti1Image(parcelation.astype(np.int), affine)
nib.save(output, outpath)



