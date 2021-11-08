import numpy as np
import nibabel as nib 

from time import time

import igraph as ig

from utils import mask2vertex, \
                  mask_COM, \
                  load_graph, \
                  path_stat

from dipy.io.streamline import load_tck
from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram





mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'
g, vertex2vox, vox2vertex = load_graph(mainpath + 'graph_th_0p20.pkl')


mask_fname = mainpath + 'wmmask.nii'
mask_img = nib.load(mask_fname)
affine = mask_img.affine
mask = mask_img.get_fdata().astype(np.bool)




# load label mask (already intersected with wm mask)
roipath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/random_parc_hemi_50.nii.gz'
label_map = nib.load(roipath).get_fdata().astype(np.int)



# this is for list of mask
# rois_vertex = [mask2vertex(roi_mask, vox2vertex) for roi_mask in rois_mask]
# this is for label map
rois_vertex = [mask2vertex(label_map==i, vox2vertex) for i in range(1, label_map.max()+1)]



# compute center-of-mass -ish voxel for each roi
rois_center_vertex = []
# for roi_mask in rois_mask:
#   rois_center_vertex.append(vox2vertex[mask_COM(roi_mask)])
for i in range(1, label_map.max()+1):
    roi_mask = (label_map==i)
    rois_center_vertex.append(vox2vertex[mask_COM(roi_mask)])




# # import required module
# import os
# # assign directory
# directory = 'files'
 
# # iterate over files in
# # that directory
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         print(f)



bundle_fname = mainpath + 'scil_decomp/' + 'step_0p5_theta_90_th_0p20_npv_50_seed_int_randparc50/' + 'centroids/' + '3_13.tck'
tractogram = load_tck(bundle_fname, 
                      reference=mask_img,
                      to_space=Space.VOX,
                      to_origin=Origin.NIFTI,
                      bbox_valid_check=False)


inv_affine = np.linalg.inv(affine)
stl = tractogram.streamlines[0]
stl_vox = inv_affine.dot(np.concatenate((stl.T, np.ones((1,stl.shape[0]))), axis=0))[:3].T





def broken(path_vox):
  last = path_vox[0]
  for i in range(1, len(path_vox)):
    new = path_vox[i]
    if (abs(last[0]-new[0]) > 1) or (abs(last[1]-new[1]) > 1) or (abs(last[2]-new[2]) > 1):
      return i
    last = new
  return None


# conv to voxel
stl_vox_nn = np.round(stl_vox).astype(np.int)
# stl_vox_nn = np.ceil(stl_vox).astype(np.int)
# stl_vox_nn = np.floor(stl_vox).astype(np.int)
# prune same vox
vox_unique = []
vox_unique.append(tuple(stl_vox_nn[0]))
for i in range(1, stl_vox_nn.shape[0]):
  tmp = tuple(stl_vox_nn[i])
  if mask[tmp]:
    if tmp != vox_unique[-1]:
      vox_unique.append(tmp)

broken_idx = broken(vox_unique)
while broken_idx is not None:
  # broken between vox_unique[broken_idx-1] and vox_unique[broken_idx]
  before = vox_unique[broken_idx-1]
  after = vox_unique[broken_idx]

  axis = np.argsort(np.abs(np.array(before) - np.array(after)))
  direc = np.sign(after[axis[2]] - before[axis[2]])
  candidate = np.array(before)
  candidate[axis[2]] += direc
  candidate = tuple(candidate)

  if ~mask[candidate]:
    direc2 = np.sign(after[axis[1]] - before[axis[1]])
    candidate2 = np.array(candidate)
    candidate2[axis[1]] += direc2
    candidate2 = tuple(candidate2)
    candidate = candidate2

  if ~mask[candidate]:
    print('candidate2 not in mask')
    broken_idx = None
  else:
    vox_unique.insert(broken_idx, candidate)
    broken_idx = broken(vox_unique)

# conv to vertex
path_vert = [vox2vertex[vox] for vox in vox_unique]
# create reverse path
path_vert_inv = path_vert[::-1]


path_stat(g, path_vert, w='neg_log', func=lambda x: x)

# edge_weights = np.array([g.es(g.get_eid(vertices[i], vertices[i+1]))[w] for i in range(len(vertices)-1)])[:,0]
# i=0
# print(i);print(g.get_eid(path_vert[i], path_vert[i+1]));i+=1






