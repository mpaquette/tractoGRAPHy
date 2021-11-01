import numpy as np

import pylab as pl 




mainpath = '/data/pt_02015/human_test_data_deconvolution_08088.b3/tp0/mrtrix/'


mat_int = np.load(mainpath + 'tc_step_0p5_theta_90_th_0p20_npv_20_seed_int.npy')
mat_int /= mat_int.sum()

mat_wm = np.load(mainpath + 'tc_step_0p5_theta_90_th_0p20_npv_10_seed_wm.npy')
mat_wm /= mat_wm.sum()

mat_path = np.load(mainpath + 'graph_mat_prob.npy')



pl.figure()
pl.subplot(1,3,1)
pl.imshow(mat_int)
pl.title('TC INT seed')
pl.subplot(1,3,2)
pl.imshow(mat_wm)
pl.title('TC WM seed')
pl.subplot(1,3,3)
pl.imshow(mat_path)
pl.title('prob path')
pl.show()



pl.figure()
pl.subplot(1,3,1)
pl.imshow(np.log(mat_int+1e-16))
pl.title('TC INT seed')
pl.subplot(1,3,2)
pl.imshow(np.log(mat_wm+1e-16))
pl.title('TC WM seed')
pl.subplot(1,3,3)
pl.imshow(np.log(mat_path))
pl.title('prob path')
pl.show()





nondiag = np.ones(mat_path.shape, dtype=np.bool)
nondiag[np.diag_indices(mat_path.shape[0])] = False

pl.figure()
pl.subplot(1,3,1)
pl.scatter(mat_int[nondiag], mat_wm[nondiag])
pl.title('int seed vs wm seed')
pl.subplot(1,3,2)
pl.scatter(mat_path[nondiag], mat_wm[nondiag])
pl.title('path vs wm seed')
pl.subplot(1,3,3)
pl.scatter(mat_path[nondiag], mat_int[nondiag])
pl.title('path vs int seed')
pl.show()




idx = np.logical_and(mat_wm[nondiag]>0, mat_int[nondiag]>0)
print(idx.sum())


pl.figure()
pl.subplot(1,3,1)
pl.scatter(np.log(mat_int[nondiag][idx]), np.log(mat_wm[nondiag][idx]))
pl.title('int seed vs wm seed')
pl.subplot(1,3,2)
pl.scatter(np.log(mat_path[nondiag][idx]), np.log(mat_wm[nondiag][idx]))
pl.title('path vs wm seed')
pl.subplot(1,3,3)
pl.scatter(np.log(mat_path[nondiag][idx]), np.log(mat_int[nondiag][idx]))
pl.title('path vs int seed')
pl.show()




# test non zero threshold
# maybe do some multinomial ish stat for what is reliable







