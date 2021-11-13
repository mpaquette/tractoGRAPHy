import numpy as np


# dist on sphere
def sphPDF(k, mu, direc):
	# Generate the PDF for a Von-Mises Fisher distribution p=3
	# at locations direc for concentration k and mean orientation mu
	C3 = k / (2*np.pi*(np.exp(k)-np.exp(-k)))
	tmp = np.exp(k*np.dot(direc,mu[:,None])).squeeze()
	return C3*tmp

# antipodally symetric
# If building a multi-peak ODF, 
# you might want un-normed indivual peak
# this will allow to mak-norm them before averaging
# to control reltaive size of the maximas
def sphPDF_sym(k, mu, direc, norm=False):
	d1 = sphPDF(k, mu, direc)
	d2 = sphPDF(k, mu, -direc)
	dd1 = (d1+d2)/2.
	if norm:
		dd1 = dd1/dd1.sum()
	return dd1




# pts_angle = np.linspace(0, 2*np.pi, 180, endpoint=False)
pts_angle = np.linspace(0, 2*np.pi, 361)


pts_3d = np.zeros((pts_angle.shape[0],3))
pts_3d[:,0] = np.cos(pts_angle)
pts_3d[:,1] = np.sin(pts_angle)



mu1 = np.array([1.4, 0.1, 0])
mu2 = np.array([0.1, 0.8, 0])

mu1 = mu1 / np.linalg.norm(mu1)
mu2 = mu2 / np.linalg.norm(mu2)

k1 = 8
k2 = 10

sf1 = sphPDF_sym(k1, mu1, pts_3d, norm=True)
sf2 = sphPDF_sym(k2, mu2, pts_3d, norm=True)
sf = (sf1+sf2)/2


import pylab as pl




pl.figure()
pl.polar(pts_angle, sf, color='black', linewidth=4)
pl.gca().set_yticklabels([])
pl.gca().axis('off')


pl.axvline(np.pi/4, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+2*+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+3*+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')

# pl.show()

p_0 = sf[np.logical_and(pts_angle>=np.pi/4, pts_angle<np.pi/4+np.pi/2)].sum()
p_3 = sf[np.logical_and(pts_angle>=np.pi/4+np.pi/2, pts_angle<np.pi/4+2*+np.pi/2)].sum()
p_2 = sf[np.logical_and(pts_angle>=np.pi/4+2*+np.pi/2, pts_angle<np.pi/4+3*+np.pi/2)].sum()
p_1 = sf[pts_angle<np.pi/4].sum() + sf[pts_angle>=np.pi/4+3*+np.pi/2].sum() - sf[-1]

p_tot = p_0 + p_1 + p_2 + p_3 


# print('Cone orientation = {:.1f} deg'.format(cone_orientation*180/(np.pi)))
# print('     {:.1f}'.format(100*p_0/p_tot))
# print('{:.1f}      {:.1f}'.format(100*p_3/p_tot, 100*p_1/p_tot))
# print('     {:.1f}'.format(100*p_2/p_tot))

pl.text(0.5, 1.0, "{:.1f}%".format(100*p_0/p_tot),
      horizontalalignment='center',
      verticalalignment='bottom',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(1.0, 0.5, "{:.1f}%".format(100*p_1/p_tot),
      horizontalalignment='left',
      verticalalignment='center',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(0.5, 0.0, "{:.1f}%".format(100*p_2/p_tot),
      horizontalalignment='center',
      verticalalignment='top',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(0.0, 0.5, "{:.1f}%".format(100*p_3/p_tot),
      horizontalalignment='right',
      verticalalignment='center',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

# pl.show()



pl.savefig('/home/paquette/Documents/'+'odf_2D.png',
			dpi=300,
			pad_inches=0.25,
			transparent=True,
			bbox_inches='tight')

pl.close()




cone_half_angle = (70*np.pi/180)

pl.figure()
cone_orientation = np.pi/2
# mask = np.abs((pts_angle - cone_orientation)) < cone_half_angle
mask = np.minimum(np.abs(pts_angle - cone_orientation), 2*np.pi-np.abs(pts_angle - cone_orientation)) < cone_half_angle



# pl.polar(pts_angle[mask], sf[mask], color='black', linewidth=4)
sf_mask = sf.copy()
sf_mask[~mask] = 0
pl.polar(pts_angle, sf_mask, color='black', linewidth=4)
pl.gca().set_yticklabels([])
pl.gca().axis('off')

pl.axvline(np.pi/4, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+2*+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+3*+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')

pl.axvline(cone_orientation+cone_half_angle, linewidth=2, alpha=0.9, color='red')
pl.axvline(cone_orientation-cone_half_angle, linewidth=2, alpha=0.9, color='red')





p_0 = sf_mask[np.logical_and(pts_angle>=np.pi/4, pts_angle<np.pi/4+np.pi/2)].sum()
p_3 = sf_mask[np.logical_and(pts_angle>=np.pi/4+np.pi/2, pts_angle<np.pi/4+2*+np.pi/2)].sum()
p_2 = sf_mask[np.logical_and(pts_angle>=np.pi/4+2*+np.pi/2, pts_angle<np.pi/4+3*+np.pi/2)].sum()
p_1 = sf_mask[pts_angle<np.pi/4].sum() + sf_mask[pts_angle>=np.pi/4+3*+np.pi/2].sum() - sf_mask[-1]

p_tot = p_0 + p_1 + p_2 + p_3 


# print('Cone orientation = {:.1f} deg'.format(cone_orientation*180/(np.pi)))
# print('     {:.1f}'.format(100*p_0/p_tot))
# print('{:.1f}      {:.1f}'.format(100*p_3/p_tot, 100*p_1/p_tot))
# print('     {:.1f}'.format(100*p_2/p_tot))

pl.text(0.5, 1.0, "{:.1f}%".format(100*p_0/p_tot),
      horizontalalignment='center',
      verticalalignment='bottom',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(1.0, 0.5, "{:.1f}%".format(100*p_1/p_tot),
      horizontalalignment='left',
      verticalalignment='center',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(0.5, 0.0, "{:.1f}%".format(100*p_2/p_tot),
      horizontalalignment='center',
      verticalalignment='top',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(0.0, 0.5, "{:.1f}%".format(100*p_3/p_tot),
      horizontalalignment='right',
      verticalalignment='center',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

# pl.show()





pl.savefig('/home/paquette/Documents/'+'odf_2D_cone0.png',
			dpi=300,
			pad_inches=0.25,
			transparent=True,
			bbox_inches='tight')

pl.close()



pl.figure()

cone_orientation = np.pi
# mask = np.abs((pts_angle - cone_orientation)) < cone_half_angle
mask = np.minimum(np.abs(pts_angle - cone_orientation), 2*np.pi-np.abs(pts_angle - cone_orientation)) < cone_half_angle

# pl.polar(pts_angle[mask], sf[mask], color='black', linewidth=4)
sf_mask = sf.copy()
sf_mask[~mask] = 0
pl.polar(pts_angle, sf_mask, color='black', linewidth=4)
pl.gca().set_yticklabels([])
pl.gca().axis('off')

pl.axvline(np.pi/4, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+2*+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+3*+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')

pl.axvline(cone_orientation+cone_half_angle, linewidth=2, alpha=0.9, color='red')
pl.axvline(cone_orientation-cone_half_angle, linewidth=2, alpha=0.9, color='red')


p_0 = sf_mask[np.logical_and(pts_angle>=np.pi/4, pts_angle<np.pi/4+np.pi/2)].sum()
p_3 = sf_mask[np.logical_and(pts_angle>=np.pi/4+np.pi/2, pts_angle<np.pi/4+2*+np.pi/2)].sum()
p_2 = sf_mask[np.logical_and(pts_angle>=np.pi/4+2*+np.pi/2, pts_angle<np.pi/4+3*+np.pi/2)].sum()
p_1 = sf_mask[pts_angle<np.pi/4].sum() + sf_mask[pts_angle>=np.pi/4+3*+np.pi/2].sum() - sf_mask[-1]

p_tot = p_0 + p_1 + p_2 + p_3 


# print('Cone orientation = {:.1f} deg'.format(cone_orientation*180/(np.pi)))
# print('     {:.1f}'.format(100*p_0/p_tot))
# print('{:.1f}      {:.1f}'.format(100*p_3/p_tot, 100*p_1/p_tot))
# print('     {:.1f}'.format(100*p_2/p_tot))

pl.text(0.5, 1.0, "{:.1f}%".format(100*p_0/p_tot),
      horizontalalignment='center',
      verticalalignment='bottom',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(1.0, 0.5, "{:.1f}%".format(100*p_1/p_tot),
      horizontalalignment='left',
      verticalalignment='center',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(0.5, 0.0, "{:.1f}%".format(100*p_2/p_tot),
      horizontalalignment='center',
      verticalalignment='top',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(0.0, 0.5, "{:.1f}%".format(100*p_3/p_tot),
      horizontalalignment='right',
      verticalalignment='center',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

# pl.show()






pl.savefig('/home/paquette/Documents/'+'odf_2D_cone3.png',
			dpi=300,
			pad_inches=0.25,
			transparent=True,
			bbox_inches='tight')

pl.close()




pl.figure()

cone_orientation = 3*np.pi/2
# mask = np.abs((pts_angle - cone_orientation)) < cone_half_angle
mask = np.minimum(np.abs(pts_angle - cone_orientation), 2*np.pi-np.abs(pts_angle - cone_orientation)) < cone_half_angle

# pl.polar(pts_angle[mask], sf[mask], color='black', linewidth=4)
sf_mask = sf.copy()
sf_mask[~mask] = 0
pl.polar(pts_angle, sf_mask, color='black', linewidth=4)
pl.gca().set_yticklabels([])
pl.gca().axis('off')

pl.axvline(np.pi/4, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+2*+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+3*+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')

pl.axvline(cone_orientation+cone_half_angle, linewidth=2, alpha=0.9, color='red')
pl.axvline(cone_orientation-cone_half_angle, linewidth=2, alpha=0.9, color='red')

p_0 = sf_mask[np.logical_and(pts_angle>=np.pi/4, pts_angle<np.pi/4+np.pi/2)].sum()
p_3 = sf_mask[np.logical_and(pts_angle>=np.pi/4+np.pi/2, pts_angle<np.pi/4+2*+np.pi/2)].sum()
p_2 = sf_mask[np.logical_and(pts_angle>=np.pi/4+2*+np.pi/2, pts_angle<np.pi/4+3*+np.pi/2)].sum()
p_1 = sf_mask[pts_angle<np.pi/4].sum() + sf_mask[pts_angle>=np.pi/4+3*+np.pi/2].sum() - sf_mask[-1]

p_tot = p_0 + p_1 + p_2 + p_3 


# print('Cone orientation = {:.1f} deg'.format(cone_orientation*180/(np.pi)))
# print('     {:.1f}'.format(100*p_0/p_tot))
# print('{:.1f}      {:.1f}'.format(100*p_3/p_tot, 100*p_1/p_tot))
# print('     {:.1f}'.format(100*p_2/p_tot))

pl.text(0.5, 1.0, "{:.1f}%".format(100*p_0/p_tot),
      horizontalalignment='center',
      verticalalignment='bottom',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(1.0, 0.5, "{:.1f}%".format(100*p_1/p_tot),
      horizontalalignment='left',
      verticalalignment='center',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(0.5, 0.0, "{:.1f}%".format(100*p_2/p_tot),
      horizontalalignment='center',
      verticalalignment='top',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(0.0, 0.5, "{:.1f}%".format(100*p_3/p_tot),
      horizontalalignment='right',
      verticalalignment='center',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

# pl.show()





pl.savefig('/home/paquette/Documents/'+'odf_2D_cone2.png',
			dpi=300,
			pad_inches=0.25,
			transparent=True,
			bbox_inches='tight')

pl.close()




pl.figure()

cone_orientation = 0
# mask = np.abs((pts_angle - cone_orientation)) < cone_half_angle
mask = np.minimum(np.abs(pts_angle - cone_orientation), 2*np.pi-np.abs(pts_angle - cone_orientation)) < cone_half_angle

# pl.polar(pts_angle[mask], sf[mask], color='black', linewidth=4)
sf_mask = sf.copy()
sf_mask[~mask] = 0
pl.polar(pts_angle, sf_mask, color='black', linewidth=4)
pl.gca().set_yticklabels([])
pl.gca().axis('off')

pl.axvline(np.pi/4, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+2*+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')
pl.axvline(np.pi/4+3*+np.pi/2, linewidth=2, alpha=0.5, linestyle='dashed')

pl.axvline(cone_orientation+cone_half_angle, linewidth=2, alpha=0.9, color='red')
pl.axvline(cone_orientation-cone_half_angle, linewidth=2, alpha=0.9, color='red')


p_0 = sf_mask[np.logical_and(pts_angle>=np.pi/4, pts_angle<np.pi/4+np.pi/2)].sum()
p_3 = sf_mask[np.logical_and(pts_angle>=np.pi/4+np.pi/2, pts_angle<np.pi/4+2*+np.pi/2)].sum()
p_2 = sf_mask[np.logical_and(pts_angle>=np.pi/4+2*+np.pi/2, pts_angle<np.pi/4+3*+np.pi/2)].sum()
p_1 = sf_mask[pts_angle<np.pi/4].sum() + sf_mask[pts_angle>=np.pi/4+3*+np.pi/2].sum() - sf_mask[-1]

p_tot = p_0 + p_1 + p_2 + p_3 


# print('Cone orientation = {:.1f} deg'.format(cone_orientation*180/(np.pi)))
# print('     {:.1f}'.format(100*p_0/p_tot))
# print('{:.1f}      {:.1f}'.format(100*p_3/p_tot, 100*p_1/p_tot))
# print('     {:.1f}'.format(100*p_2/p_tot))

pl.text(0.5, 1.0, "{:.1f}%".format(100*p_0/p_tot),
      horizontalalignment='center',
      verticalalignment='bottom',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(1.0, 0.5, "{:.1f}%".format(100*p_1/p_tot),
      horizontalalignment='left',
      verticalalignment='center',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(0.5, 0.0, "{:.1f}%".format(100*p_2/p_tot),
      horizontalalignment='center',
      verticalalignment='top',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

pl.text(0.0, 0.5, "{:.1f}%".format(100*p_3/p_tot),
      horizontalalignment='right',
      verticalalignment='center',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=0.0),
      transform=pl.gca().transAxes)

# pl.show()




# pl.show()

pl.savefig('/home/paquette/Documents/'+'odf_2D_cone1.png',
			dpi=300,
			pad_inches=0.25,
			transparent=True,
			bbox_inches='tight')

pl.close()








