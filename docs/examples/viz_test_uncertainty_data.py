"""
This spript includes the implementation of cone of uncertainty using matrix
perturbation analysis
"""
import numpy as np
from fury import actor, window

from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

import dipy.reconst.dti as dti
from dipy.data import get_fnames

from fury.actor import _color_fa, _fa

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

data, affine = load_nifti(hardi_fname)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

from dipy.segment.mask import median_otsu

maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3,
                             numpass=1, autocrop=True, dilate=2)

tenmodel = dti.TensorModel(gtab)

tenfit = tenmodel.fit(maskdata)
evals = tenfit.evals[:, 50:, 28:29]
evecs = tenfit.evecs[:, 50:, 28:29]

d = tenfit.quadratic_form
tensor_vals = dti.lower_triangular(tenfit.quadratic_form)
dti_params = dti.eig_from_lo_tri(tensor_vals)

signal = tenmodel.predict(dti_params)
signalOI = signal[:, 50:, 28:29]  # signal data of one slice
b_matrix = tenmodel.design_matrix  # B-matrix of size: [160, 7]

valid_mask = np.abs(evecs).max(axis=(-2, -1)) > 0
indices = np.nonzero(valid_mask)

dofs_vecs = evecs[indices]
dofs_vals = evals[indices]
signal = signalOI[indices]  # Signal S of size: [900, 160]

# Uncertainty calculations ----------------------------------------------------

# s_variance = np.var(signal, axis=0)
# s_mean = np.mean(signal, axis=0)
# cov_e = s_variance/s_mean  # covariance matrix of e
# sigma_e = np.dot(np.dot(np.transpose(b_matrix), np.diag(cov_e)), b_matrix)

import dipy.denoise.noise_estimate as ne
sigma = ne.estimate_sigma(data) # standard deviation of the noise of size [160]

# Angles for cone of uncertainty ----------------------------------------------

angles = np.ones(dofs_vecs.shape[0])
for i in range(dofs_vecs.shape[0]):

    sigma_e = np.diag(signal[i]/sigma**2)  # np.diag(cov_e)
    k = np.dot(np.transpose(b_matrix), sigma_e)
    sigma_ = np.dot(k, b_matrix)
    delta_D = np.linalg.inv(sigma_[:3, :3])
    '''
    dd = np.diag(sigma_)
    delta_DD = dti.from_lower_triangular(np.array([dd[0], dd[3], dd[1], dd[4], dd[5], dd[2]]))
    delta_D = np.linalg.inv(delta_DD)
    '''

    D_ = dofs_vecs
    eigen_vals = dofs_vals[i]

    e1, e2, e3 = np.array(D_[i, :, 0]), np.array(D_[i, :, 1]),\
                 np.array(D_[i, :, 2])
    lambda1, lambda2, lambda3 = eigen_vals[0], eigen_vals[1], eigen_vals[2]

    if (lambda1 > lambda2 and lambda1 > lambda3):
        # The perturbation of the eigenvector associated with the largest
        # eigenvalue is given by
        a = np.dot(np.outer(np.dot(e1, delta_D), np.transpose(e2)) /
                   (lambda1 - lambda2), e2)
        b = np.dot(np.outer(np.dot(e1, delta_D), np.transpose(e3)) /
                   (lambda1 - lambda3), e3)
        delta_e1 = a + b

        # The angle \theta between the perturbed principal eigenvector of D
        theta = np.arctan(np.linalg.norm(delta_e1))
        angles[i] = theta
    else:
        theta = 0.0872665

centers = np.asarray(indices).T
num_centers = centers.shape[0]
colors = [_color_fa(_fa(dofs_vals[i, :]), dofs_vecs[i, ...])
          for i in range(num_centers)]
colors = np.asarray(colors)

sc = .6
uncertainty_cones = actor.doubleCone(axes=dofs_vecs, lengths=dofs_vals,
                                     angles=angles, centers=centers, scales=sc,
                                     colors=colors)

scene = window.Scene()
scene.background([255, 255, 255])

scene.add(uncertainty_cones)

scene.reset_camera()
scene.reset_clipping_range()

window.show(scene, reset_camera=False)