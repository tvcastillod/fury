"""
This spript includes the implementation of cone of uncertainty using matrix
perturbation analysis
"""
import numpy as np

from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

import dipy.reconst.dti as dti

from dipy.data import get_fnames

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

data, affine = load_nifti(hardi_fname)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

from dipy.segment.mask import median_otsu

maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3,
                             numpass=1, autocrop=True, dilate=2)

tenmodel = dti.TensorModel(gtab)

tenfit = tenmodel.fit(maskdata)
evals = tenfit.evals[13:43, 44:74, 28:29]
evecs = tenfit.evecs[13:43, 44:74, 28:29]

d = tenfit.quadratic_form
tensor_vals = dti.lower_triangular(tenfit.quadratic_form)
dti_params = dti.eig_from_lo_tri(tensor_vals)

signal = tenmodel.predict(dti_params)
signalOI = signal[13:43, 44:74, 28:29]  # signal data of one slice
# tenfit.predict(gtab)
# dti.tensor_prediction(dti_params, gtab, 1)
b_matrix = tenmodel.design_matrix  # B-matrix of size: [160, 7]

valid_mask = np.abs(evecs).max(axis=(-2, -1)) > 0
indices = np.nonzero(valid_mask)

dofs_vecs = evecs[indices]
dofs_vals = evals[indices]
signal = signalOI[indices] # Signal S of size: [900, 160]

# Uncertainty calculations ----------------------------------------------------

s_variance = np.var(signal, axis=0)
s_mean = np.mean(signal, axis=0)
cov_e = s_variance/s_mean  # covariance matrix of e
sigma_e = np.dot(np.dot(np.transpose(b_matrix), np.diag(cov_e)), b_matrix)

'''
import dipy.denoise.noise_estimate as ne
sigma = ne.estimate_sigma(data) # standard deviation of the noise of size: [160]
#pertubation_matrix = s_mean/sigma
'''

# Angles for cone of uncertainty ----------------------------------------------

angles = np.ones(dofs_vecs.shape[0])
for i in range(dofs_vecs.shape[0]):
    '''
    sigma_e = np.diag(cov_e) #signal[i]/sigma**2
    k = np.dot(np.transpose(b_matrix), sigma_e)
    sigma_ = np.dot(k, b_matrix)
    delta_D = dti.from_lower_triangular(sigma_[:-1, :-1])
    '''
    D_ = dofs_vecs
    eigen_vals = dofs_vals[i]

    e1, e2, e3 = np.array(D_[i, :, 0]), np.array(D_[i, :, 1]), np.array(D_[i, :, 2])
    lambda1, lambda2, lambda3 = eigen_vals[0], eigen_vals[1], eigen_vals[2]
    delta_D = dofs_vecs[i]  # temp delta_D 3x3 matrix

    # The perturbation of the eigenvector associated with the largest eigenvalue is given by
    a = np.dot(np.outer(np.dot(e1, delta_D), np.transpose(e2))/ (lambda1 - lambda2), e2)
    b = np.dot(np.outer(np.dot(e1, delta_D), np.transpose(e3))/ (lambda1 - lambda3), e3)
    delta_e1 = a + b

    # The angle \theta between the perturbed principal eigenvector of D
    theta = np.arctan(np.linalg.norm(delta_e1))
