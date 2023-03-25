"""
Test for ellipsoid Tensor Glyph implementation using SDFs.
"""

import numpy as np

from fury import actor, window
from fury.actor import _color_fa, _fa

from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.segment.mask import median_otsu
import dipy.reconst.dti as dti

from fury.primitive import prim_sphere


class Sphere:
    vertices = None
    faces = None


if __name__ == '__main__':
    evecs, affine = load_nifti("tensor_data/roi_evecs.nii.gz")
    evals, affine = load_nifti("tensor_data/roi_evals.nii.gz")

    dt_affine = np.eye(4)

    valid_mask = np.abs(evecs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)

    centers = np.asarray(indices).T

    num_centers = centers.shape[0]

    dofs_vecs = evecs[indices]
    dofs_vals = evals[indices]

    colors = [_color_fa(_fa(dofs_vals[i, :]), dofs_vecs[i, ...])
              for i in range(num_centers)]
    colors = np.asarray(colors)

    # NEW DATA
    x, y, s, z = evals.shape
    mevals = evals[:, :, :].reshape((x * y * s, z))
    x, y, s, k, l = evecs.shape
    mevecs = evecs[:, :, :].reshape((x * y * s, k, l))

    sc = .6
    box_sd_stg_actor = actor.ellipsoid(axes=mevecs, lengths=mevals,
                                       centers=centers, scales=sc,
                                       colors=colors)

    vertices, faces = prim_sphere('repulsion724', True)

    sphere = Sphere()
    sphere.vertices = vertices
    sphere.faces = faces

    dt_affine = np.eye(4)
    mask = np.ones((evals.shape[:3]))
    tensor_slicer_actor = actor.tensor_slicer(evals, evecs, affine=dt_affine,
                                       sphere=sphere, scale=.3, mask=mask)

    # Scene setup
    scene = window.Scene()

    #scene.add(box_sd_stg_actor)
    scene.add(tensor_slicer_actor)

    scene.reset_camera()
    scene.reset_clipping_range()

    window.show(scene, reset_camera=False)
