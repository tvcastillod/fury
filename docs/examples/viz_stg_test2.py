"""
This spript includes the basic implementation of the Superquadric Tensor Glyph
using SDFs.
"""

import numpy as np

from fury import actor, window
from fury.actor import _color_fa, _fa
from fury.primitive import prim_sphere


class Sphere:
    vertices = None
    faces = None


def generate_tensors():
    # Inspired by test_peak.py generate_peaks
    vecs10 = np.array([[-.6, .5, -.6], [-.8, -.4, .5], [-.1, -.7, -.7]])
    vecs01 = np.array([[.1, .6, -.8], [.6, .5, .5], [-.8, .6, .3]])
    vecs11 = np.array([[.7, .5, -.5], [0, -.7, -.7], [-.7, .6, -.5]])
    vecs21 = np.array([[.7, -.3, -.6], [.2, -.8, .6], [.7, .6, .5]])

    tensors_vecs = np.zeros((3, 2, 1, 3, 3))

    tensors_vecs[1, 0, 0, :, :] = vecs10
    tensors_vecs[0, 1, 0, :, :] = vecs01
    tensors_vecs[1, 1, 0, :, :] = vecs11
    tensors_vecs[2, 1, 0, :, :] = vecs21

    tensors_vals = np.zeros((3, 2, 1, 3))

    tensors_vals[1, 0, 0, :] = np.array([.2, .3, .3])
    tensors_vals[0, 1, 0, :] = np.array([.4, .7, .2])
    tensors_vals[1, 1, 0, :] = np.array([.8, .4, .3])
    tensors_vals[2, 1, 0, :] = np.array([.7, .5, .1])
    return tensors_vecs, tensors_vals


if __name__ == '__main__':
    # Default tensor setup
    vertices, faces = prim_sphere('repulsion724', True)

    sphere = Sphere()
    sphere.vertices = vertices
    sphere.faces = faces

    mevecs, mevals = generate_tensors()

    dt_affine = np.eye(4)
    #dt_affine[0, 3] = -1

    tensor_actor = actor.tensor_slicer(mevals, mevecs, affine=dt_affine,
                                       sphere=sphere, scale=.5)

    mevecs_shape = mevals.shape

    tensor_actor.display_extent(0, mevecs_shape[0], 0, mevecs_shape[1],
                                0, mevecs_shape[2])

    # Peak setup

    p_affine = np.eye(4)
    p_affine[1, 3] = 2

    peak_actor = actor.peak(mevecs, peaks_values=mevals, affine=p_affine)

    # FURY's glyphs standardization

    valid_mask = np.abs(mevecs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)

    centers = np.asarray(indices).T
    num_centers = centers.shape[0]

    # It seems that a more standard parameter for glyph information would be
    # Degrees of Freedom (DoF). According to this:
    # https://en.wikipedia.org/wiki/Gordon_Kindlmann

    dofs_vecs = mevecs[indices]
    dofs_vals = mevals[indices]

    colors = [_color_fa(_fa(dofs_vals[i, :]), dofs_vecs[i, ...])
              for i in range(num_centers)]
    colors = np.asarray(colors)

    # max_vals = dofs_vals.max(axis=-1)
    argmax_vals = dofs_vals.argmax(axis=-1)
    max_vals = dofs_vals[np.arange(len(dofs_vals)), argmax_vals]
    max_vecs = dofs_vecs[np.arange(len(dofs_vals)), argmax_vals, :]

    # SDF Superquadric Tensor Glyph setup

    # Rectangle version

    # Box version

    box_centers = centers
    box_centers[:, 1] += -2

    # Asymmetric box version

    # Symmetric box version

    box_sd_stg_actor = actor.box(box_centers, colors=colors, scales=max_vals)

    # Billboard version

    # TODO: Add billboard version
    # centers (N, 3)
    # dof (N, 3, 4) ((x, y, z), eigenvalue)
    # dof_vals (N, 3)

    # Scene setup
    scene = window.Scene()

    scene.add(tensor_actor)
    scene.add(peak_actor)
    scene.add(box_sd_stg_actor)

    scene.reset_camera()
    scene.reset_clipping_range()

    window.show(scene, reset_camera=False)