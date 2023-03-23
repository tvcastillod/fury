from fury import actor, window, utils
from fury.actor import _color_fa, _fa
from fury.actors.peak import (PeakActor, _orientation_colors,
                              _peaks_colors_from_points, _points_to_vtk_cells)
from fury.lib import numpy_support

import numpy as np
import numpy.testing as npt


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

def test_main(interactive = True):
    mevecs, mevals = generate_tensors()
    valid_mask = np.abs(mevecs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)
    centers = np.asarray(indices).T
    box_centers = centers
    box_centers[:, 1] += -2
    num_centers = centers.shape[0]
    dofs_vecs = mevecs[indices]
    dofs_vals = mevals[indices]
    colors = [_color_fa(_fa(dofs_vals[i, :]), dofs_vecs[i, ...])
              for i in range(num_centers)]
    colors = np.asarray(colors)
    eigen_vals = np.array([mevals[0][1][0], mevals[1][0][0],
                           mevals[1][1][0], mevals[2][1][0]])
    eigen_vecs = np.array([mevecs[0][1][0], mevecs[1][0][0],
                           mevecs[1][1][0], mevecs[2][1][0]])

    ellipsoid_actor = actor.ellipsoid(eigen_vecs, eigen_vals, box_centers,
                                       1, colors, 1)

    scene = window.Scene()
    scene.background((255, 255, 255))
    scene.add(ellipsoid_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene, reset_camera=False)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 4)
