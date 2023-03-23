from fury import actor, window, utils
from fury.actor import _color_fa, _fa
from fury.actors.peak import (PeakActor, _orientation_colors,
                              _peaks_colors_from_points, _points_to_vtk_cells)
from fury.lib import numpy_support

import numpy as np
import numpy.testing as npt


def generate_ellipsoids():
    vecs10 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vecs01 = np.array([[.87, .5, 0], [-.5, .87, 0], [0, 0, 1]])
    vecs11 = np.array([[.71, .71, 0], [-.71, .71, 0], [0, 0, 1]])
    vecs21 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    axes = np.zeros((3, 2, 1, 3, 3))

    axes[1, 0, 0, :, :] = vecs10
    axes[0, 1, 0, :, :] = vecs01
    axes[1, 1, 0, :, :] = vecs11
    axes[2, 1, 0, :, :] = vecs21

    lens = np.zeros((3, 2, 1, 3))
    lens[1, 0, 0, :] = np.array([2, 1, .5])
    lens[0, 1, 0, :] = np.array([2, 1, .5])
    lens[1, 1, 0, :] = np.array([2, 1, .5])
    lens[2, 1, 0, :] = np.array([2, 1, .5])
    return axes, lens


def test_main(interactive=False):
    axes, lens = generate_ellipsoids()
    valid_mask = np.abs(axes).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)
    centers = np.asarray(indices).T
    centers[:, 1] += -2

    lengths = np.array([lens[0][1][0], lens[1][0][0],
                        lens[1][1][0], lens[2][1][0]])
    axes = np.array([axes[0][1][0], axes[1][0][0],
                     axes[1][1][0], axes[2][1][0]])
    colors = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 1, 0]])

    ellipsoid_actor = actor.ellipsoid(centers=centers, axes=axes,
                                      lengths=lengths, scales=1,
                                      colors=colors)

    scene = window.Scene()
    scene.background((255, 255, 255))
    scene.add(ellipsoid_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene, reset_camera=False)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, find_objects=True, colors=colors)
    npt.assert_equal(report.objects, 4)
