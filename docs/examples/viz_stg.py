import numpy as np

from fury import actor, window
from fury.primitive import prim_sphere


class Sphere:
    vertices = None
    faces = None


if __name__ == '__main__':
    evals = np.array([1.4, .35, .35]) * 10 ** (-3)
    evecs = np.eye(3)

    mevals = np.zeros((2, 2, 1, 3))
    mevecs = np.zeros((2, 2, 1, 3, 3))

    mevals[..., :] = evals
    mevecs[..., :, :] = evecs

    # Default tensor setup
    vertices, faces = prim_sphere('symmetric724', True)

    sphere = Sphere()
    sphere.vertices = vertices
    sphere.faces = faces

    dt_affine = np.eye(4)
    dt_affine[0, 3] = -1

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

    valid_idx_dirs = mevecs[indices]

    valid_dirs = valid_idx_dirs[
                 :, ~np.all(np.abs(valid_idx_dirs).max(axis=-1) == 0, axis=0),
                 :]

    # SDF Superquadric Tensor Glyph setup

    # Box version
    box_centers = centers
    box_centers[:, 0] += 1

    box_sd_stg_actor = actor.box(box_centers, scales=1)

    # Billboard version

    #TODO: Add billboard version

    # Scene setup
    scene = window.Scene()

    scene.add(tensor_actor)
    scene.add(peak_actor)
    scene.add(box_sd_stg_actor)

    scene.reset_camera()
    scene.reset_clipping_range()

    window.show(scene, reset_camera=False)

    # TODO: Add colorfa test here as previous test moved to DIPY
