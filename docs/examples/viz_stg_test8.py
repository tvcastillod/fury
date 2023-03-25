"""
Keyframe animation tensors
"""
import itertools

import numpy as np
from numpy import genfromtxt
import csv
from fury import actor, window
from fury.actor import _color_fa, _fa
from fury.primitive import prim_sphere
from dipy.io.image import load_nifti


class Sphere:
    vertices = None
    faces = None

if __name__ == '__main__':
    evecs, affine = load_nifti("tensor_data/slice_evecs.nii.gz")
    evals, affine = load_nifti("tensor_data/slice_evals.nii.gz")

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
    scene = window.Scene()
    #'''
    tensors = actor.ellipsoid(axes=mevecs,
                              lengths=mevals,
                              centers=centers, scales=sc,
                              colors=colors)
    scene.add(tensors)
    #'''
    vertices, faces = prim_sphere('repulsion724', True)

    sphere = Sphere()
    sphere.vertices = vertices
    sphere.faces = faces

    dt_affine = np.eye(4)
    mask = np.ones((evals.shape[:3]))

    tensor_slicer_actor = actor.tensor_slicer(evals, evecs, affine=dt_affine,
                                       mask=mask, sphere=sphere, scale=.3)

    # Scene setup
    scene.add(tensor_slicer_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    #window.show(scene, reset_camera=False)

    def timer_callback(_obj, _event):
        global timer_id
        cnt = next(counter)
        if cnt % 167 == 0:
            frame_rates.append(showm.frame_rate)
            showm.destroy_timer(timer_id)
            timer_id = showm.add_timer_callback(True, 1, timer_callback)
        showm.scene.azimuth(10)
        showm.render()

        if cnt == 10000:
            showm.exit()

    showm = window.ShowManager(scene, size=(500, 500), reset_camera=False,
                               order_transparent=True)

    counter = itertools.count()
    frame_rates = []

    timer_id = showm.add_timer_callback(True, 1, timer_callback)

    showm.start()

    with open('tensor_data/data.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(frame_rates)

    '''
    my_data = genfromtxt('tensor_data/data.csv', delimiter=',')
    print(np.mean(my_data, axis=1))
    '''