"""
Keyframe animation tensors
"""
import csv
import itertools
import os
from datetime import timedelta
from time import time
from tracemalloc import start

import numpy as np
from dipy.data.fetcher import dipy_home
from dipy.io.image import load_nifti
from numpy import genfromtxt

from fury import actor, window
from fury.actor import _color_fa, _fa
from fury.primitive import prim_sphere


class Sphere:
    vertices = None
    faces = None


def timer_callback(_obj, _event):
    global frame_rates, prev_time, start_time, showm
    time_diff = timedelta(seconds=time() - start_time)
    # Runs for 15 seconds
    if time_diff.seconds > 15:
        showm.exit()
    else:
        if time_diff.seconds > prev_time:
            print(time_diff)
            frame_rates.append(showm.frame_rate)
        prev_time = time_diff.seconds
        showm.scene.azimuth(10)
        showm.render()


if __name__ == '__main__':
    global frame_rates, prev_time, start_time, showm
    
    dataset_dir = os.path.join(dipy_home, 'stanford_hardi')
    
    scene = window.Scene()
    
    #evecs, _ = load_nifti(os.path.join(dataset_dir, 'roi_evecs.nii.gz'))
    #evals, _ = load_nifti(os.path.join(dataset_dir, 'roi_evals.nii.gz'))
    evecs, _ = load_nifti("tensor_data/slice_evecs.nii.gz")
    evals, _ = load_nifti("tensor_data/slice_evals.nii.gz")

    affine = np.eye(4)

    valid_mask = np.abs(evecs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)

    centers = np.asarray(indices).T

    num_centers = centers.shape[0]

    dofs_vecs = evecs[indices]
    dofs_vals = evals[indices]

    colors = [
        _color_fa(_fa(dofs_vals[i, :]), dofs_vecs[i, ...])
        for i in range(num_centers)]
    colors = np.asarray(colors)

    """
    x, y, s, z = evals.shape
    mevals = evals[:, :, :].reshape((x * y * s, z))
    x, y, s, k, l = evecs.shape
    mevecs = evecs[:, :, :].reshape((x * y * s, k, l))
    """
    
    #tensors = actor.ellipsoid(
    #    centers, colors=colors, axes=dofs_vecs, lengths=dofs_vals, scales=.6)
    #scene.add(tensors)
    
    #window.show(scene)


    data_shape = evals.shape[:3]
    mask = np.ones((data_shape)).astype(bool)
    vertices, faces = prim_sphere('repulsion100', True)
    #vertices, faces = prim_sphere('repulsion200', True)
    #vertices, faces = prim_sphere('repulsion724', True)
    sphere = Sphere()
    sphere.vertices = vertices
    sphere.faces = faces

    tensor_slicer_actor = actor.tensor_slicer(
        evals, evecs, affine=affine, mask=mask, sphere=sphere, scale=.3)
    tensor_slicer_actor.display_extent(
        0, data_shape[0], 0, data_shape[1], 0, data_shape[2])
    scene.add(tensor_slicer_actor)

    
    scene.reset_camera()
    scene.reset_clipping_range()

    #window.show(scene)

    showm = window.ShowManager(
        scene, size=(500, 500), reset_camera=False, order_transparent=True)

    prev_time = 0
    
    frame_rates = []

    # Running every 1/25 = 40 ms
    timer_id = showm.add_timer_callback(True, 40, timer_callback)

    start_time = time()
    
    showm.start()

    with open(
        'tensor_data/data.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(frame_rates)

    """
    my_data = genfromtxt(os.path.join(dataset_dir, 'data.csv'), delimiter=',')
    print(np.mean(my_data, axis=1))
    """
