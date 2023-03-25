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

def display_data():
    my_data = genfromtxt('tensor_data/data.csv', delimiter=',')
    mean_data = np.mean(my_data, axis=1)
    mean_data = np.reshape(mean_data, (4, 6))
    data = {
        "slice POLY": mean_data[0],
        "slice SDF": mean_data[1],
        "  roi SDF": mean_data[2],
        "brain SDF": mean_data[3],
    }
    print("\nmean fps data")
    for key in data:
        print(key, ' : ', data[key])

def timer_callback(_obj, _event):
    cnt = next(counter)
    if cnt % 40 == 0:  # 80, 40, 10 (valores para slice, roi y brain)
        frame_rates.append(showm.frame_rate)

        #showm.add_timer_callback(True, 1, timer_callback)
    showm.scene.azimuth(10)
    showm.render()

    if cnt == 2500:  # 5000, 2500, 625 (valores para slice, roi y brain)
        showm.exit()  # termina la animación

if __name__ == '__main__':
    evecs, _ = load_nifti("tensor_data/roi_evecs.nii.gz")
    evals, _ = load_nifti("tensor_data/roi_evals.nii.gz")
    print(evals.shape)
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

    sc = .6
    scene = window.Scene()

    tensors = actor.ellipsoid(axes=dofs_vecs, lengths=dofs_vals, centers=centers,
                              scales=sc, colors=colors)
    scene.add(tensors)
    '''   
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
    '''
    scene.pitch(87)
    #tensor_slicer_actor = actor.tensor_slicer(evals, evecs, affine=dt_affine,
    #                                   mask=mask, sphere=sphere, scale=.3)
    #scene.add(tensor_slicer_actor)

    scene.reset_camera()
    scene.reset_clipping_range()
    window.show(scene, reset_camera=False)

    showm = window.ShowManager(scene, size=(500, 500), reset_camera=False,
                               order_transparent=True)
    showm.initialize()
    global counter, frame_rates
    counter = itertools.count()
    frame_rates = []

    showm.add_timer_callback(True, 1, timer_callback)
    showm.start()

    print(frame_rates)
    print(np.mean(frame_rates, axis=0))

    with open('tensor_data/data.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(frame_rates)

    #display_data()
