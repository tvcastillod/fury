"""
This script includes ODF implementation with sdf definition.
"""
import os
import numpy as np
from dipy.data.fetcher import dipy_home
from dipy.io.image import load_nifti
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf

from fury import actor, window

if __name__ == "__main__":
    show_man = window.ShowManager(size=(1920, 1080))
    show_man.scene.background((1, 1, 1))

    coeffs = np.array([
        [
            0.2820735, 0.15236554, -0.04038717, -0.11270988, -0.04532376,
            0.14921817, 0.00257928, 0.0040734, -0.05313807, 0.03486542,
            0.04083064, 0.02105767, -0.04389586, -0.04302812, 0.1048641
        ],
        [
            0.28549338, 0.0978267, -0.11544838, 0.12525354, -0.00126003,
            0.00320594, 0.04744155, -0.07141446, 0.03211689, 0.04711322,
            0.08064896, 0.00154299, 0.00086506, 0.00162543, -0.00444893
        ],
        [
            0.28208936, -0.13133252, -0.04701012, -0.06303016, -0.0468775,
            0.02348355, 0.03991898, 0.02587433, 0.02645416, 0.00668765,
            0.00890633, 0.02189304, 0.00387415, 0.01665629, -0.01427194
        ],
        [
            -0.2739740312099, 0.2526670396328, 1.8922271728516,
            0.2878578901291, -0.5339795947075, -0.2620058953762,
            0.1580424904823, 0.0329004973173, -0.1322413831949,
            -0.1332057565451, 1.0894461870193, -0.6319401264191,
            -0.0416776277125, -1.0772529840469,  0.1423762738705
        ]
    ])

    # fmt: on
    centers = np.array([[0, -1, 0], [1, -1, 0], [2, -1, 0], [3, -1, 0]])
    scales =  np.ones(4) * .5

    odf_actor = actor.odf_impl(centers, coeffs, "repulsion100", scales, 1.0)

    show_man.scene.add(odf_actor)

    sphere = get_sphere("symmetric724")

    sh_basis = "descoteaux07"
    sh_order = 4

    sh = np.zeros((4, 1, 1, 15))
    sh[0, 0, 0, :] = coeffs[0, :]
    sh[1, 0, 0, :] = coeffs[1, :]
    sh[2, 0, 0, :] = coeffs[2, :]
    sh[3, 0, 0, :] = coeffs[3, :]

    tensor_sf = sh_to_sf(
        sh, sh_order_max=sh_order, basis_type=sh_basis, sphere=sphere, legacy=True
    )
    odf_slicer_actor = actor.odf_slicer(tensor_sf, sphere=sphere, norm=True)

    show_man.scene.add(odf_slicer_actor)

    show_man.start()

    show_man = window.ShowManager(size=(1920, 1080))
    show_man.scene.background((1, 1, 1))
    dataset_dir = os.path.join(dipy_home, "stanford_hardi")

    coeffs, affine = load_nifti("docs/experimental/odf_slice_1.nii")

    valid_mask = np.abs(coeffs).max(axis=(-1)) > 0
    indices = np.nonzero(valid_mask)

    centers = np.asarray(indices).T

    x, y, z, s = coeffs.shape
    coeffs = coeffs[:, :, :].reshape((x * y * z, s))
    odf_actor = actor.odf_impl(centers=centers[:50], coeffs=coeffs[:50])

    show_man.scene.add(odf_actor)
    show_man.start()
