import numpy as np
import numpy.testing as npt
from fury.texture.utils import uv_calculations

def test_uv_calculations():
   uv_coords = uv_calculations(1)
   expected_uv1 = np.array([
      [0.001, 0.001], [0.001, 0.999], [0.999, 0.999], [0.999, 0.001],
      [0.001, 0.001], [0.001, 0.999], [0.999, 0.999], [0.999, 0.001]
   ])
   npt.assert_array_almost_equal(uv_coords, expected_uv1, decimal=3)

   uv_coords = uv_calculations(3)
   expected_uv3 = np.array([
      [0.001, 0.667], [0.001, 0.999], [0.999, 0.999], [0.999, 0.667],
      [0.001, 0.667], [0.001, 0.999], [0.999, 0.999], [0.999, 0.667],
      [0.001, 0.334], [0.001, 0.665], [0.999, 0.665], [0.999, 0.334],
      [0.001, 0.334], [0.001, 0.665], [0.999, 0.665], [0.999, 0.334],
      [0.001, 0.001], [0.001, 0.332], [0.999, 0.332], [0.999, 0.001],
      [0.001, 0.001], [0.001, 0.332], [0.999, 0.332], [0.999, 0.001]
   ])
   npt.assert_array_almost_equal(uv_coords, expected_uv3, decimal=3)

   uv_coords = uv_calculations(7)
   expected_uv7 = np.array([
      [0.001, 0.858], [0.001, 0.999], [0.999, 0.999], [0.999, 0.858],
      [0.001, 0.858], [0.001, 0.999], [0.999, 0.999], [0.999, 0.858],
      [0.001, 0.715], [0.001, 0.856], [0.999, 0.856], [0.999, 0.715],
      [0.001, 0.715], [0.001, 0.856], [0.999, 0.856], [0.999, 0.715],
      [0.001, 0.572], [0.001, 0.713], [0.999, 0.713], [0.999, 0.572],
      [0.001, 0.572], [0.001, 0.713], [0.999, 0.713], [0.999, 0.572],
      [0.001, 0.429], [0.001, 0.570], [0.999, 0.570], [0.999, 0.429],
      [0.001, 0.429], [0.001, 0.570], [0.999, 0.570], [0.999, 0.429],
      [0.001, 0.286], [0.001, 0.427], [0.999, 0.427], [0.999, 0.286],
      [0.001, 0.286], [0.001, 0.427], [0.999, 0.427], [0.999, 0.286],
      [0.001, 0.143], [0.001, 0.284], [0.999, 0.284], [0.999, 0.143],
      [0.001, 0.143], [0.001, 0.284], [0.999, 0.284], [0.999, 0.143],
      [0.001, 0.001], [0.001, 0.141], [0.999, 0.141], [0.999, 0.001],
      [0.001, 0.001], [0.001, 0.141], [0.999, 0.141], [0.999, 0.001]
   ])
   npt.assert_array_almost_equal(uv_coords, expected_uv7, decimal=3)
