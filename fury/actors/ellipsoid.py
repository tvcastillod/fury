import os

import numpy as np

from fury import actor
from fury.colormap import create_colormap
from fury.shaders import (attribute_to_actor, import_fury_shader,
                          shader_to_actor, compose_shader)
from fury.utils import (set_polydata_vertices, set_polydata_triangles,
                        set_polydata_colors, apply_affine)
from fury.lib import Actor, PolyData, PolyDataMapper


class EllipsoidActor(Actor):
    """
    VTK actor for visualizing Ellipsoids.

    Parameters
    ----------
    axes : ndarray (3, 3) or (N, 3, 3)
        Axes lengths
    lengths : ndarray (3, ) or (N, 3)
        Orientation of the principal axes
    centers : ndarray(N, 3)
        Ellipsoid positions
    scales : int or ndarray (N, ), optional
        Ellipsoid size, default(1)
    colors : ndarray (3, ) or (N, 3)
        RGB colors used to show the ellipsoids'
    opacity : float
        Takes values from 0 (fully transparent) to 1 (opaque).

    """
    def __init__(self, axes, lengths, centers, scales, colors, opacity):
        self.axes = axes
        self.lengths = lengths
        self.centers = centers
        self.scales = scales
        self.colors = colors
        self.set_opacity(opacity)

    def set_opacity(self, opacity):
        """
        Set opacity value of Ellipsoid to display.
        """
        self.GetProperty().SetOpacity(opacity)
