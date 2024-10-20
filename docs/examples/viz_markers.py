"""
======================================================================
Fury Markers
======================================================================

This example shows how to use the marker actor.
"""

import numpy as np

import fury

n = 10000

###############################################################################
# There are nine types 2d markers: circle, square, diamond, triangle, pentagon,
# hexagon, heptagon, cross and plus.

marker_symbols = ["o", "s", "d", "^", "p", "h", "s6", "x", "+"]
markers = [np.random.choice(marker_symbols) for i in range(n)]

centers = np.random.normal(size=(n, 3), scale=10)

colors = np.random.uniform(size=(n, 3))

############################################################################
# You can control the edge color and edge width for each marker

nodes_actor = fury.actor.markers(
    centers,
    marker=markers,
    edge_width=0.1,
    edge_color=[255, 255, 0],
    colors=colors,
    scales=0.5,
)

############################################################################
# In addition, an 3D sphere it's also a valid type of marker

nodes_3d_actor = fury.actor.markers(
    centers + np.ones_like(centers) * 25,
    marker="3d",
    colors=colors,
    scales=0.5,
)

scene = fury.window.Scene()

scene.add(nodes_actor)
scene.add(nodes_3d_actor)

interactive = False

if interactive:
    fury.window.show(scene, size=(600, 600))

fury.window.record(scene=scene, out_path="viz_markers.png", size=(600, 600))
