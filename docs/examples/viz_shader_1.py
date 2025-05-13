# Import the canvas and run function from wgpu's GUI abstraction layer
from wgpu.gui.auto import WgpuCanvas, run

# Import the pygfx rendering and geometry library
import pygfx as gfx


# Create a GPU-backed canvas that will be the drawing surface (a window or offscreen target)
canvas = WgpuCanvas(size=(500, 400))

# Create a renderer that will handle drawing the scene onto the canvas using WebGPU
renderer = gfx.WgpuRenderer(canvas)

# Create a camera that assumes input coordinates are already in Normalized Device Coordinates (no transformation)
camera = gfx.NDCCamera()
# Creates an orthographic camera with specified width and height of the view volume
#camera = gfx.OrthographicCamera(width=2, height=2)



# --------------------------
# Define the triangle object
# --------------------------

# Create a triangle mesh with vertex positions and per-vertex colors
triangle = gfx.Mesh(
    gfx.Geometry(
        indices=[(0, 1, 2)],  # Define one triangle using vertices 0, 1, 2
        positions=[
            (0.0, -0.5, 0),   # Bottom vertex
            (0.5, 0.5, 0),    # Right vertex
            (-0.5, 0.75, 0)   # Left vertex
        ],
        colors=[
            (1, 1, 0, 1),     # Yellow
            (1, 0, 1, 1),     # Magenta
            (0, 1, 1, 1),     # Cyan
        ],
    ),
    gfx.MeshBasicMaterial(color_mode="vertex"),  # Use vertex colors, interpolated across the triangle
)

# Move the triangle left along the X-axis to avoid overlapping with the square
triangle.local.position = (-0.6, 0, 0)


# --------------------------
# Define the square object
# --------------------------

# Create a square mesh by defining it with two triangles (quad)
square = gfx.Mesh(
    gfx.Geometry(
        indices=[(0, 1, 2), (2, 3, 0)],  # Two triangles: 0-1-2 and 2-3-0
        positions=[
            (-0.5, -0.5, 0),  # Bottom-left corner
            (0.5, -0.5, 0),   # Bottom-right corner
            (0.5, 0.5, 0),    # Top-right corner
            (-0.5, 0.5, 0),   # Top-left corner
        ],
        colors=[
            (1, 0, 0, 1),     # Red
            (0, 1, 0, 1),     # Green
            (0, 0, 1, 1),     # Blue
            (1, 1, 0, 1),     # Yellow
        ],
    ),
    gfx.MeshBasicMaterial(color_mode="vertex"),  # Use vertex colors, interpolated across the square
)

# Move the square right along the X-axis to avoid overlapping with the triangle
square.local.position = (0.6, 0, 0)


# --------------------------
# Set up and render the scene
# --------------------------

# Create a scene that can hold multiple objects
scene = gfx.Scene()

# Add both the triangle and square to the scene
scene.add(triangle, square)


# Main entry point of the script
if __name__ == "__main__":
    # Register a draw function that renders the scene using the camera
    canvas.request_draw(lambda: renderer.render(scene, camera))

    # Start the GUI event loop to display the window
    run()
