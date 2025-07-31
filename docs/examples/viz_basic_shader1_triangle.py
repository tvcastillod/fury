# Import the required modules
import wgpu  # Low-level GPU access (WebGPU)
from wgpu.gui.auto import WgpuCanvas, run  # GUI canvas and event loop integration
import pygfx as gfx  # High-level rendering abstraction built on WGPU
from pygfx.renderers.wgpu import (
    Binding,              # Used to define resource bindings for shaders
    BaseShader,           # Base class to implement custom WGPU shaders
    RenderMask,           # Enum to categorize rendering passes (e.g., opaque, transparent)
    register_wgpu_render_function,  # Decorator to link objects and materials to shaders
)

#### Custom object, material, and matching render function

# -----------------------------------------------------------------------------
# Custom world object, material, and rendering logic using WGPU via pygfx
# -----------------------------------------------------------------------------

# Define a new type of world object – a logical representation in the scene graph
class Triangle(gfx.WorldObject):
    pass  # No additional properties needed for this minimal example

# Define a custom material – holds shader-specific properties (empty for now)
class TriangleMaterial(gfx.Material):
    pass  # Used only to differentiate this shader pipeline

# Register the render function so pygfx knows how to render this object/material pair
@register_wgpu_render_function(Triangle, TriangleMaterial)
class TriangleShader(BaseShader):
    # Mark as render-shader (as opposed to compute-shader)
    type = "render" ### must be "render" or "compute"

    def get_bindings(self, wobject, shared):
        # Define what GPU resources (buffers/textures) the shader will use
        ## Our only binding is a uniform buffer
        bindings = {
            0: Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
        }
        '''
        * name: the name in wgsl
        * type: "buffer/subtype", "sampler/subtype", "texture/subtype", "storage_texture/subtype".
            The subtype depends on the type: BufferBindingType, SamplerBindingType, TextureSampleType, or StorageTextureAccess.
        * resource: the Buffer, GfxTextureView, or GfxSampler object.
        '''
        ### Generate the WGSL code for these bindings
        self.define_bindings(0, bindings)
        ### The "bindings" are grouped as a dict of dicts. Often only
        ### bind-group 0 is used.
        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        ### Result. All fields are mandatory.
        ## We draw triangles, no culling
        # Specify pipeline-level GPU configuration:
        # - Drawing triangles
        # - No face culling (we draw one triangle, so no need to skip any side)
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }
        '''
        Options for primitive topology:
            point_list = "point-list"
            line_list = "line-list"
            line_strip = "line-strip"
            triangle_list = "triangle-list"
            triangle_strip = "triangle-strip"
        '''

    def get_render_info(self, wobject, shared):
        ## Since we draw only one triangle we need just 3 vertices.
        ## Our triangle is opaque (render mask 1).

        ### Result. All fields are mandatory. The RenderMask.all is a safe
        ### value; other values are optimizations.
        # Render 1 triangle (3 vertices), with opaque rendering mask
        # - "indices": (3, 1) = 3 vertices, 1 instance
        return {
            "indices": (3, 1),
            "render_mask": RenderMask.all,
        }

    def get_code(self):
        ## Here we put together the full (templated) shader code
        # Return WGSL (WebGPU Shading Language) source for vertex and fragment stages
        return """
        {$ include 'pygfx.std.wgsl' $}  // Include standard definitions (uniforms, outputs)

        // Vertex shader: defines positions of the triangle's 3 vertices
        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
            // Define positions in screen space (logical pixels)
            var positions = array<vec2<f32>, 3>(
                vec2<f32>(10.0, 10.0), vec2<f32>(90.0, 10.0), vec2<f32>(10.0, 90.0)
            );
            // Convert screen space to normalized device coordinates (-1 to +1)
            let p = 2.0 * positions[index] / u_stdinfo.logical_size - 1.0;
            return vec4<f32>(p, 0.0, 1.0);  // Return final vertex position
        }

        // Fragment shader: sets the color of the triangle
        @fragment
        fn fs_main() -> FragmentOutput {
            var out: FragmentOutput;
            out.color = vec4<f32>(1.0, 0.7, 0.2, 1.0);  // Orange color
            return out;
        }
        """


## Setup scene
# -----------------------------------------------------------------------------
# Setup rendering environment
# -----------------------------------------------------------------------------

renderer = gfx.WgpuRenderer(WgpuCanvas()) # Create a canvas for rendering


# Use a simple NDC (Normalized Device Coordinates) camera – no projection/transformations
camera = gfx.NDCCamera()  # Camera is not actually used by the shader, but required by pygfx

# Create the Triangle object with its associated material
t = Triangle(None, TriangleMaterial())

# Create a scene and add the triangle object to it
scene = gfx.Scene()
scene.add(t)

# -----------------------------------------------------------------------------
# Run the rendering loop
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Request the renderer to draw the scene each frame
    renderer.request_draw(lambda: renderer.render(scene, camera))
    run()  # Start the GUI event loop (platform-specific)
