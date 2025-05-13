# Example taken from: https://docs.pygfx.org/stable/_gallery/feature_demo/custom_object3.html#sphx-glr-gallery-feature-demo-custom-object3-py
import numpy as np
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
from fury.window import ShowManager
import pygfx as gfx
from pygfx.renderers.wgpu import (
    Binding,
    BaseShader,
    RenderMask,
    register_wgpu_render_function,
)
from wgpu.gui.auto import WgpuCanvas, run
from pygfx.renderers.wgpu import WgpuRenderer
from pygfx.controllers import OrbitController

# Custom object, material, and matching render function


class Triangle(gfx.WorldObject):
    pass


class TriangleMaterial(gfx.Material):
    uniform_type = dict(
        gfx.Material.uniform_type,
        color="4xf4",
    )

    def __init__(self, *, color="white", **kwargs):
        super().__init__(**kwargs)
        self.color = color

    @property
    def color(self):
        """The uniform color of the triangle."""
        return gfx.Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, color):
        color = gfx.Color(color)
        self.uniform_buffer.data["color"] = color
        self.uniform_buffer.update_full()
        self._store.color_is_transparent = color.a < 1

    @property
    def color_is_transparent(self):
        """Whether the color is (semi) transparent (i.e. not fully opaque)."""
        # Note the use of the the _store to make this attribute trackable,
        # so that when it changes, the shader is updated automatically.
        return self._store.color_is_transparent


@register_wgpu_render_function(Triangle, TriangleMaterial)
class TriangleShader(BaseShader):
    type = "render"

    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry

        # This is how we set templating variables (dict-like access on the shader).
        # Look for "{{scale}}" in the WGSL code below.
        self["scale"] = 0.2

        # Three uniforms and one storage buffer with positions
        bindings = {
            0: Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            1: Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            2: Binding("u_material", "buffer/uniform", wobject.material.uniform_buffer),
            3: Binding(
                "s_positions", "buffer/read_only_storage", geometry.positions, "VERTEX"
            ),
        }
        self.define_bindings(0, bindings)
        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        # We draw triangles, no culling
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        material = wobject.material
        geometry = wobject.geometry

        # Determine number of vertices
        n = 3 * geometry.positions.nitems

        # Define in what passes this object is drawn.
        # Using RenderMask.all is a good default. The rest is optimization.
        render_mask = wobject.render_mask
        if not render_mask:  # i.e. set to auto
            render_mask = RenderMask.all
            if material.is_transparent:
                render_mask = RenderMask.transparent
            elif material.color_is_transparent:
                render_mask = RenderMask.transparent
            else:
                render_mask = RenderMask.opaque

        return {
            "indices": (n, 1),
            "render_mask": render_mask,
        }

    def get_code(self):
        return """
        {$ include 'pygfx.std.wgsl' $}

        @vertex
fn vs_main(@builtin(vertex_index) index: u32) -> Varyings {
    // Compute quad vertex index
    let vertex_index = i32(index) / 6;
    let sub_index = i32(index) % 6;

    // Center position of the billboard
    let center = load_s_positions(vertex_index); // vec3 in model coords

    // Define screen-aligned quad corners (2 triangles = 6 vertices)
    var quad_offsets = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), // Triangle 1
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), // Triangle 2
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );

    // Billboard size in world units
    let shape = vec2<f32>(0.5, 0.5); // example size

    // Get normalized offset
    let offset = quad_offsets[sub_index];

    // Extract view matrix from stdinfo
    let view = u_stdinfo.cam_transform;

    // Extract camera right (X) and up (Y) vectors from the view matrix
    let right = vec3<f32>(view[0].x, view[1].x, view[2].x);
    let up = vec3<f32>(view[0].y, view[1].y, view[2].y);

    // Billboard logic: offset from center in world coordinates
    let pos = center + offset.x * shape.x * right + offset.y * shape.y * up;

    // Project to clip space
    let world_pos = u_wobject.world_transform * vec4<f32>(pos, 1.0);
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    // Output varyings
    var varyings: Varyings;
    varyings.position = vec4<f32>(ndc_pos);
    varyings.color = vec4<f32>(u_material.color);

    varyings.norm = vec3<f32>(normalize(vec3<f32>(offset.x, offset.y, 0.0)));


    return varyings;
}


        @fragment
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            var out: FragmentOutput;
            let a = u_material.color.a * u_material.opacity;
            out.color = vec4<f32>(u_material.color.rgb, a);
            return out;
        }
        """


# Setup scene
camera = gfx.OrthographicCamera(10, 10)

t = Triangle(
    gfx.Geometry(positions=np.random.uniform(-4, 4, size=(100, 3)).astype(np.float32)),
    TriangleMaterial(color="yellow"),
)
t.local.x = 2  # set offset to demonstrate that it works

scene = gfx.Scene()
scene.add(t)


if __name__ == "__main__":
    show_m = ShowManager(scene=scene, camera=camera)
    show_m.start()

