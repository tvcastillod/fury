import os

import numpy as np

from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf
from fury import actor
from fury.lib import FloatArray, Texture
from fury.shaders import (
    attribute_to_actor,
    compose_shader,
    import_fury_shader,
    shader_to_actor,
)
from fury.texture.utils import uv_calculations
from fury.utils import minmax_norm, numpy_to_vtk_image_data, set_polydata_tcoords


def sh_odf_calc(centers, coeffs, sphere_type, scales, opacity):
    """
    Visualize one or many ODFs with different features.

    Parameters
    ----------
    centers : ndarray(N, 3)
        ODFs positions.
    coeffs : ndarray
        2D ODFs array in SH coefficients.
    scales : float or ndarray (N, )
        ODFs size.
    opacity : float
        Takes values from 0 (fully transparent) to 1 (opaque).

    Returns
    -------
    box_actor: Actor

    """
    odf_actor = actor.box(centers=centers, scales=1.0)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(odf_actor, big_centers, "center")

    big_scales = np.repeat(scales, 8, axis=0)
    attribute_to_actor(odf_actor, big_scales, "scale")

    minmax = np.array([coeffs.min(axis=1), coeffs.max(axis=1)]).T
    big_minmax = np.repeat(minmax, 8, axis=0)
    attribute_to_actor(odf_actor, big_minmax, "minmax")

    #sphere = get_sphere("symmetric724")
    sphere = get_sphere(sphere_type)
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
    tensor_sf_max = abs(tensor_sf.reshape(4, 100)).max(axis=1)
    print(tensor_sf_max)
    print(coeffs.max(axis=1))

    sfmax = np.array(tensor_sf_max)
    big_sfmax = np.repeat(sfmax, 8, axis=0)
    attribute_to_actor(odf_actor, big_sfmax, "sfmax")

    odf_actor_pd = odf_actor.GetMapper().GetInput()

    n_glyphs = coeffs.shape[0]
    uv_vals = np.array(uv_calculations(n_glyphs))
    num_pnts = uv_vals.shape[0]

    t_coords = FloatArray()
    t_coords.SetNumberOfComponents(2)
    t_coords.SetNumberOfTuples(num_pnts)
    [t_coords.SetTuple(i, uv_vals[i]) for i in range(num_pnts)]

    set_polydata_tcoords(odf_actor_pd, t_coords)

    arr = minmax_norm(coeffs)
    arr *= 255
    grid = numpy_to_vtk_image_data(arr.astype(np.uint8))

    texture = Texture()
    texture.SetInputDataObject(grid)
    texture.Update()

    odf_actor.GetProperty().SetTexture("texture0", texture)

    # TODO: Set int uniform
    odf_actor.GetShaderProperty().GetFragmentCustomUniforms().SetUniformf(
        "numCoeffs", 15
    )

    vs_dec = """
        in vec3 center;
        in float scale;
        in vec2 minmax;
        in float sfmax;

        out vec4 vertexMCVSOutput;
        out vec3 centerMCVSOutput;
        out float scaleVSOutput;
        out vec2 minmaxVSOutput;
        out float sfmaxVSOutput;
        out vec3 camPosMCVSOutput;
    """

    vs_impl = """
        vertexMCVSOutput = vertexMC;
        centerMCVSOutput = center;
        scaleVSOutput = scale;
        minmaxVSOutput = minmax;
        sfmaxVSOutput = sfmax;
        vec3 camPos = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
    """

    shader_to_actor(odf_actor, "vertex", decl_code=vs_dec, impl_code=vs_impl)

    fs_defs = "#define PI 3.1415926535898"

    fs_unifs = """
        uniform mat4 MCVCMatrix;
    """

    fs_vs_vars = """
        in vec4 vertexMCVSOutput;
        in vec3 centerMCVSOutput;
        in float scaleVSOutput;
        in vec2 minmaxVSOutput;
        in float sfmaxVSOutput;
    """

    coeffs_norm = import_fury_shader(os.path.join("utils", "minmax_norm.glsl"))

    # Functions needed to calculate the associated Legendre polynomial
    factorial = import_fury_shader(os.path.join("utils", "factorial.glsl"))

    # Adapted from https://patapom.com/blog/SHPortal/
    # "Evaluate an Associated Legendre Polynomial P(l,m,x) at x
    # For more, see “Numerical Methods in C: The Art of Scientific Computing”,
    # Cambridge University Press, 1992, pp 252-254"
    legendre_polys = import_fury_shader(os.path.join("utils", "legendre_polynomial.glsl"))

    norm_const = import_fury_shader(os.path.join("utils", "sh_normalization_factor.glsl"))

    spherical_harmonics = import_fury_shader(os.path.join("ray_tracing\odf", "sh_function.glsl"))

    sdf_map = import_fury_shader(os.path.join("sdf", "sd_spherical_harmonics.frag"))

    cast_ray = import_fury_shader(os.path.join("ray_tracing\odf", "sh_cast_ray.frag"))

    central_diffs_normals = """
    vec3 centralDiffsNormals( in vec3 pos )
    {
        const vec2 eps = vec2(0.0001,0.0);

        return normalize( vec3(
                                map(pos+eps.xyy).x - map(pos-eps.xyy).x,
                                map(pos+eps.yxy).x - map(pos-eps.yxy).x,
                                map(pos+eps.yyx).x - map(pos-eps.yyx).x ) );
    }
    """

    # Applies the non-linearity that maps linear RGB to sRGB
    linear_to_srgb = import_fury_shader(
        os.path.join("lighting", "linear_to_srgb.frag")
    )

    # Inverse of linear_to_srgb()
    srgb_to_linear = import_fury_shader(
        os.path.join("lighting", "srgb_to_linear.frag")
    )

    # Turns a linear RGB color (i.e. rec. 709) into sRGB
    linear_rgb_to_srgb = import_fury_shader(
        os.path.join("lighting", "linear_rgb_to_srgb.frag")
    )

    # Inverse of linear_rgb_to_srgb()
    srgb_to_linear_rgb = import_fury_shader(
        os.path.join("lighting", "srgb_to_linear_rgb.frag")
    )

    # Logarithmic tonemapping operator. Input and output are linear RGB.
    tonemap = import_fury_shader(os.path.join("lighting", "tonemap.frag"))

    blinn_phong_model = import_fury_shader(
        os.path.join("lighting", "blinn_phong_model.frag")
    )

    # fmt: off
    fs_dec = compose_shader([
        fs_defs, fs_unifs, fs_vs_vars, coeffs_norm, factorial, legendre_polys,
        norm_const, spherical_harmonics, sdf_map, cast_ray,
        central_diffs_normals, linear_to_srgb, srgb_to_linear,
        linear_rgb_to_srgb, srgb_to_linear_rgb, tonemap, blinn_phong_model
    ])
    # fmt: on

    shader_to_actor(odf_actor, "fragment", decl_code=fs_dec, debug=False)

    sdf_frag_impl = """
        vec3 pnt = vertexMCVSOutput.xyz;

        vec3 ro = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);

        vec3 rd = normalize(pnt - ro);

        vec3 ld = normalize(ro - pnt);

        ro += pnt - ro;

        vec3 t = castRay(ro, rd);

        if(t.y > -.5)
        {
            vec3 pos = ro - centerMCVSOutput + t.x * rd;
            //vec3 normal = centralDiffsNormals(pos);
            vec3 colorDir = srgbToLinearRgb(abs(normalize(pos)));
            fragOutput0 = vec4(colorDir, opacity);
        }
        else
        {
            discard;
        }
    """

    shader_to_actor(
        odf_actor, "fragment", impl_code=sdf_frag_impl, block="picking"
    )

    return odf_actor

