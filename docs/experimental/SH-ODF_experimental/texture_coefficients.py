"""
This script includes TEXTURE experimentation for passing SH coefficients
"""
import os

import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf

from fury import actor, window
from fury.lib import FloatArray, Texture
from fury.shaders import (
    attribute_to_actor,
    compose_shader,
    import_fury_shader,
    shader_to_actor,
)
from fury.utils import numpy_to_vtk_image_data, set_polydata_tcoords

if __name__ == "__main__":
    show_man = window.ShowManager(size=(1920, 1080))
    show_man.scene.background((1, 1, 1))

    # fmt: off
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
        ]
    ])
    # fmt: on

    centers = np.array([[0, -1, 0], [1, -1, 0], [2, -1, 0]])
    scales = np.array([1, 2, 2])

    odf_actor = actor.box(centers=centers, scales=1.0)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(odf_actor, big_centers, "center")

    big_scales = np.repeat(scales, 8, axis=0)
    attribute_to_actor(odf_actor, big_scales, "scale")

    minmax = np.array([coeffs.min(axis=1), coeffs.max(axis=1)]).T
    big_minmax = np.repeat(minmax, 8, axis=0)
    attribute_to_actor(odf_actor, big_minmax, "minmax")

    odf_actor_pd = odf_actor.GetMapper().GetInput()

    # fmt: off
    uv_vals = np.array(
        [
            [0, 2 / 3], [0, 1], [1, 1], [1, 2 / 3],
            [0, 2 / 3], [0, 1], [1, 1], [1, 2 / 3],  # glyph1
            [0, 1 / 3], [0, 2 / 3], [1, 2 / 3], [1, 1 / 3],
            [0, 1 / 3], [0, 2 / 3], [1, 2 / 3], [1, 1 / 3],  # glyph2
            [0, 0], [0, 1 / 3], [1, 1 / 3], [1, 0],
            [0, 0], [0, 1 / 3], [1, 1 / 3], [1, 0]  # glyph3
        ]
    ) + [
            [0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1],
            [0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1], # glyph1
            [0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1],
            [0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1], # glyph2
            [0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1],
            [0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1] # glyph3
        ]
    # fmt: on

    num_pnts = uv_vals.shape[0]

    t_coords = FloatArray()
    t_coords.SetNumberOfComponents(2)
    t_coords.SetNumberOfTuples(num_pnts)
    [t_coords.SetTuple(i, uv_vals[i]) for i in range(num_pnts)]

    set_polydata_tcoords(odf_actor_pd, t_coords)

    min = coeffs.min(axis=1)
    max = coeffs.max(axis=1)
    newmin = 0
    newmax = 1
    arr = np.array(
        [
            (coeffs[i] - min[i]) * ((newmax - newmin) / (max[i] - min[i]))
            + newmin
            for i in range(coeffs.shape[0])
        ]
    )
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

    out vec4 vertexMCVSOutput;
    out vec3 centerMCVSOutput;
    out float scaleVSOutput;
    out vec2 minmaxVSOutput;
    """

    vs_impl = """
    vertexMCVSOutput = vertexMC;
    centerMCVSOutput = center;
    scaleVSOutput = scale;
    minmaxVSOutput = minmax;
    vec3 camPos = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
    """

    shader_to_actor(odf_actor, "vertex", decl_code=vs_dec, impl_code=vs_impl)

    fs_defs = "#define PI 3.1415926535898"

    fs_unifs = """
    uniform mat4 MCVCMatrix;
    uniform samplerCube texture_0;
    //uniform int k;
    """

    fs_vs_vars = """
    in vec4 vertexMCVSOutput;
    in vec3 centerMCVSOutput;
    in float scaleVSOutput;
    in vec2 minmaxVSOutput;
    """

    coeffs_norm = """
    float coeffsNorm(float coef)
    {
        float min = 0;
        float max = 1;
        float newMin = minmaxVSOutput.x;
        float newMax = minmaxVSOutput.y;
        return (coef - min) * ((newMax - newMin) / (max - min)) + newMin;
    }
    """

    # Functions needed to calculate the associated Legendre polynomial
    factorial = """
    int factorial(int v)
    {
        int t = 1;
        for(int i = 2; i <= v; i++)
        {
            t *= i;
        }
        return t;
    }
    """

    # Adapted from https://patapom.com/blog/SHPortal/
    # "Evaluate an Associated Legendre Polynomial P(l,m,x) at x
    # For more, see “Numerical Methods in C: The Art of Scientific Computing”,
    # Cambridge University Press, 1992, pp 252-254"
    legendre_polys = """
    float P(int l, int m, float x )
    {
        float pmm = 1;

        float somx2 = sqrt((1 - x) * (1 + x));
        float fact = 1;
        for (int i=1; i<=m; i++) {
            pmm *= -fact * somx2;
            fact += 2;
        }

        if( l == m )
            return pmm;

        float pmmp1 = x * (2 * m + 1) * pmm;
        if(l == m + 1)
            return pmmp1;

        float pll = 0;
        for (float ll=m + 2; ll<=l; ll+=1) {
            pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
            pmm = pmmp1;
            pmmp1 = pll;
        }

        return pll;
    }
    """

    norm_const = """
    float K(int l, int m)
    {
        float n = (2 * l + 1) * factorial(l - m);
        float d = 4 * PI * factorial(l + m);
        return sqrt(n / d);
    }
    """

    spherical_harmonics = """
    float SH(int l, int m, in vec3 s)
    {
        vec3 ns = normalize(s);
        float thetax = ns.y;
        float phi = atan(ns.z, ns.x) + PI / 2;
        float v = K(l, abs(m)) * P(l, abs(m), thetax);
        if(m != 0)
            v *= sqrt(2);
        if(m > 0)
            v *= sin(m * phi);
        if(m < 0)
            v *= cos(-m * phi);

        return v;
    }
    """

    sdf_map = """
    vec3 map( in vec3 p )
    {
        p = p - centerMCVSOutput;
        vec3 p00 = p;

        float r, d; vec3 n, s, res;

        #define SHAPE (vec3(d-abs(r), sign(r),d))
        //#define SHAPE (vec3(d-0.35, -1.0+2.0*clamp(0.5 + 16.0*r,0.0,1.0),d))
        d=length(p00);
        n=p00 / d;
        // ================================================================
        float i = 1 / (numCoeffs * 2);

        float c = texture(texture0, vec2(i, tcoordVCVSOutput.y)).x;
        r = coeffsNorm(c) * SH(0, 0, n);

        c = texture(texture0, vec2(i + 1 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(2, -2, n);

        c = texture(texture0, vec2(i + 2 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(2, -1, n);

        c = texture(texture0, vec2(i + 3 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(2, 0, n);

        c = texture(texture0, vec2(i + 4 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(2, 1, n);

        c = texture(texture0, vec2(i + 5 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(2, 2, n);

        c = texture(texture0, vec2(i + 6 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, -4, n);

        c = texture(texture0, vec2(i + 7 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, -3, n);

        c = texture(texture0, vec2(i + 8 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, -2, n);

        c = texture(texture0, vec2(i + 9 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, -1, n);

        c = texture(texture0, vec2(i + 10 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, 0, n);

        c = texture(texture0, vec2(i + 11 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, 1, n);

        c = texture(texture0, vec2(i + 12 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, 2, n);

        c = texture(texture0, vec2(i + 13 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, 3, n);

        c = texture(texture0, vec2(i + 14 / numCoeffs, tcoordVCVSOutput.y)).x;
        r += coeffsNorm(c) * SH(4, 4, n);

        r *= scaleVSOutput;
        // ================================================================
        s = SHAPE;
        res = s;
        return vec3(res.x, .5 + .5 * res.y, res.z);
    }
    """

    central_diffs_normals = """
    vec3 centralDiffsNormals(in vec3 pos)
    {
        //vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
        vec2 e = vec2(.001, -1);
        return normalize(
            e.xyy * map(pos + e.xyy).x + e.yyx * map(pos + e.yyx).x +
            e.yxy * map(pos + e.yxy).x + e.xxx * map(pos + e.xxx).x );
    }
    """

    """
    central_diffs_normals = import_fury_shader(
        os.path.join("sdf", "central_diffs.frag")
    )
    """

    cast_ray = """
    vec3 castRay(in vec3 ro, vec3 rd)
    {
        vec3 res = vec3(1e10, -1, 1);

        float maxd = 4;
        float h = 1;
        float t = 0;
        vec2  m = vec2(-1);

        for(int i = 0; i < 2000; i++)
        {
            if(h < .01 || t > maxd)
                break;
            vec3 res = map(ro + rd * t);
            h = res.x;
            m = res.yz;
            t += h * .1;
        }

        if(t < maxd && t < res.x)
            res = vec3(t, m);

        return res;
    }
    """

    blinn_phong_model = import_fury_shader(
        os.path.join("lighting", "blinn_phong_model.frag")
    )

    # fmt: off
    fs_dec = compose_shader([
        fs_defs, fs_unifs, fs_vs_vars, coeffs_norm, factorial, legendre_polys,
        norm_const, spherical_harmonics, sdf_map, central_diffs_normals,
        cast_ray, blinn_phong_model
    ])
    # fmt: on

    shader_to_actor(odf_actor, "fragment", decl_code=fs_dec, debug=False)

    sdf_frag_impl = """
    vec3 pnt = vertexMCVSOutput.xyz;

    vec3 ro = (-MCVCMatrix[3] * MCVCMatrix).xyz;

    vec3 rd = normalize(pnt - ro);

    vec3 ld = normalize(ro - pnt);

    ro += pnt - ro;

    vec3 t = castRay(ro, rd);

    if(t.y > -.5)
    {
        vec3 pos = ro + t.y * rd;

        vec3 normal = centralDiffsNormals(pos);

        float occ = clamp(2 * t.z, 0, 1);
        //float sss = pow(clamp(1 + dot(normal, rd), 0, 1), 1);
        float sss = clamp(1 + dot(normal, rd), 0, 1);

        vec3 lin  = 2.5 * occ * vec3(1) * (.6 + .4 * normal.y);
        lin += 1 * sss * vec3(1, .95, .7) * occ;

        vec3 mater = .5 * mix(vec3(1, 1, 0), vec3(1), t.y);

        fragOutput0 = vec4(vec3(1, 0, 0) * lin, opacity);
    }
    else
    {
        discard;
    }
    """

    shader_to_actor(
        odf_actor, "fragment", impl_code=sdf_frag_impl, block="picking"
    )

    show_man.scene.add(odf_actor)

    sphere = get_sphere("repulsion724")

    sh_basis = "descoteaux07"
    sh_basis = "tournier07"
    sh_order = 4

    sh = np.zeros((3, 1, 1, 15))
    sh[0, 0, 0, :] = coeffs[0, :]
    sh[1, 0, 0, :] = coeffs[1, :]
    sh[2, 0, 0, :] = coeffs[2, :]

    tensor_sf = sh_to_sf(
        sh, sh_order=sh_order, basis_type=sh_basis, sphere=sphere, legacy=False
    )

    odf_slicer_actor = actor.odf_slicer(
        tensor_sf, sphere=sphere, scale=0.5, colormap="plasma"
    )

    show_man.scene.add(odf_slicer_actor)

    show_man.start()