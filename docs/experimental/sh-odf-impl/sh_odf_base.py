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
from fury.texture.utils import uv_calculations
from fury.utils import (
    minmax_norm,
    numpy_to_vtk_image_data,
    set_polydata_tcoords,
)

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
    scales = np.ones(4) * 0.5

    odf_actor = actor.box(centers=centers, scales=1)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(odf_actor, big_centers, "center")

    big_scales = np.repeat(scales, 8, axis=0)
    attribute_to_actor(odf_actor, big_scales, "scale")

    minmax = np.array([coeffs.min(axis=1), coeffs.max(axis=1)]).T
    big_minmax = np.repeat(minmax, 8, axis=0)
    attribute_to_actor(odf_actor, big_minmax, "minmax")

    sphere = get_sphere(name="repulsion100")
    num_verts = sphere.vertices.shape[0]

    num_glyphs = coeffs.shape[0]

    max_num_coeffs = coeffs.shape[-1]
    max_sh_degree = int((np.sqrt(8 * max_num_coeffs + 1) - 3) / 2)
    max_poly_degree = 2 * max_sh_degree + 2
    viz_sh_degree = max_sh_degree

    # TODO: Find a way to avoid reshaping the SH coefficients
    sh = np.zeros((max_sh_degree, 1, 1, max_num_coeffs))
    sh[0, 0, 0, :] = coeffs[0, :]
    sh[1, 0, 0, :] = coeffs[1, :]
    sh[2, 0, 0, :] = coeffs[2, :]
    sh[3, 0, 0, :] = coeffs[3, :]

    sh_basis = "descoteaux07"

    fODFs = sh_to_sf(
        sh, sh_order_max=max_sh_degree, basis_type=sh_basis, sphere=sphere
    )

    max_fODFs = abs(fODFs.reshape(num_glyphs, num_verts)).max(axis=1)
    big_max_fodfs = np.repeat(max_fODFs, 8, axis=0)
    attribute_to_actor(odf_actor, big_max_fodfs, "maxfODFs")

    odf_actor_pd = odf_actor.GetMapper().GetInput()

    uv_vals = np.array(uv_calculations(num_glyphs))
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

    odf_actor.GetShaderProperty().GetFragmentCustomUniforms().SetUniformf(
        "shDegree", viz_sh_degree
    )

    vs_dec = """
    uniform float shDegree;

    in vec3 center;
    in vec2 minmax;
    in float maxfODFs;
    in float scale;

    flat out float numCoeffsVSOutput;

    out vec4 vertexMCVSOutput;
    out vec3 camPosMCVSOutput;
    out vec3 centerMCVSOutput;
    out vec2 minmaxVSOutput;
    out float maxfODFsVSOutput;
    out float scaleVSOutput;
    """

    vs_impl = """
    camPosMCVSOutput = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
    centerMCVSOutput = center;
    maxfODFsVSOutput = maxfODFs;
    minmaxVSOutput = minmax;
    numCoeffsVSOutput = (shDegree + 1) * (shDegree + 2) / 2;
    scaleVSOutput = scale;
    vertexMCVSOutput = vertexMC;
    """

    shader_to_actor(odf_actor, "vertex", decl_code=vs_dec, impl_code=vs_impl)

    fs_defs = "#define PI 3.1415926535898"

    fs_unifs = """
    uniform mat4 MCVCMatrix;
    uniform float psiMin = 99999.0;
    uniform float psiMax = -99999.0;
    """

    fs_vs_vars = """
    flat in float numCoeffsVSOutput;

    in vec4 vertexMCVSOutput;
    in vec3 centerMCVSOutput;
    in vec2 minmaxVSOutput;
    in float maxfODFsVSOutput;
    in float scaleVSOutput;
    """

    coeffs_norm = import_fury_shader(os.path.join("utils", "minmax_norm.glsl"))

    # Functions needed to calculate the associated Legendre polynomial
    # TODO: Precompute factorial values
    factorial = import_fury_shader(os.path.join("utils", "factorial.glsl"))

    # Adapted from https://patapom.com/blog/SHPortal/
    # "Evaluate an Associated Legendre Polynomial P(l,m,x) at x
    # For more, see “Numerical Methods in C: The Art of Scientific Computing”,
    # Cambridge University Press, 1992, pp 252-254"
    assoc_legendre_poly = import_fury_shader(
        os.path.join("sdf", "odf", "assoc_legendre_poly.frag")
    )

    norm_fact = import_fury_shader(
        os.path.join("sdf", "odf", "sh_norm_factor.frag")
    )

    spherical_harmonics = import_fury_shader(
        os.path.join("sdf", "odf", "sh_function.frag")
    )

    sdf_map = """
    /*
    vec3 map_max( in vec3 p )
    {
        p = p - centerMCVSOutput;
        vec3 p00 = p;

        float r, d; vec3 n, s, res;

        // TODO: Move out of the function
        #define SHAPE (vec3(d-abs(r), sign(r),d))
        //#define SHAPE (vec3(d-0.35, -1.0+2.0*clamp(0.5 + 16.0*r,0.0,1.0),d))

        d=length(p00);
        n=p00 / d;

        float i = 1 / (numCoeffsVSOutput * 2);
        float shCoeffs[15];
        float maxCoeff = 0.0;
        for(int j=0; j < numCoeffsVSOutput; j++){
            shCoeffs[j] = rescale(
                texture(
                    texture0,
                    vec2(i + j / numCoeffsVSOutput, tcoordVCVSOutput.y)).x,
                    0, 1, minmaxVSOutput.x, minmaxVSOutput.y
            );// /abs(minmaxVSOutput.y);
        }
        r = shCoeffs[0] * calculateSH(0, 0, n);
        r += shCoeffs[1] * calculateSH(2, -2, n);
        r += shCoeffs[2] * calculateSH(2, -1, n);
        r += shCoeffs[3] * calculateSH(2, 0, n);
        r += shCoeffs[4] * calculateSH(2, 1, n);
        r += shCoeffs[5] * calculateSH(2, 2, n);
        r += shCoeffs[6] * calculateSH(4, -4, n);
        r += shCoeffs[7] * calculateSH(4, -3, n);
        r += shCoeffs[8] * calculateSH(4, -2, n);
        r += shCoeffs[9] * calculateSH(4, -1, n);
        r += shCoeffs[10] * calculateSH(4, 0, n);
        r += shCoeffs[11] * calculateSH(4, 1, n);
        r += shCoeffs[12] * calculateSH(4, 2, n);
        r += shCoeffs[13] * calculateSH(4, 3, n);
        r += shCoeffs[14] * calculateSH(4, 4, n);

        //r *= scaleVSOutput * .9;
        // ================================================================
        s = SHAPE;
        res=s;
        return vec3(res.x, .5 + .5 * res.y, res.z);
    }
    */

    vec3 map( in vec3 p )
    {
        p = p - centerMCVSOutput;
        vec3 p00 = p;

        float r, d; vec3 n, s, res;

        // TODO: Move out of the function
        #define SHAPE (vec3(d-abs(r), sign(r),d))
        //#define SHAPE (vec3(d-0.35, -1.0+2.0*clamp(0.5 + 16.0*r,0.0,1.0),d))

        d=length(p00);
        n=p00 / d;

        float i = 1 / (numCoeffsVSOutput * 2);
        float shCoeffs[15];
        float maxCoeff = 0.0;
        for(int j=0; j < numCoeffsVSOutput; j++){
            shCoeffs[j] = rescale(
                texture(
                    texture0,
                    vec2(i + j / numCoeffsVSOutput, tcoordVCVSOutput.y)).x,
                    0, 1, minmaxVSOutput.x, minmaxVSOutput.y
            );// /abs(minmaxVSOutput.y);
        }
        r = shCoeffs[0] * calculateSH(0, 0, n);
        r += shCoeffs[1]* calculateSH(2, -2, n);
        r += shCoeffs[2]* calculateSH(2, -1, n);
        r += shCoeffs[3] * calculateSH(2, 0, n);
        r += shCoeffs[4] * calculateSH(2, 1, n);
        r += shCoeffs[5] * calculateSH(2, 2, n);
        r += shCoeffs[6] * calculateSH(4, -4, n);
        r += shCoeffs[7] * calculateSH(4, -3, n);
        r += shCoeffs[8]* calculateSH(4, -2, n);
        r += shCoeffs[9]* calculateSH(4, -1, n);
        r += shCoeffs[10]* calculateSH(4, 0, n);
        r += shCoeffs[11]* calculateSH(4, 1, n);
        r += shCoeffs[12]* calculateSH(4, 2, n);
        r += shCoeffs[13]* calculateSH(4, 3, n);
        r += shCoeffs[14]* calculateSH(4, 4, n);

        /*
        // OPTION 2
        float psiMin = 0.0;
        float psiMax = 0.0;
        for (int j = 0; j < numCoeffsVSOutput; j++) {
            float absCoeff = abs(shCoeffs[j]); // Take absolute value of each coefficient
            psiMax += absCoeff; // Upper bound estimate
            psiMin -= absCoeff; // Lower bound estimate
        }
        r /= (psiMax-psiMin);
        */

        /*
        // OPTION 1
        float maxCoeff = 0.0;
        float minCoeff = 0.0;
        for (int i = 0; i < numCoeffsVSOutput; i++) {
            maxCoeff += abs(shCoeffs[i]);
        }
        if (maxCoeff > 0.0) {
            r /= maxCoeff;
        }
        */

        //r /= abs(minmaxVSOutput.y);
        //r /= abs(maxfODFsVSOutput);
        r /= maxfODFsVSOutput;
        //r /=  abs(MAX_SF);
        r *= scaleVSOutput * .9;
        // ================================================================
        s = SHAPE;
        res=s;
        return vec3(res.x, .5 + .5 * res.y, res.z);
    }
    """

    cast_ray = """
    vec3 castRay(in vec3 ro, vec3 rd)
    {
        vec3 res = vec3(1e10, -1, 1);

        float maxd = 1;
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

    central_diffs_normals = """
    /*
    vec3 centralDiffsNormals(in vec3 p, float eps)
    {
        vec2 h = vec2(eps, 0);
        return normalize(vec3(mapp(p + h.xyy) - mapp(p - h.xyy),
                            mapp(p + h.yxy) - mapp(p - h.yxy),
                            mapp(p + h.yyx) - mapp(p - h.yyx)));
    }
    */
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
        fs_defs, fs_unifs, fs_vs_vars, coeffs_norm, factorial, assoc_legendre_poly,
        norm_fact, spherical_harmonics, sdf_map, cast_ray,
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

    vec3 color = vec3(1.);

    if(t.y > -.5)
    {
        vec3 pos = ro - centerMCVSOutput + t.x * rd;
        vec3 normal = centralDiffsNormals(pos);

        /*
        float occ = clamp(2 * t.z, 0, 1);
        //float sss = pow(clamp(1 + dot(normal, rd), 0, 1), 1);
        float sss = clamp(1 + dot(normal, rd), 0, 1);

        vec3 lin  = 2.5 * occ * vec3(1) * (.6 + .4 * normal.y);
        lin += 1 * sss * vec3(1, .95, .7) * occ;

        vec3 mater = .5 * mix(vec3(1, 1, 0), vec3(1), t.y);

        fragOutput0 = vec4(vec3(1, 0, 0) * lin, opacity);
        */
        vec3 colorDir = srgbToLinearRgb(abs(normalize(pos)));
        float attenuation = dot(ld, normal);
        color = blinnPhongIllumModel(
            //attenuation, lightColor0, diffuseColor, specularPower,
            attenuation, lightColor0, colorDir, specularPower,
            specularColor, ambientColor);
    }
    else
    {
        discard;
    }

    //fragOutput0 = vec4(linearToSrgb(color * colorDir), opacity);
    vec3 outColor = linearRgbToSrgb(tonemap(color));
    fragOutput0 = vec4(outColor, opacity);
    """

    shader_to_actor(
        odf_actor, "fragment", impl_code=sdf_frag_impl, block="picking"
    )

    show_man.scene.add(odf_actor)

    sphere = get_sphere(name="repulsion724")

    sh_basis = "descoteaux07"
    # sh_basis = "tournier07"
    sh_order = 4

    sh = np.zeros((4, 1, 1, 15))
    sh[0, 0, 0, :] = coeffs[0, :]
    sh[1, 0, 0, :] = coeffs[1, :]
    sh[2, 0, 0, :] = coeffs[2, :]
    sh[3, 0, 0, :] = coeffs[3, :]

    fODFs = sh_to_sf(
        sh,
        sh_order_max=sh_order,
        basis_type=sh_basis,
        sphere=sphere,
        legacy=True,
    )
    odf_slicer_actor = actor.odf_slicer(fODFs, sphere=sphere, norm=True)

    show_man.scene.add(odf_slicer_actor)

    show_man.start()
