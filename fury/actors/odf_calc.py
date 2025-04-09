import os

import numpy as np

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
import scipy.special as sps
from fury.primitive import prim_sphere


def spherical_harmonics(m_values, l_values, theta, phi, use_scipy=True):
    """Compute spherical harmonics.

    This may take scalar or array arguments. The inputs will be broadcast
    against each other.

    Parameters
    ----------
    m_values : array of int ``|m| <= l``
        The phase factors (m) of the harmonics.
    l_values : array of int ``l >= 0``
        The orders (l) of the harmonics.
    theta : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    phi : float [0, pi]
        The polar (colatitudinal) coordinate.
    use_scipy : bool, optional
        If True, use scipy implementation.

    Returns
    -------
    y_mn : complex float
        The harmonic $Y^m_l$ sampled at ``theta`` and ``phi``.

    Notes
    -----
    This is a faster implementation of scipy.special.sph_harm for
    scipy version < 0.15.0. For scipy 0.15 and onwards, we use the scipy
    implementation of the function.

    The usual definitions for ``theta` and `phi`` used in DIPY are interchanged
    in the method definition to agree with the definitions in
    scipy.special.sph_harm, where `theta` represents the azimuthal coordinate
    and `phi` represents the polar coordinate.

    Although scipy uses a naming convention where ``m`` is the order and ``n``
    is the degree of the SH, the opposite of DIPY's, their definition for
    both parameters is the same as ours, with ``l >= 0`` and ``|m| <= l``.
    """
    if use_scipy:
        return sps.sph_harm(m_values, l_values, theta, phi, dtype=complex)

    x = np.cos(phi)
    val = sps.lpmv(m_values, l_values, x).astype(complex)
    val *= np.sqrt((2 * l_values + 1) / 4.0 / np.pi)
    val *= np.exp(0.5 * (sps.gammaln(l_values - m_values + 1) - sps.gammaln(l_values + m_values + 1)))
    val = val * np.exp(1j * m_values * theta)
    return val

def real_sh_descoteaux_from_index(m_values, l_values, theta, phi, legacy=True):
    """ Compute real spherical harmonics as in Descoteaux et al. 2007 [1]_,
    where the real harmonic $Y^m_l$ is defined to be:

        Imag($Y^m_l$) * sqrt(2)      if m > 0
        $Y^0_l$                      if m = 0
        Real($Y^m_l$) * sqrt(2)      if m < 0

    This may take scalar or array arguments. The inputs will be broadcast
    against each other.

    Parameters
    ----------
    m_values : array of int ``|m| <= l``
        The phase factors (m) of the harmonics.
    l_values : array of int ``l >= 0``
        The orders (l) of the harmonics.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    legacy: bool, optional
        If true, uses DIPY's legacy descoteaux07 implementation (where |m|
        is used for m < 0). Else, implements the basis as defined in
        Descoteaux et al. 2007 (without the absolute value).

    Returns
    -------
    real_sh : real float
        The real harmonic $Y^m_l$ sampled at ``theta`` and ``phi``.

    References
    ----------
     .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    """
    # In the cited paper, the basis is defined without the absolute value
    sh = spherical_harmonics(m_values, l_values, phi, theta)

    real_sh = np.where(m_values > 0, sh.imag, sh.real)
    real_sh *= np.where(m_values == 0, 1., np.sqrt(2))

    return real_sh

def sph_harm_ind_list(sh_order_max, full_basis=False):
    """
    Returns the order (``l``) and phase_factor (``m``) of all the symmetric
    spherical harmonics of order less then or equal to ``sh_order_max``.
    The results, ``m_list`` and ``l_list`` are kx1 arrays, where k depends on
    ``sh_order_max``.
    They can be passed to :func:`real_sh_descoteaux_from_index` and
    :func:``real_sh_tournier_from_index``.

    Parameters
    ----------
    sh_order_max : int
        The maximum order (l) of the spherical harmonic basis.
        Even int > 0, max order to return
    full_basis: bool, optional
        True for SH basis with even and odd order terms

    Returns
    -------
    m_list : array of int
        phase factors (m) of even spherical harmonics
    l_list : array of int
        orders (l) of even spherical harmonics

    See Also
    --------
    shm.real_sh_descoteaux_from_index, shm.real_sh_tournier_from_index

    """
    if full_basis:
        l_range = np.arange(0, sh_order_max + 1, dtype=int)
        ncoef = int((sh_order_max + 1) * (sh_order_max + 1))
    else:
        if sh_order_max % 2 != 0:
            raise ValueError('sh_order_max must be an even integer >= 0')
        l_range = np.arange(0, sh_order_max + 1, 2, dtype=int)
        ncoef = int((sh_order_max + 2) * (sh_order_max + 1) // 2)

    l_list = np.repeat(l_range, l_range * 2 + 1)
    offset = 0
    m_list = np.empty(ncoef, 'int')
    for ii in l_range:
        m_list[offset:offset + 2 * ii + 1] = np.arange(-ii, ii + 1)
        offset = offset + 2 * ii + 1

    # makes the arrays ncoef by 1, allows for easy broadcasting later in code
    return m_list, l_list

def real_sh_descoteaux(sh_order_max, theta, phi,
                       full_basis=False,
                       legacy=True):
    """ Compute real spherical harmonics as in Descoteaux et al. 2007 [1]_,
    where the real harmonic $Y^m_l$ is defined to be:

        Imag($Y^m_l$) * sqrt(2)      if m > 0
        $Y^0_l$                      if m = 0
        Real($Y^m_l$) * sqrt(2)      if m < 0

    This may take scalar or array arguments. The inputs will be broadcast
    against each other.

    Parameters
    ----------
    sh_order_max : int
        The maximum order (l) of the spherical harmonic basis.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    full_basis: bool, optional
        If true, returns a basis including odd order SH functions as well as
        even order SH functions. Otherwise returns only even order SH
        functions.
    legacy: bool, optional
        If true, uses DIPY's legacy descoteaux07 implementation (where |m|
        for m < 0). Else, implements the basis as defined in Descoteaux et al.
        2007.

    Returns
    -------
    real_sh : real float
        The real harmonic $Y^m_l$ sampled at ``theta`` and ``phi``.
    m_values : array of int
        The phase factor (m) of the harmonics.
    l_values : array of int
        The order (l) of the harmonics.

    References
    ----------
     .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    """
    m_value, l_value = sph_harm_ind_list(sh_order_max, full_basis)

    phi = np.reshape(phi, [-1, 1])
    theta = np.reshape(theta, [-1, 1])

    real_sh = real_sh_descoteaux_from_index(m_value, l_value, theta, phi,
                                            legacy)

    return real_sh, m_value, l_value

sph_harm_lookup = {None: real_sh_descoteaux,
                   #"tournier07": real_sh_tournier,
                   "descoteaux07": real_sh_descoteaux}

def sh_to_sf(sh, sphere, sh_order_max=4, basis_type=None,
             full_basis=False, legacy=True):
    """Spherical harmonics (SH) to spherical function (SF).

    Parameters
    ----------
    sh : ndarray
        SH coefficients representing a spherical function.
    sphere : Sphere
        The points on which to sample the spherical function.
    sh_order_max : int, optional
        Maximum SH order (l) in the SH fit.  For ``sh_order_max``, there will be
        ``(sh_order_max + 1) * (sh_order_max + 2) / 2`` SH coefficients for a
        symmetric basis and ``(sh_order_max + 1) * (sh_order_max + 1)``
        coefficients for a full SH basis.
    basis_type : {None, 'tournier07', 'descoteaux07'}, optional
        ``None`` for the default DIPY basis,
        ``tournier07`` for the Tournier 2007 [2]_[3]_ basis,
        ``descoteaux07`` for the Descoteaux 2007 [1]_ basis,
        (``None`` defaults to ``descoteaux07``).
    full_basis: bool, optional
        True to use a SH basis containing even and odd order SH functions.
        Else, use a SH basis consisting only of even order SH functions.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.

    Returns
    -------
    sf : ndarray
         Spherical function values on the ``sphere``.

    References
    ----------
    .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    .. [2] Tournier J.D., Calamante F. and Connelly A. Robust determination
           of the fibre orientation distribution in diffusion MRI:
           Non-negativity constrained super-resolved spherical deconvolution.
           NeuroImage. 2007;35(4):1459-1472.
    .. [3] Tournier J-D, Smith R, Raffelt D, Tabbara R, Dhollander T,
           Pietsch M, et al. MRtrix3: A fast, flexible and open software
           framework for medical image processing and visualisation.
           NeuroImage. 2019 Nov 15;202:116-137.
    """

    sph_harm_basis = sph_harm_lookup.get(basis_type)

    if sph_harm_basis is None:
        raise ValueError("Invalid basis name.")
    B, m_values, l_values = sph_harm_basis(sh_order_max, sphere.theta,
                                           sphere.phi,
                                           full_basis=full_basis,
                                           legacy=legacy)

    sf = np.dot(sh, B.T)

    return sf

class Sphere:
    theta = None
    phi = None

def compute_theta_phi(vertices):
    vertices = np.array(vertices)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    r = np.linalg.norm(vertices, axis=1)

    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = np.mod(phi, 2 * np.pi)  # normalize phi to [0, 2π]

    return theta, phi

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
    odf_actor.GetMapper().SetVBOShiftScaleMethod(False)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(odf_actor, big_centers, "center")

    big_scales = np.repeat(scales, 8, axis=0)
    attribute_to_actor(odf_actor, big_scales, "scale")

    minmax = np.array([coeffs.min(axis=1), coeffs.max(axis=1)]).T
    big_minmax = np.repeat(minmax, 8, axis=0)
    attribute_to_actor(odf_actor, big_minmax, "minmax")

    #sphere = get_sphere(sphere_type)

    vertices, _ = prim_sphere(name="repulsion100", gen_faces=True)
    theta, phi = compute_theta_phi(vertices)
    sphere = Sphere()
    sphere.theta = theta
    sphere.phi = phi

    sh_basis = "descoteaux07"
    n_coeffs = coeffs.shape[-1]
    sh_order = int((np.sqrt(8 * n_coeffs + 1) - 3) / 2)

    n_glyphs = coeffs.shape[0]

    tensor_sf = sh_to_sf(
        coeffs, sh_order_max=sh_order, basis_type=sh_basis, sphere=sphere, legacy=True
    )
    tensor_sf_max = abs(tensor_sf.reshape(n_glyphs, 100)).max(axis=1)

    sfmax = np.array(tensor_sf_max)
    big_sfmax = np.repeat(sfmax, 8, axis=0)
    attribute_to_actor(odf_actor, big_sfmax, "sfmax")

    odf_actor_pd = odf_actor.GetMapper().GetInput()

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

    def_coeff = f"#define NCOEFF {n_coeffs}"

    # TODO: Set int uniform
    odf_actor.GetShaderProperty().GetFragmentCustomUniforms().SetUniformf(
        "numCoeffs", n_coeffs
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

    sdf_map_1 = """
    vec3 map( in vec3 p )
    {
        p = p - centerMCVSOutput;
        vec3 p00 = p;

        float r, d; vec3 n, s, res;

        #define SHAPE (vec3(d-abs(r), sign(r),d))
        d=length(p00);
        n=p00 / d;
        float i = 1 / (numCoeffs * 2);
        float shCoeffs[NCOEFF];
        float maxCoeff = 0.0;
        for(int j=0; j < numCoeffs; j++){
            shCoeffs[j] = rescale(
                texture(
                    texture0,
                    vec2(i + j / numCoeffs, tcoordVCVSOutput.y)).x, 0, 1,
                    minmaxVSOutput.x, minmaxVSOutput.y
            );
        }
    """

    sh_list = """r = shCoeffs[0] * SH(0, 0, n);
        r += shCoeffs[1] * SH(2, -2, n);
        r += shCoeffs[2] * SH(2, -1, n);
        r += shCoeffs[3] * SH(2, 0, n);
        r += shCoeffs[4] * SH(2, 1, n);
        r += shCoeffs[5] * SH(2, 2, n);
        r += shCoeffs[6] * SH(4, -4, n);
        r += shCoeffs[7] * SH(4, -3, n);
        r += shCoeffs[8] * SH(4, -2, n);
        r += shCoeffs[9] * SH(4, -1, n);
        r += shCoeffs[10] * SH(4, 0, n);
        r += shCoeffs[11] * SH(4, 1, n);
        r += shCoeffs[12] * SH(4, 2, n);
        r += shCoeffs[13] * SH(4, 3, n);
        r += shCoeffs[14] * SH(4, 4, n);
        r += shCoeffs[15] * SH(6, -6, n);
        r += shCoeffs[16] * SH(6, -5, n);
        r += shCoeffs[17] * SH(6, -4, n);
        r += shCoeffs[18] * SH(6, -3, n);
        r += shCoeffs[19] * SH(6, -2, n);
        r += shCoeffs[20] * SH(6, -1, n);
        r += shCoeffs[21] * SH(6, 0, n);
        r += shCoeffs[22] * SH(6, 1, n);
        r += shCoeffs[23] * SH(6, 2, n);
        r += shCoeffs[24] * SH(6, 3, n);
        r += shCoeffs[25] * SH(6, 4, n);
        r += shCoeffs[26] * SH(6, 5, n);
        r += shCoeffs[27] * SH(6, 6, n);
        r += shCoeffs[28] * SH(8, -8, n);
        r += shCoeffs[29] * SH(8, -7, n);
        r += shCoeffs[30] * SH(8, -6, n);
        r += shCoeffs[31] * SH(8, -5, n);
        r += shCoeffs[32] * SH(8, -4, n);
        r += shCoeffs[33] * SH(8, -3, n);
        r += shCoeffs[34] * SH(8, -2, n);
        r += shCoeffs[35] * SH(8, -1, n);
        r += shCoeffs[36] * SH(8, 0, n);
        r += shCoeffs[37] * SH(8, 1, n);
        r += shCoeffs[38] * SH(8, 2, n);
        r += shCoeffs[39] * SH(8, 3, n);
        r += shCoeffs[40] * SH(8, 4, n);
        r += shCoeffs[41] * SH(8, 5, n);
        r += shCoeffs[42] * SH(8, 6, n);
        r += shCoeffs[43] * SH(8, 7, n);
        r += shCoeffs[44] * SH(8, 8, n);
    """
    sdf_map_2 = """
        r /= abs(sfmaxVSOutput);
        r *= scaleVSOutput * .9;

        s = SHAPE;
        res=s;
        return vec3(res.x, .5 + .5 * res.y, res.z);
    }
    """

    sdf_map = sdf_map_1 + "\n".join(sh_list.splitlines()[:n_coeffs]) + sdf_map_2
    #sdf_map = import_fury_shader(os.path.join("sdf", "sd_spherical_harmonics.#frag"))

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
        norm_const, spherical_harmonics, def_coeff, sdf_map, cast_ray,
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

