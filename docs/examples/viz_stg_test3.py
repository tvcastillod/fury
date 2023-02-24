"""
This spript includes the basic implementation of the Superquadric Tensor Glyph
using SDFs.
"""

import numpy as np
import os

from fury import actor, window
from fury.actor import _color_fa, _fa
from fury.primitive import prim_sphere
from fury.shaders import (attribute_to_actor, compose_shader,
                          import_fury_shader, shader_to_actor)


class Sphere:
    vertices = None
    faces = None


def generate_tensors():
    # Inspired by test_peak.py generate_peaks
    vecs10 = np.array([[-.6, .5, -.6], [-.8, -.4, .5], [-.1, -.7, -.7]])
    vecs01 = np.array([[.1, .6, -.8], [.6, .5, .5], [-.8, .6, .3]])
    vecs11 = np.array([[.7, .5, -.5], [0, -.7, -.7], [-.7, .6, -.5]])
    vecs21 = np.array([[.7, -.3, -.6], [.2, -.8, .6], [.7, .6, .5]])
    tensors_vecs = np.zeros((3, 2, 1, 3, 3))

    tensors_vecs[1, 0, 0, :, :] = vecs10
    tensors_vecs[0, 1, 0, :, :] = vecs01
    tensors_vecs[1, 1, 0, :, :] = vecs11
    tensors_vecs[2, 1, 0, :, :] = vecs21

    tensors_vals = np.zeros((3, 2, 1, 3))

    tensors_vals[1, 0, 0, :] = np.array([.2, .3, .3])
    tensors_vals[0, 1, 0, :] = np.array([.4, .7, .2])
    tensors_vals[1, 1, 0, :] = np.array([.8, .4, .3])
    tensors_vals[2, 1, 0, :] = np.array([.7, .5, .1])
    return tensors_vecs, tensors_vals


if __name__ == '__main__':
    # Default tensor setup
    vertices, faces = prim_sphere('repulsion724', True)

    sphere = Sphere()
    sphere.vertices = vertices
    sphere.faces = faces

    mevecs, mevals = generate_tensors()

    dt_affine = np.eye(4)
    # dt_affine[0, 3] = -1

    tensor_actor = actor.tensor_slicer(mevals, mevecs, affine=dt_affine,
                                       sphere=sphere, scale=.5)

    mevecs_shape = mevals.shape

    tensor_actor.display_extent(0, mevecs_shape[0], 0, mevecs_shape[1],
                                0, mevecs_shape[2])

    # Peak setup

    p_affine = np.eye(4)
    p_affine[1, 3] = 2

    peak_actor = actor.peak(mevecs, peaks_values=mevals, affine=p_affine)

    # FURY's glyphs standardization

    valid_mask = np.abs(mevecs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)

    centers = np.asarray(indices).T

    num_centers = centers.shape[0]

    # It seems that a more standard parameter for glyph information would be
    # Degrees of Freedom (DoF). According to this:
    # https://en.wikipedia.org/wiki/Gordon_Kindlmann

    dofs_vecs = mevecs[indices]
    dofs_vals = mevals[indices]

    colors = [_color_fa(_fa(dofs_vals[i, :]), dofs_vecs[i, ...])
              for i in range(num_centers)]
    colors = np.asarray(colors)

    # max_vals = dofs_vals.max(axis=-1)
    argmax_vals = dofs_vals.argmax(axis=-1)
    max_vals = dofs_vals[np.arange(len(dofs_vals)), argmax_vals]
    max_vecs = dofs_vecs[np.arange(len(dofs_vals)), argmax_vals, :]

    # SDF Superquadric Tensor Glyph setup

    # Rectangle version

    # Box version

    box_centers = centers
    box_centers[:, 1] += -2

    # Asymmetric box version

    #box_sd_stg_actor = actor.box(box_centers, directions=max_vecs,
    #                             colors=colors, scales=max_vals)

    # Symmetric box version

    box_sd_stg_actor = actor.box(box_centers, colors=colors, scales=max_vals)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_centers, 'center')

    big_scales = np.repeat(max_vals, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_scales, 'scale')

    evals = np.array([mevals[0][1][0], mevals[1][0][0],
                      mevals[1][1][0], mevals[2][1][0]])
    print("EIGEN VALUES ", evals)
    big_values = np.repeat(np.array(evals, dtype=float), 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_values, 'evals')

    eigenvecs = np.array([mevecs[0][1][0], mevecs[1][0][0],
                          mevecs[1][1][0], mevecs[2][1][0]])
    print("EIGEN VECTORS ", eigenvecs)

    evec1 = np.array(
        [mevecs[0][1][0][0], mevecs[1][0][0][0], mevecs[1][1][0][0],
         mevecs[2][1][0][0]])
    evec2 = np.array(
        [mevecs[0][1][0][1], mevecs[1][0][0][1], mevecs[1][1][0][1],
         mevecs[2][1][0][1]])
    evec3 = np.array(
        [mevecs[0][1][0][2], mevecs[1][0][0][2], mevecs[1][1][0][2],
         mevecs[2][1][0][2]])
    vector1 = actor.arrow(centers=box_centers, directions=evec1, colors=colors)
    big_vectors_1 = np.repeat(evec1, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_vectors_1, 'evec1')
    vector2 = actor.arrow(centers=box_centers, directions=evec2, colors=colors)
    big_vectors_2 = np.repeat(evec2, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_vectors_2, 'evec2')
    vector3 = actor.arrow(centers=box_centers, directions=evec3, colors=colors)
    big_vectors_3 = np.repeat(evec3, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_vectors_3, 'evec3')
    # Billboard version

    # TODO: Add billboard version

    vs_dec = \
        """
        in vec3 center;
        in float scale;
        in vec3 evals;
        in vec3 evec1;
        in vec3 evec2;
        in vec3 evec3;
    
        out vec4 vertexMCVSOutput;
        out vec3 centerMCVSOutput;
        out float scaleVSOutput;
        out vec3 evalsVSOutput;
        out mat3 tensorMatrix;
        """

    vs_impl = \
        """
        vertexMCVSOutput = vertexMC;
        centerMCVSOutput = center;
        scaleVSOutput = scale;
        evalsVSOutput = normalize(evals);
        mat3 T = mat3(1/evalsVSOutput.x, 0.0, 0.0,
                      0.0, 1/evalsVSOutput.y, 0.0,
                      0.0, 0.0, 1/evalsVSOutput.z);
        mat3 R = mat3(evec1, evec2, evec3);
        tensorMatrix = inverse(R) * T * R;
        """

    shader_to_actor(box_sd_stg_actor, 'vertex', decl_code=vs_dec,
                    impl_code=vs_impl)

    fs_vars_dec = \
        """
        in vec4 vertexMCVSOutput;
        in vec3 centerMCVSOutput;
        in float scaleVSOutput;
        in vec3 evalsVSOutput;
        in mat3 tensorMatrix;
    
        uniform mat4 MCVCMatrix;
        """

    sd_sphere = import_fury_shader(os.path.join('sdf', 'sd_sphere.frag'))

    # TODO: Fix scaling < 1
    sdf_map = \
        """
        float map(in vec3 position)
        {
            return sdSphere(tensorMatrix * (position - centerMCVSOutput), .5) 
                * min(evalsVSOutput.x, min(evalsVSOutput.y, evalsVSOutput.z));
        }
        """

    central_diffs_normal = import_fury_shader(os.path.join(
        'sdf', 'central_diffs.frag'))

    cast_ray = import_fury_shader(os.path.join(
        'ray_marching', 'cast_ray.frag'))

    blinn_phong_model = import_fury_shader(os.path.join(
        'lighting', 'blinn_phong_model.frag'))

    fs_dec = compose_shader([fs_vars_dec, sd_sphere, sdf_map,
                             central_diffs_normal, cast_ray,
                             blinn_phong_model])

    shader_to_actor(box_sd_stg_actor, 'fragment', decl_code=fs_dec)

    sdf_frag_impl = \
        """
        vec3 pnt = vertexMCVSOutput.xyz;
    
        // Ray Origin
        // Camera position in world space
        vec3 ro = (-MCVCMatrix[3] * MCVCMatrix).xyz;
    
        // Ray Direction
        vec3 rd = normalize(pnt - ro);
    
        // Light Direction
        vec3 ld = normalize(ro - pnt);
    
        ro += pnt - ro;
    
        float t = castRay(ro, rd);
    
        if(t < 20)
        {
            vec3 pos = ro + t * rd;
            vec3 normal = centralDiffsNormals(pos, .0001);
            // Light Attenuation
            float la = dot(ld, normal);
            vec3 color = blinnPhongIllumModel(la, lightColor0, diffuseColor, 
                specularPower, specularColor, ambientColor);
            fragOutput0 = vec4(color, opacity);
        }
        else
        {
            discard;
        }
        """

    shader_to_actor(box_sd_stg_actor, 'fragment', impl_code=sdf_frag_impl,
                    block='light')
    # Scene setup
    scene = window.Scene()
    scene.background((255, 255, 255))

    scene.add(tensor_actor)
    scene.add(peak_actor)
    scene.add(box_sd_stg_actor)

    scene.reset_camera()
    scene.reset_clipping_range()

    window.show(scene, reset_camera=False)
