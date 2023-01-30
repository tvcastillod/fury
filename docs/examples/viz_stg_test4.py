"""
This spript includes the basic implementation of the Superquadric Tensor Glyph
using SDFs.
"""

import numpy as np
import os

from fury import actor, window
from fury.shaders import (attribute_to_actor, compose_shader,
                          import_fury_shader, shader_to_actor)


if __name__ == '__main__':
    # SDF Superquadric Tensor Glyph setup

    # Rectangle version

    # Box version

    centers = np.array([[0, 0, 0], [5, -5, 5], [-7, 7, -7], [10, 10, 10],
                        [10.5, 11.5, 11.5], [12, -12, -12], [-17, 17, 17],
                        [-22, -22, 22]])
    colors = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1],
                       [1, 0, 0], [0, 1, 0], [0, 1, 1]])
    scales = [6, .4, 1.2, 1, .2, .7, 3, 2]

    # Asymmetric box version

    # box_sd_stg_actor = actor.box(box_centers, directions=max_vecs,
    #                             colors=colors, scales=max_vals)

    # Symmetric box version

    box_sd_stg_actor = actor.box(centers, colors=colors, scales=scales)
    box_sd_stg_actor.GetMapper().SetVBOShiftScaleMethod(False)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_centers, 'center')

    big_scales = np.repeat(scales, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_scales, 'scale')

    # Billboard version

    # TODO: Add billboard version

    vs_dec = \
        """
        in vec3 center;
        in float scale;

        out vec4 vertexMCVSOutput;
        out vec3 centerMCVSOutput;
        out float scaleVSOutput;
        """

    vs_impl = \
        """
        vertexMCVSOutput = vertexMC;
        centerMCVSOutput = center;
        scaleVSOutput = scale;
        """

    shader_to_actor(box_sd_stg_actor, 'vertex', decl_code=vs_dec,
                    impl_code=vs_impl)

    fs_vars_dec = \
        """
        in vec4 vertexMCVSOutput;
        in vec3 centerMCVSOutput;
        in float scaleVSOutput;

        uniform mat4 MCVCMatrix;
        """

    sd_sphere = import_fury_shader(os.path.join('sdf', 'sd_sphere.frag'))

    # TODO: Fix scaling < 1
    sdf_map = \
        """
        float map(in vec3 position)
        {
            vec3 pos = position - centerMCVSOutput;
            float scaleFac = scaleVSOutput / 2;
            pos /= scaleFac;
            return sdSphere(pos, 1);
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
        //fragOutput0 = vec4(pnt, opacity);
        //vec3 normVertexMC = pnt - centerMCVSOutput;
        //fragOutput0 = vec4(normVertexMC, opacity);
        //float scaleFac = scaleVSOutput / 2;
        //normVertexMC /= scaleFac;
        //fragOutput0 = vec4(normVertexMC, opacity);
                
        // Ray Origin
        // Camera position in world space
        vec3 ro = (-MCVCMatrix[3] * MCVCMatrix).xyz;
        
        // Ray Direction
        vec3 rd = normalize(pnt - ro);
        //fragOutput0 = vec4(rd, opacity);
        
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
    scene.background((1, 1, 1))

    scene.add(box_sd_stg_actor)

    scene.reset_camera()
    scene.reset_clipping_range()

    window.show(scene, reset_camera=False)
