import os

import numpy as np

from fury import actor
from fury.lib import Actor
from fury.shaders import (attribute_to_actor, import_fury_shader,
                          shader_to_actor, compose_shader)


class DoubleConeActor(Actor):
    """
    VTK actor for visualizing Double cones.

    """
    def __init__(self, centers, axes, lengths, angles, colors, scales, opacity):
        self.centers = centers
        self.axes = axes
        self.lengths = lengths
        self.angles = angles
        self.colors = colors
        self.scales = scales
        self.opacity = opacity
        self.SetMapper(actor.box(self.centers, colors=self.colors,
                                 scales=scales).GetMapper())
        self.GetMapper().SetVBOShiftScaleMethod(False)
        self.GetProperty().SetOpacity(self.opacity)

        big_centers = np.repeat(self.centers, 8, axis=0)
        attribute_to_actor(self, big_centers, 'center')

        big_scales = np.repeat(self.scales, 8, axis=0)
        attribute_to_actor(self, big_scales, 'scale')

        big_values = np.repeat(np.array(self.lengths, dtype=float), 8, axis=0)
        attribute_to_actor(self, big_values, 'evals')

        evec1 = np.array([item[0] for item in self.axes])
        evec2 = np.array([item[1] for item in self.axes])
        evec3 = np.array([item[2] for item in self.axes])

        big_vectors_1 = np.repeat(evec1, 8, axis=0)
        attribute_to_actor(self, big_vectors_1, 'evec1')
        big_vectors_2 = np.repeat(evec2, 8, axis=0)
        attribute_to_actor(self, big_vectors_2, 'evec2')
        big_vectors_3 = np.repeat(evec3, 8, axis=0)
        attribute_to_actor(self, big_vectors_3, 'evec3')

        big_angles = np.repeat(np.array(self.angles, dtype=float), 8, axis=0)
        attribute_to_actor(self, big_angles, 'angle')

        vs_dec = \
            """
            in vec3 center;
            in float scale;
            in vec3 evals;
            in vec3 evec1;
            in vec3 evec2;
            in vec3 evec3;
            in float angle;

            out vec4 vertexMCVSOutput;
            out vec3 centerMCVSOutput;
            out float scaleVSOutput;
            out vec3 evalsVSOutput;
            out mat3 rotationMatrix;
            out float angleVSOutput;
            """

        vs_impl = \
            """
            vertexMCVSOutput = vertexMC;
            centerMCVSOutput = center;
            scaleVSOutput = scale;
            mat3 R = mat3(normalize(evec1), normalize(evec2), normalize(evec3));
            float a = radians(90);
            mat3 rot = mat3(cos(a),-sin(a),0,
                            sin(a),cos(a), 0, 
                            0,     0,      1);
            rotationMatrix = transpose(R) * rot;
            angleVSOutput = angle;
            """

        shader_to_actor(self, 'vertex', decl_code=vs_dec,
                        impl_code=vs_impl)

        fs_vars_dec = \
            """
            in vec4 vertexMCVSOutput;
            in vec3 centerMCVSOutput;
            in float scaleVSOutput;
            in vec3 evalsVSOutput;
            in mat3 rotationMatrix;
            in float angleVSOutput;

            uniform mat4 MCVCMatrix;
            """

        sd_sphere = import_fury_shader(os.path.join('sdf', 'sd_sphere.frag'))

        sdf_map = \
            """
            float opUnion( float d1, float d2 ) { return min(d1,d2); }
            
            float sdCone( vec3 p, vec2 c, float h )
            {
                // c is the sin/cos of the angle, h is height
                // Alternatively pass q instead of (c,h),
                // which is the point at the base in 2D
                vec2 q = h*vec2(c.x/c.y,-1.0);
                  
                vec2 w = vec2( length(p.xz), p.y );
                vec2 a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );
                vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
                float k = sign( q.y );
                float d = min(dot( a, a ),dot(b, b));
                float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
                return sqrt(d)*sign(s);
            }
            
            float sdDoubleCone( vec3 p, vec2 c, float h )
            {
                return opUnion(sdCone(p,c,h),sdCone(-p,c,h));
            }
            
            float map(in vec3 position)
            {
                float a = clamp(angleVSOutput, 0, 6.283);
                //float a = angleVSOutput;
                vec2 angle = vec2(sin(a), cos(a));
                return sdDoubleCone((position - centerMCVSOutput)/scaleVSOutput
                    *rotationMatrix, angle, .5*angle.y) * scaleVSOutput;
                
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

        shader_to_actor(self, 'fragment', decl_code=fs_dec)

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
                vec3 color = blinnPhongIllumModel(la, lightColor0, 
                    diffuseColor, specularPower, specularColor, ambientColor);
                fragOutput0 = vec4(color, opacity);
            }
            else
            {
                discard;
            }
            """

        shader_to_actor(self, 'fragment', impl_code=sdf_frag_impl,
                        block='light')
