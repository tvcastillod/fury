void eval_sh_2(out float out_shs[6], vec3 point)
{
    float x, y, z, z2, c0, s0, c1, s1, d, a;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    d = 0.282094792;
    out_shs[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    out_shs[3] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    out_shs[2] = -c1 * d;
    out_shs[4] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[1] = c0 * d;
    out_shs[5] = s0 * d;
}
