void eval_sh_6(out float out_shs[28], vec3 point)
{
    float x, y, z, z2, c0, s0, c1, s1, d, a, b;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    d = 0.282094792;
    out_shs[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    out_shs[3] = d;
    b = z2 * (a - 0.266666667);
    a = b - 0.257142857 * a;
    d = 3.702494142 * a;
    out_shs[10] = d;
    b = z2 * a - 0.253968254 * b;
    a = b - 0.252525253 * a;
    d = 14.684485724 * a;
    out_shs[21] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    out_shs[4] = c1 * d;
    out_shs[2] = s1 * d;
    a = (z2 - 0.2) * z;
    b = a - 0.228571429 * z;
    d = -4.683325805 * b;
    out_shs[11] = c1 * d;
    out_shs[9] = s1 * d;
    a = z2 * b - 0.238095238 * a;
    b = a - 0.242424242 * b;
    d = -19.226504963 * b;
    out_shs[22] = c1 * d;
    out_shs[20] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[5] = c0 * d;
    out_shs[1] = s0 * d;
    a = z2 - 0.142857143;
    d = 3.311611435 * a;
    out_shs[12] = c0 * d;
    out_shs[8] = s0 * d;
    b = z2 * (a - 0.19047619);
    a = b - 0.212121212 * a;
    d = 15.199886782 * a;
    out_shs[23] = c0 * d;
    out_shs[19] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -1.77013077 * z;
    out_shs[13] = c1 * d;
    out_shs[7] = s1 * d;
    a = (z2 - 0.111111111) * z;
    b = a - 0.161616162 * z;
    d = -10.133257855 * b;
    out_shs[24] = c1 * d;
    out_shs[18] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.625835735;
    out_shs[14] = c0 * d;
    out_shs[6] = s0 * d;
    a = z2 - 9.09090909e-02;
    d = 5.550213908 * a;
    out_shs[25] = c0 * d;
    out_shs[17] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.366619162 * z;
    out_shs[26] = c1 * d;
    out_shs[16] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.683184105;
    out_shs[27] = c0 * d;
    out_shs[15] = s0 * d;
}
