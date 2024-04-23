void eval_sh_grad_6(out float out_shs[28], out vec3 out_grads[28], vec3 point)
{
    float x, y, z, z2, c0, s0, c1, s1, d, a, b;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    c0 = 1.0;
    s0 = 0.0;
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
    out_grads[4][0] = c0 * d;
    out_grads[2][0] = s0 * d;
    out_grads[4][1] = s0 * d;
    out_grads[2][1] = c0 * d;
    d = 1.892349392 * z;
    out_grads[3][2] = d;
    a = (z2 - 0.2) * z;
    b = a - 0.228571429 * z;
    d = -4.683325805 * b;
    out_shs[11] = c1 * d;
    out_shs[9] = s1 * d;
    out_grads[11][0] = c0 * d;
    out_grads[9][0] = s0 * d;
    out_grads[11][1] = s0 * d;
    out_grads[9][1] = c0 * d;
    d = 14.809976568 * b;
    out_grads[10][2] = d;
    a = z2 * b - 0.238095238 * a;
    b = a - 0.242424242 * b;
    d = -19.226504963 * b;
    out_shs[22] = c1 * d;
    out_shs[20] = s1 * d;
    out_grads[22][0] = c0 * d;
    out_grads[20][0] = s0 * d;
    out_grads[22][1] = s0 * d;
    out_grads[20][1] = c0 * d;
    d = 88.106914343 * b;
    out_grads[21][2] = d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    out_shs[5] = c0 * d;
    out_shs[1] = s0 * d;
    d = 1.092548431;
    out_grads[5][0] = c1 * d;
    out_grads[1][0] = s1 * d;
    out_grads[5][1] = -s1 * d;
    out_grads[1][1] = c1 * d;
    d = -1.092548431;
    out_grads[4][2] = c1 * d;
    out_grads[2][2] = s1 * d;
    a = z2 - 0.142857143;
    d = 3.311611435 * a;
    out_shs[12] = c0 * d;
    out_shs[8] = s0 * d;
    d = 6.62322287 * a;
    out_grads[12][0] = c1 * d;
    out_grads[8][0] = s1 * d;
    out_grads[12][1] = -s1 * d;
    out_grads[8][1] = c1 * d;
    d = -14.049977415 * a;
    out_grads[11][2] = c1 * d;
    out_grads[9][2] = s1 * d;
    b = z2 * (a - 0.19047619);
    a = b - 0.212121212 * a;
    d = 15.199886782 * a;
    out_shs[23] = c0 * d;
    out_shs[19] = s0 * d;
    d = 30.399773564 * a;
    out_grads[23][0] = c1 * d;
    out_grads[19][0] = s1 * d;
    out_grads[23][1] = -s1 * d;
    out_grads[19][1] = c1 * d;
    d = -96.132524816 * a;
    out_grads[22][2] = c1 * d;
    out_grads[20][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -1.77013077 * z;
    out_shs[13] = c1 * d;
    out_shs[7] = s1 * d;
    d = -5.310392309 * z;
    out_grads[13][0] = c0 * d;
    out_grads[7][0] = s0 * d;
    out_grads[13][1] = s0 * d;
    out_grads[7][1] = c0 * d;
    d = 6.62322287 * z;
    out_grads[12][2] = c0 * d;
    out_grads[8][2] = s0 * d;
    a = (z2 - 0.111111111) * z;
    b = a - 0.161616162 * z;
    d = -10.133257855 * b;
    out_shs[24] = c1 * d;
    out_shs[18] = s1 * d;
    d = -30.399773564 * b;
    out_grads[24][0] = c0 * d;
    out_grads[18][0] = s0 * d;
    out_grads[24][1] = s0 * d;
    out_grads[18][1] = c0 * d;
    d = 60.799547128 * b;
    out_grads[23][2] = c0 * d;
    out_grads[19][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.625835735;
    out_shs[14] = c0 * d;
    out_shs[6] = s0 * d;
    d = 2.503342942;
    out_grads[14][0] = c1 * d;
    out_grads[6][0] = s1 * d;
    out_grads[14][1] = -s1 * d;
    out_grads[6][1] = c1 * d;
    d = -1.77013077;
    out_grads[13][2] = c1 * d;
    out_grads[7][2] = s1 * d;
    a = z2 - 9.09090909e-02;
    d = 5.550213908 * a;
    out_shs[25] = c0 * d;
    out_shs[17] = s0 * d;
    d = 22.200855632 * a;
    out_grads[25][0] = c1 * d;
    out_grads[17][0] = s1 * d;
    out_grads[25][1] = -s1 * d;
    out_grads[17][1] = c1 * d;
    d = -30.399773564 * a;
    out_grads[24][2] = c1 * d;
    out_grads[18][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.366619162 * z;
    out_shs[26] = c1 * d;
    out_shs[16] = s1 * d;
    d = -11.833095811 * z;
    out_grads[26][0] = c0 * d;
    out_grads[16][0] = s0 * d;
    out_grads[26][1] = s0 * d;
    out_grads[16][1] = c0 * d;
    d = 11.100427816 * z;
    out_grads[25][2] = c0 * d;
    out_grads[17][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.683184105;
    out_shs[27] = c0 * d;
    out_shs[15] = s0 * d;
    d = 4.099104631;
    out_grads[27][0] = c1 * d;
    out_grads[15][0] = s1 * d;
    out_grads[27][1] = -s1 * d;
    out_grads[15][1] = c1 * d;
    d = -2.366619162;
    out_grads[26][2] = c1 * d;
    out_grads[16][2] = s1 * d;
    out_grads[0][0] = 0.0;
    out_grads[0][1] = 0.0;
    out_grads[0][2] = 0.0;
    out_grads[5][2] = 0.0;
    out_grads[3][0] = 0.0;
    out_grads[3][1] = 0.0;
    out_grads[1][2] = 0.0;
    out_grads[14][2] = 0.0;
    out_grads[10][0] = 0.0;
    out_grads[10][1] = 0.0;
    out_grads[6][2] = 0.0;
    out_grads[27][2] = 0.0;
    out_grads[21][0] = 0.0;
    out_grads[21][1] = 0.0;
    out_grads[15][2] = 0.0;
}