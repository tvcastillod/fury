void evalShGrad12(out float outSH[91], out vec3 outGrads[91], vec3 point)
{
    float x, y, z, z2, c0, s0, c1, s1, d, a, b;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    c0 = 1.0;
    s0 = 0.0;
    d = 0.282094792;
    outSH[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    outSH[3] = d;
    b = z2 * (a - 0.266666667);
    a = b - 0.257142857 * a;
    d = 3.702494142 * a;
    outSH[10] = d;
    b = z2 * a - 0.253968254 * b;
    a = b - 0.252525253 * a;
    d = 14.684485724 * a;
    outSH[21] = d;
    b = z2 * a - 0.251748252 * b;
    a = b - 0.251282051 * a;
    d = 58.473368113 * a;
    outSH[36] = d;
    b = z2 * a - 0.250980392 * b;
    a = b - 0.250773994 * a;
    d = 233.240148813 * a;
    outSH[55] = d;
    b = z2 * a - 0.250626566 * b;
    a = b - 0.250517598 * a;
    d = 931.186918633 * a;
    outSH[78] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    outSH[2] = -c1 * d;
    outSH[4] = s1 * d;
    outGrads[2][0] = -c0 * d;
    outGrads[4][0] = s0 * d;
    outGrads[2][1] = s0 * d;
    outGrads[4][1] = c0 * d;
    d = 1.892349392 * z;
    outGrads[3][2] = d;
    a = (z2 - 0.2) * z;
    b = a - 0.228571429 * z;
    d = -4.683325805 * b;
    outSH[9] = -c1 * d;
    outSH[11] = s1 * d;
    outGrads[9][0] = -c0 * d;
    outGrads[11][0] = s0 * d;
    outGrads[9][1] = s0 * d;
    outGrads[11][1] = c0 * d;
    d = 14.809976568 * b;
    outGrads[10][2] = d;
    a = z2 * b - 0.238095238 * a;
    b = a - 0.242424242 * b;
    d = -19.226504963 * b;
    outSH[20] = -c1 * d;
    outSH[22] = s1 * d;
    outGrads[20][0] = -c0 * d;
    outGrads[22][0] = s0 * d;
    outGrads[20][1] = s0 * d;
    outGrads[22][1] = c0 * d;
    d = 88.106914343 * b;
    outGrads[21][2] = d;
    a = z2 * b - 0.244755245 * a;
    b = a - 0.246153846 * b;
    d = -77.964490818 * b;
    outSH[35] = -c1 * d;
    outSH[37] = s1 * d;
    outGrads[35][0] = -c0 * d;
    outGrads[37][0] = s0 * d;
    outGrads[35][1] = s0 * d;
    outGrads[37][1] = c0 * d;
    d = 467.786944906 * b;
    outGrads[36][2] = d;
    a = z2 * b - 0.247058824 * a;
    b = a - 0.247678019 * b;
    d = -314.500952502 * b;
    outSH[54] = -c1 * d;
    outSH[56] = s1 * d;
    outGrads[54][0] = -c0 * d;
    outGrads[56][0] = s0 * d;
    outGrads[54][1] = s0 * d;
    outGrads[56][1] = c0 * d;
    d = 2332.401488133 * b;
    outGrads[55][2] = d;
    a = z2 * b - 0.248120301 * a;
    b = a - 0.248447205 * b;
    d = -1265.233874957 * b;
    outSH[77] = -c1 * d;
    outSH[79] = s1 * d;
    outGrads[77][0] = -c0 * d;
    outGrads[79][0] = s0 * d;
    outGrads[77][1] = s0 * d;
    outGrads[79][1] = c0 * d;
    d = 11174.243023595 * b;
    outGrads[78][2] = d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    outSH[1] = c0 * d;
    outSH[5] = s0 * d;
    d = 1.092548431;
    outGrads[1][0] = c1 * d;
    outGrads[5][0] = s1 * d;
    outGrads[1][1] = -s1 * d;
    outGrads[5][1] = c1 * d;
    d = -1.092548431;
    outGrads[2][2] = -c1 * d;
    outGrads[4][2] = s1 * d;
    a = z2 - 0.142857143;
    d = 3.311611435 * a;
    outSH[8] = c0 * d;
    outSH[12] = s0 * d;
    d = 6.62322287 * a;
    outGrads[8][0] = c1 * d;
    outGrads[12][0] = s1 * d;
    outGrads[8][1] = -s1 * d;
    outGrads[12][1] = c1 * d;
    d = -14.049977415 * a;
    outGrads[9][2] = -c1 * d;
    outGrads[11][2] = s1 * d;
    b = z2 * (a - 0.19047619);
    a = b - 0.212121212 * a;
    d = 15.199886782 * a;
    outSH[19] = c0 * d;
    outSH[23] = s0 * d;
    d = 30.399773564 * a;
    outGrads[19][0] = c1 * d;
    outGrads[23][0] = s1 * d;
    outGrads[19][1] = -s1 * d;
    outGrads[23][1] = c1 * d;
    d = -96.132524816 * a;
    outGrads[20][2] = -c1 * d;
    outGrads[22][2] = s1 * d;
    b = z2 * a - 0.223776224 * b;
    a = b - 0.230769231 * a;
    d = 65.229772956 * a;
    outSH[34] = c0 * d;
    outSH[38] = s0 * d;
    d = 130.459545912 * a;
    outGrads[34][0] = c1 * d;
    outGrads[38][0] = s1 * d;
    outGrads[34][1] = -s1 * d;
    outGrads[38][1] = c1 * d;
    d = -545.751435723 * a;
    outGrads[35][2] = -c1 * d;
    outGrads[37][2] = s1 * d;
    b = z2 * a - 0.235294118 * b;
    a = b - 0.238390093 * a;
    d = 272.365814381 * a;
    outSH[53] = c0 * d;
    outSH[57] = s0 * d;
    d = 544.731628762 * a;
    outGrads[53][0] = c1 * d;
    outGrads[57][0] = s1 * d;
    outGrads[53][1] = -s1 * d;
    outGrads[57][1] = c1 * d;
    d = -2830.508572514 * a;
    outGrads[54][2] = -c1 * d;
    outGrads[56][2] = s1 * d;
    b = z2 * a - 0.240601504 * b;
    a = b - 0.242236025 * a;
    d = 1121.509962433 * a;
    outSH[76] = c0 * d;
    outSH[80] = s0 * d;
    d = 2243.019924866 * a;
    outGrads[76][0] = c1 * d;
    outGrads[80][0] = s1 * d;
    outGrads[76][1] = -s1 * d;
    outGrads[80][1] = c1 * d;
    d = -13917.572624524 * a;
    outGrads[77][2] = -c1 * d;
    outGrads[79][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -1.77013077 * z;
    outSH[7] = -c1 * d;
    outSH[13] = s1 * d;
    d = -5.310392309 * z;
    outGrads[7][0] = -c0 * d;
    outGrads[13][0] = s0 * d;
    outGrads[7][1] = s0 * d;
    outGrads[13][1] = c0 * d;
    d = 6.62322287 * z;
    outGrads[8][2] = c0 * d;
    outGrads[12][2] = s0 * d;
    a = (z2 - 0.111111111) * z;
    b = a - 0.161616162 * z;
    d = -10.133257855 * b;
    outSH[18] = -c1 * d;
    outSH[24] = s1 * d;
    d = -30.399773564 * b;
    outGrads[18][0] = -c0 * d;
    outGrads[24][0] = s0 * d;
    outGrads[18][1] = s0 * d;
    outGrads[24][1] = c0 * d;
    d = 60.799547128 * b;
    outGrads[19][2] = c0 * d;
    outGrads[23][2] = s0 * d;
    a = z2 * b - 0.188811189 * a;
    b = a - 0.205128205 * b;
    d = -48.175380057 * b;
    outSH[33] = -c1 * d;
    outSH[39] = s1 * d;
    d = -144.52614017 * b;
    outGrads[33][0] = -c0 * d;
    outGrads[39][0] = s0 * d;
    outGrads[33][1] = s0 * d;
    outGrads[39][1] = c0 * d;
    d = 391.378637737 * b;
    outGrads[34][2] = c0 * d;
    outGrads[38][2] = s0 * d;
    a = z2 * b - 0.215686275 * a;
    b = a - 0.222910217 * b;
    d = -213.661323441 * b;
    outSH[52] = -c1 * d;
    outSH[58] = s1 * d;
    d = -640.983970322 * b;
    outGrads[52][0] = -c0 * d;
    outGrads[58][0] = s0 * d;
    outGrads[52][1] = s0 * d;
    outGrads[58][1] = c0 * d;
    d = 2178.926515046 * b;
    outGrads[53][2] = c0 * d;
    outGrads[57][2] = s0 * d;
    a = z2 * b - 0.228070175 * a;
    b = a - 0.231884058 * b;
    d = -915.709049803 * b;
    outSH[75] = -c1 * d;
    outSH[81] = s1 * d;
    d = -2747.127149409 * b;
    outGrads[75][0] = -c0 * d;
    outGrads[81][0] = s0 * d;
    outGrads[75][1] = s0 * d;
    outGrads[81][1] = c0 * d;
    d = 11215.099624332 * b;
    outGrads[76][2] = c0 * d;
    outGrads[80][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.625835735;
    outSH[6] = c0 * d;
    outSH[14] = s0 * d;
    d = 2.503342942;
    outGrads[6][0] = c1 * d;
    outGrads[14][0] = s1 * d;
    outGrads[6][1] = -s1 * d;
    outGrads[14][1] = c1 * d;
    d = -1.77013077;
    outGrads[7][2] = -c1 * d;
    outGrads[13][2] = s1 * d;
    a = z2 - 9.09090909e-02;
    d = 5.550213908 * a;
    outSH[17] = c0 * d;
    outSH[25] = s0 * d;
    d = 22.200855632 * a;
    outGrads[17][0] = c1 * d;
    outGrads[25][0] = s1 * d;
    outGrads[17][1] = -s1 * d;
    outGrads[25][1] = c1 * d;
    d = -30.399773564 * a;
    outGrads[18][2] = -c1 * d;
    outGrads[24][2] = s1 * d;
    b = z2 * (a - 0.13986014);
    a = b - 0.169230769 * a;
    d = 31.097074109 * a;
    outSH[32] = c0 * d;
    outSH[40] = s0 * d;
    d = 124.388296437 * a;
    outGrads[32][0] = c1 * d;
    outGrads[40][0] = s1 * d;
    outGrads[32][1] = -s1 * d;
    outGrads[40][1] = c1 * d;
    d = -240.876900283 * a;
    outGrads[33][2] = -c1 * d;
    outGrads[39][2] = s1 * d;
    b = z2 * a - 0.188235294 * b;
    a = b - 0.20123839 * a;
    d = 151.081370682 * a;
    outSH[51] = c0 * d;
    outSH[59] = s0 * d;
    d = 604.325482728 * a;
    outGrads[51][0] = c1 * d;
    outGrads[59][0] = s1 * d;
    outGrads[51][1] = -s1 * d;
    outGrads[59][1] = c1 * d;
    d = -1495.629264084 * a;
    outGrads[52][2] = -c1 * d;
    outGrads[58][2] = s1 * d;
    b = z2 * a - 0.210526316 * b;
    a = b - 0.217391304 * a;
    d = 686.781787352 * a;
    outSH[74] = c0 * d;
    outSH[82] = s0 * d;
    d = 2747.127149409 * a;
    outGrads[74][0] = c1 * d;
    outGrads[82][0] = s1 * d;
    outGrads[74][1] = -s1 * d;
    outGrads[82][1] = c1 * d;
    d = -8241.381448228 * a;
    outGrads[75][2] = -c1 * d;
    outGrads[81][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.366619162 * z;
    outSH[16] = -c1 * d;
    outSH[26] = s1 * d;
    d = -11.833095811 * z;
    outGrads[16][0] = -c0 * d;
    outGrads[26][0] = s0 * d;
    outGrads[16][1] = s0 * d;
    outGrads[26][1] = c0 * d;
    d = 11.100427816 * z;
    outGrads[17][2] = c0 * d;
    outGrads[25][2] = s0 * d;
    a = (z2 - 7.69230769e-02) * z;
    b = a - 0.123076923 * z;
    d = -17.24955311 * b;
    outSH[31] = -c1 * d;
    outSH[41] = s1 * d;
    d = -86.247765552 * b;
    outGrads[31][0] = -c0 * d;
    outGrads[41][0] = s0 * d;
    outGrads[31][1] = s0 * d;
    outGrads[41][1] = c0 * d;
    d = 124.388296437 * b;
    outGrads[32][2] = c0 * d;
    outGrads[40][2] = s0 * d;
    a = z2 * b - 0.152941176 * a;
    b = a - 0.173374613 * b;
    d = -95.552248675 * b;
    outSH[50] = -c1 * d;
    outSH[60] = s1 * d;
    d = -477.761243376 * b;
    outGrads[50][0] = -c0 * d;
    outGrads[60][0] = s0 * d;
    outGrads[50][1] = s0 * d;
    outGrads[60][1] = c0 * d;
    d = 906.488224092 * b;
    outGrads[51][2] = c0 * d;
    outGrads[59][2] = s0 * d;
    a = z2 * b - 0.187969925 * a;
    b = a - 0.198757764 * b;
    d = -471.12841933 * b;
    outSH[73] = -c1 * d;
    outSH[83] = s1 * d;
    d = -2355.642096651 * b;
    outGrads[73][0] = -c0 * d;
    outGrads[83][0] = s0 * d;
    outGrads[73][1] = s0 * d;
    outGrads[83][1] = c0 * d;
    d = 5494.254298819 * b;
    outGrads[74][2] = c0 * d;
    outGrads[82][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.683184105;
    outSH[15] = c0 * d;
    outSH[27] = s0 * d;
    d = 4.099104631;
    outGrads[15][0] = c1 * d;
    outGrads[27][0] = s1 * d;
    outGrads[15][1] = -s1 * d;
    outGrads[27][1] = c1 * d;
    d = -2.366619162;
    outGrads[16][2] = -c1 * d;
    outGrads[26][2] = s1 * d;
    a = z2 - 6.66666667e-02;
    d = 7.984991491 * a;
    outSH[30] = c0 * d;
    outSH[42] = s0 * d;
    d = 47.909948945 * a;
    outGrads[30][0] = c1 * d;
    outGrads[42][0] = s1 * d;
    outGrads[30][1] = -s1 * d;
    outGrads[42][1] = c1 * d;
    d = -51.748659331 * a;
    outGrads[31][2] = -c1 * d;
    outGrads[41][2] = s1 * d;
    b = z2 * (a - 0.109803922);
    a = b - 0.139318885 * a;
    d = 53.41533086 * a;
    outSH[49] = c0 * d;
    outSH[61] = s0 * d;
    d = 320.491985161 * a;
    outGrads[49][0] = c1 * d;
    outGrads[61][0] = s1 * d;
    outGrads[49][1] = -s1 * d;
    outGrads[61][1] = c1 * d;
    d = -477.761243376 * a;
    outGrads[50][2] = -c1 * d;
    outGrads[60][2] = s1 * d;
    b = z2 * a - 0.160401003 * b;
    a = b - 0.175983437 * a;
    d = 293.800188384 * a;
    outSH[72] = c0 * d;
    outSH[84] = s0 * d;
    d = 1762.801130306 * a;
    outGrads[72][0] = c1 * d;
    outGrads[84][0] = s1 * d;
    outGrads[72][1] = -s1 * d;
    outGrads[84][1] = c1 * d;
    d = -3297.898935312 * a;
    outGrads[73][2] = -c1 * d;
    outGrads[83][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -2.915706641 * z;
    outSH[29] = -c1 * d;
    outSH[43] = s1 * d;
    d = -20.409946485 * z;
    outGrads[29][0] = -c0 * d;
    outGrads[43][0] = s0 * d;
    outGrads[29][1] = s0 * d;
    outGrads[43][1] = c0 * d;
    d = 15.969982982 * z;
    outGrads[30][2] = c0 * d;
    outGrads[42][2] = s0 * d;
    a = (z2 - 5.88235294e-02) * z;
    b = a - 9.90712074e-02 * z;
    d = -25.910241313 * b;
    outSH[48] = -c1 * d;
    outSH[62] = s1 * d;
    d = -181.371689194 * b;
    outGrads[48][0] = -c0 * d;
    outGrads[62][0] = s0 * d;
    outGrads[48][1] = s0 * d;
    outGrads[62][1] = c0 * d;
    d = 213.661323441 * b;
    outGrads[49][2] = c0 * d;
    outGrads[61][2] = s0 * d;
    a = z2 * b - 0.127819549 * a;
    b = a - 0.149068323 * b;
    d = -165.101452729 * b;
    outSH[71] = -c1 * d;
    outSH[85] = s1 * d;
    d = -1155.7101691 * b;
    outGrads[71][0] = -c0 * d;
    outGrads[85][0] = s0 * d;
    outGrads[71][1] = s0 * d;
    outGrads[85][1] = c0 * d;
    d = 1762.801130306 * b;
    outGrads[72][2] = c0 * d;
    outGrads[84][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.72892666;
    outSH[28] = c0 * d;
    outSH[44] = s0 * d;
    d = 5.831413281;
    outGrads[28][0] = c1 * d;
    outGrads[44][0] = s1 * d;
    outGrads[28][1] = -s1 * d;
    outGrads[44][1] = c1 * d;
    d = -2.915706641;
    outGrads[29][2] = -c1 * d;
    outGrads[43][2] = s1 * d;
    a = z2 - 5.26315789e-02;
    d = 10.577811722 * a;
    outSH[47] = c0 * d;
    outSH[63] = s0 * d;
    d = 84.622493774 * a;
    outGrads[47][0] = c1 * d;
    outGrads[63][0] = s1 * d;
    outGrads[47][1] = -s1 * d;
    outGrads[63][1] = c1 * d;
    d = -77.73072394 * a;
    outGrads[48][2] = -c1 * d;
    outGrads[62][2] = s1 * d;
    b = z2 * (a - 9.02255639e-02);
    a = b - 0.118012422 * a;
    d = 82.550726364 * a;
    outSH[70] = c0 * d;
    outSH[86] = s0 * d;
    d = 660.405810914 * a;
    outGrads[70][0] = c1 * d;
    outGrads[86][0] = s1 * d;
    outGrads[70][1] = -s1 * d;
    outGrads[86][1] = c1 * d;
    d = -825.507263643 * a;
    outGrads[71][2] = -c1 * d;
    outGrads[85][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -3.4318953 * z;
    outSH[46] = -c1 * d;
    outSH[64] = s1 * d;
    d = -30.887057699 * z;
    outGrads[46][0] = -c0 * d;
    outGrads[64][0] = s0 * d;
    outGrads[46][1] = s0 * d;
    outGrads[64][1] = c0 * d;
    d = 21.155623443 * z;
    outGrads[47][2] = c0 * d;
    outGrads[63][2] = s0 * d;
    a = (z2 - 4.76190476e-02) * z;
    b = a - 8.2815735e-02 * z;
    d = -36.028090689 * b;
    outSH[69] = -c1 * d;
    outSH[87] = s1 * d;
    d = -324.252816204 * b;
    outGrads[69][0] = -c0 * d;
    outGrads[87][0] = s0 * d;
    outGrads[69][1] = s0 * d;
    outGrads[87][1] = c0 * d;
    d = 330.202905457 * b;
    outGrads[70][2] = c0 * d;
    outGrads[86][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.767395118;
    outSH[45] = c0 * d;
    outSH[65] = s0 * d;
    d = 7.673951182;
    outGrads[45][0] = c1 * d;
    outGrads[65][0] = s1 * d;
    outGrads[45][1] = -s1 * d;
    outGrads[65][1] = c1 * d;
    d = -3.4318953;
    outGrads[46][2] = -c1 * d;
    outGrads[64][2] = s1 * d;
    a = z2 - 4.34782609e-02;
    d = 13.3042542 * a;
    outSH[68] = c0 * d;
    outSH[88] = s0 * d;
    d = 133.042542003 * a;
    outGrads[68][0] = c1 * d;
    outGrads[88][0] = s1 * d;
    outGrads[68][1] = -s1 * d;
    outGrads[88][1] = c1 * d;
    d = -108.084272068 * a;
    outGrads[69][2] = -c1 * d;
    outGrads[87][2] = s1 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -3.923210529 * z;
    outSH[67] = -c1 * d;
    outSH[89] = s1 * d;
    d = -43.155315818 * z;
    outGrads[67][0] = -c0 * d;
    outGrads[89][0] = s0 * d;
    outGrads[67][1] = s0 * d;
    outGrads[89][1] = c0 * d;
    d = 26.608508401 * z;
    outGrads[68][2] = c0 * d;
    outGrads[88][2] = s0 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.800821996;
    outSH[66] = c0 * d;
    outSH[90] = s0 * d;
    d = 9.609863949;
    outGrads[66][0] = c1 * d;
    outGrads[90][0] = s1 * d;
    outGrads[66][1] = -s1 * d;
    outGrads[90][1] = c1 * d;
    d = -3.923210529;
    outGrads[67][2] = -c1 * d;
    outGrads[89][2] = s1 * d;
    outGrads[0][0] = 0.0;
    outGrads[0][1] = 0.0;
    outGrads[0][2] = 0.0;
    outGrads[1][2] = 0.0;
    outGrads[3][0] = 0.0;
    outGrads[3][1] = 0.0;
    outGrads[5][2] = 0.0;
    outGrads[6][2] = 0.0;
    outGrads[10][0] = 0.0;
    outGrads[10][1] = 0.0;
    outGrads[14][2] = 0.0;
    outGrads[15][2] = 0.0;
    outGrads[21][0] = 0.0;
    outGrads[21][1] = 0.0;
    outGrads[27][2] = 0.0;
    outGrads[28][2] = 0.0;
    outGrads[36][0] = 0.0;
    outGrads[36][1] = 0.0;
    outGrads[44][2] = 0.0;
    outGrads[45][2] = 0.0;
    outGrads[55][0] = 0.0;
    outGrads[55][1] = 0.0;
    outGrads[65][2] = 0.0;
    outGrads[66][2] = 0.0;
    outGrads[78][0] = 0.0;
    outGrads[78][1] = 0.0;
    outGrads[90][2] = 0.0;
}
