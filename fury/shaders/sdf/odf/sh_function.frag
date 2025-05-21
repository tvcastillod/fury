float calculateSH(int l, int m, vec3 s, bool legacy, bool isDescoteauxBasis)
{
    vec3 normS = normalize(s);
    float thetaX = normS.z;
    float phi = atan(normS.y, normS.x);
    int mParam = isDescoteauxBasis ? (legacy ? abs(m) : m) : abs(m);
    float cons1 = isDescoteauxBasis ? sqrt(2.0) : (legacy ? sqrt(2.0) : 1.0);
    float cons2 = isDescoteauxBasis ? sin(m * phi) : cos(m * phi);
    float cons3 = isDescoteauxBasis ? cos(-m * phi) : sin(-m * phi);
    float result = calculateKFactor(l, mParam) * calcAssocLegendrePoly(l, mParam, thetaX);
    if(m != 0)
        result *= cons1;
    if(m > 0)
        result *= cons2;
    if(m < 0)
        result *= cons3;
    return result;
}
