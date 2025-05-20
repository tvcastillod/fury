float calculateSH(int l, int m, vec3 s, bool legacy)
{
    vec3 normS = normalize(s);
    float thetaX = normS.z;
    float phi = atan(normS.y, normS.x);
    int mParam = legacy ? abs(m) : m;
    float result = calculateKFactor(l, mParam) * calcAssocLegendrePoly(l, mParam, thetaX);
    if(m != 0)
        result *= sqrt(2);
    if(m > 0)
        result *= sin(m * phi);
    if(m < 0)
        result *= cos(-m * phi);

    return result;
}
