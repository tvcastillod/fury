float SH(int l, int m, in vec3 s)
{
    vec3 ns = normalize(s);
    float thetax = ns.z;
    float phi = atan(ns.y, ns.x);
    float v = K(l, abs(m)) * P(l, abs(m), thetax);
    if(m != 0)
        v *= sqrt(2);
    if(m > 0)
        v *= sin(m * phi);
    if(m < 0)
        v *= cos(-m * phi);

    return v;
}
