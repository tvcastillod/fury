float K(int l, int m)
{
    float n = (2 * l + 1) * factorial(l - m);
    float d = 4 * PI * factorial(l + m);
    return sqrt(n / d);
}
