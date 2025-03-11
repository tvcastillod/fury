float P(int l, int m, float x )
{
    float pmm = 1;

    float somx2 = sqrt((1 - x) * (1 + x));
    float fact = 1;
    for (int i=1; i<=m; i++) {
        pmm *= -fact * somx2;
        fact += 2;
    }

    if( l == m )
        return pmm;

    float pmmp1 = x * (2 * m + 1) * pmm;
    if(l == m + 1)
        return pmmp1;

    float pll = 0;
    for (float ll=m + 2; ll<=l; ll+=1) {
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }

    return pll;
}
