float calcAssocLegendrePoly(int l, int m, float x)
{
    int absM = abs(m);
    float legendrePmm = 1.0; // P(m, m, x)
    float sqrtTerm = sqrt((1.0 - x) * (1.0 + x)); // sqrt(1 - x^2) term
    float factor = 1.0;

    // Calculate P(m, m, x)
    for (int i = 1; i <= absM; ++i) {
        legendrePmm *= -factor * sqrtTerm;
        factor += 2.0;
    }

    if (l == absM) {
        legendrePmm = legendrePmm;
    }

    float legendrePmmPlus1 = x * (2.0 * absM + 1.0) * legendrePmm; // P(m+1, m, x)
    if (l == absM + 1) {
        legendrePmm = legendrePmmPlus1;
    }

    float legendrePlm = 0.0; // P(l, m, x)
    float prevPmm = legendrePmm; // P(l-2, m, x) in iteration
    float prevPmmPlus1 = legendrePmmPlus1; // P(l-1, m, x) in iteration

    // Calculate P(l, m, x) using recurrence relation for l > m + 1
    for (int lIndex = absM + 2; lIndex <= l; ++lIndex) {
        legendrePlm = ((2.0 * lIndex - 1.0) * x * prevPmmPlus1 - (lIndex + absM - 1.0) * prevPmm) / (lIndex - absM);
        prevPmm = prevPmmPlus1;
        prevPmmPlus1 = legendrePlm;
    }

    float result = (l == absM) ? legendrePmm : (l == absM + 1) ? legendrePmmPlus1 : legendrePlm;

    // Apply symmetry for negative m
    if (m < 0) {
        float sign = (absM % 2 == 0) ? 1.0 : -1.0;
        float ratio = 1.0;
        for (int i = l - absM + 1; i <= l + absM; ++i) {
            ratio /= i;
        }
        result *= sign * ratio;
    }

    return result;
}