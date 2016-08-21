#include <cstdio>

// which version of SPMD to use
#define V2

#ifdef V1
//#include "spmd_sse2-i32x4.h"
#include "spmd_avx2-i32x8.h"

void simple(float vin[], float vout[], 
            int n)
{
    spmd_foreach(0, n, [&](const lint& index)
    {
        vfloat v = index[vin];
        spmd_ifelse(v < 3.0f,
            [&] { v = v * v; },
        /* else */
            [&] { v = sqrt(v); });
        index[vout] = v;
    });
}

int main()
{
    __declspec(align(32)) float vin[16], vout[16];
    for (int i = 0; i < 16; ++i)
        vin[i] = (float)i;

    simple(vin, vout, 16);

    for (int i = 0; i < 16; ++i)
        printf("%d: simple(%f) = %f\n", i, vin[i], vout[i]);
}
#endif

#ifdef V2
#include "spmdv2_avx2-i32x8.h"

struct simple : spmd_kernel
{
    simple(float vin[], float vout[],
           int n)
    {
        spmd_foreach(0, n, [&](const lint& index)
        {
            vfloat v = load(index[vin]);
            spmd_ifelse(less(v, 3.0f),
                [&] { store(v, mul(v, v)); },
            /* else */
                [&] { store(v, sqrt(v)); });
            store(index[vout], v);
        });
    }
};

int main()
{
    __declspec(align(32)) float vin[16], vout[16];
    for (int i = 0; i < 16; ++i)
        vin[i] = (float)i;

    simple(vin, vout, 16);

    for (int i = 0; i < 16; ++i)
        printf("%d: simple(%f) = %f\n", i, vin[i], vout[i]);
}
#endif