#include <cstdio>

// which version to use
//#define SCALAR
#define CPPSPMD
//#define ISPC

// which test to run
//#define SIMPLE
#define NOISE

#define SPMD_NOISE_OPTIMIZATION

#ifdef SCALAR
#include <cmath>

#ifdef SIMPLE
void simple(float vin[], float vout[], int count) {
    for (int index = 0; index < count; index++) {
        float v = vin[index];
        
        if (v < 3.0f)
            v = v * v;
        else
            v = sqrt(v);

        vout[index] = v;
    }
}
#endif // SIMPLE

#ifdef NOISE
#define NOISE_PERM_SIZE 256

static int NoisePerm[2 * NOISE_PERM_SIZE] = {
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140,
    36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120,
    234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
    88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168,  68, 175, 74, 165, 71, 
    134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 
    230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161,
    1, 216, 80, 73, 209, 76, 132, 187, 208,  89, 18, 169, 200, 196, 135, 130, 
    116, 188, 159, 86, 164, 100, 109, 198, 173, 186,  3, 64, 52, 217, 226, 250,
    124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 
    47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152,  2, 44,
    154, 163, 70, 221, 153, 101, 155, 167,  43, 172, 9, 129, 22, 39, 253,  19, 
    98, 108, 110, 79, 113, 224, 232, 178, 185,  112, 104, 218, 246, 97, 228, 251,
    34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249,
    14, 239, 107, 49, 192, 214,  31, 181, 199, 106, 157, 184, 84, 204, 176, 115,
    121, 50, 45, 127,  4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 
    243, 141, 128, 195, 78, 66, 215, 61, 156, 180, 151, 160, 137, 91, 90, 15,
    131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99,
    37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252,
    219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 
    136, 171, 168,  68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158,
    231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245,
    40, 244, 102, 143, 54,  65, 25, 63, 161,  1, 216, 80, 73, 209, 76, 132, 187,
    208,  89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 
    198, 173, 186,  3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118,
    126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
    223, 183, 170, 213, 119, 248, 152,  2, 44, 154, 163, 70, 221, 153, 101, 155, 
    167,  43, 172, 9, 129, 22, 39, 253,  19, 98, 108, 110, 79, 113, 224, 232,
    178, 185,  112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144,
    12, 191, 179, 162, 241,  81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214,
    31, 181, 199, 106, 157, 184,  84, 204, 176, 115, 121, 50, 45, 127,  4, 150,
    254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 
    66, 215, 61, 156, 180
};


inline float Clamp(float v, float low, float high) {
    return v < low ? low : ((v > high) ? high : v);
}


inline float SmoothStep(float low, float high, float value) {
    float v = Clamp((value - low) / (high - low), 0.f, 1.f);
    return v * v * (-2.f * v  + 3.f);
}


inline int Floor2Int(float val) {
    return (int)floorf(val);
}


inline float Grad(int x, int y, int z, float dx, float dy, float dz) {
    int h = NoisePerm[NoisePerm[NoisePerm[x]+y]+z];
    h &= 15;
    float u = h<8 || h==12 || h==13 ? dx : dy;
    float v = h<4 || h==12 || h==13 ? dy : dz;
    return ((h&1) ? -u : u) + ((h&2) ? -v : v);
}


inline float NoiseWeight(float t) {
    float t3 = t*t*t;
    float t4 = t3*t;
    return 6.f*t4*t - 15.f*t4 + 10.f*t3;
}


inline float Lerp(float t, float low, float high) {
    return (1.f - t) * low + t * high;
}


static float Noise(float x, float y, float z) {
    // Compute noise cell coordinates and offsets
    int ix = Floor2Int(x), iy = Floor2Int(y), iz = Floor2Int(z);
    float dx = x - ix, dy = y - iy, dz = z - iz;

    // Compute gradient weights
    ix &= (NOISE_PERM_SIZE-1);
    iy &= (NOISE_PERM_SIZE-1);
    iz &= (NOISE_PERM_SIZE-1);
    float w000 = Grad(ix,   iy,   iz,   dx,   dy,   dz);
    float w100 = Grad(ix+1, iy,   iz,   dx-1, dy,   dz);
    float w010 = Grad(ix,   iy+1, iz,   dx,   dy-1, dz);
    float w110 = Grad(ix+1, iy+1, iz,   dx-1, dy-1, dz);
    float w001 = Grad(ix,   iy,   iz+1, dx,   dy,   dz-1);
    float w101 = Grad(ix+1, iy,   iz+1, dx-1, dy,   dz-1);
    float w011 = Grad(ix,   iy+1, iz+1, dx,   dy-1, dz-1);
    float w111 = Grad(ix+1, iy+1, iz+1, dx-1, dy-1, dz-1);

    // Compute trilinear interpolation of weights
    float wx = NoiseWeight(dx), wy = NoiseWeight(dy), wz = NoiseWeight(dz);
    float x00 = Lerp(wx, w000, w100);
    float x10 = Lerp(wx, w010, w110);
    float x01 = Lerp(wx, w001, w101);
    float x11 = Lerp(wx, w011, w111);
    float y0 = Lerp(wy, x00, x10);
    float y1 = Lerp(wy, x01, x11);
    return Lerp(wz, y0, y1);
}


static float Turbulence(float x, float y, float z, int octaves) {
    float omega = 0.6f;

    float sum = 0., lambda = 1., o = 1.;
    for (int i = 0; i < octaves; ++i) {
        sum += fabsf(o * Noise(lambda * x, lambda * y, lambda * z));
        lambda *= 1.99f;
        o *= omega;
    }
    return sum * 0.5f;
}


void noise(float x0, float y0, float x1, float y1,
           int width, int height, float output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; ++i) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = (j * width + i);
            output[index] = Turbulence(x, y, 0.6f, 8);
        }
    }
}
#endif // NOISE

#endif // SCALAR

#ifdef CPPSPMD
#include "spmd_avx2-i32x8.h"

#ifdef SIMPLE
struct simple : spmd_kernel
{
    void _call(float vin[], float vout[],
               int n)
    {
        spmd_foreach(0, n, [&](const lint& index)
        {
            vfloat v = load(index[vin]);
            spmd_ifelse(v < 3.0f,
                [&] { store(v, v * v); },
            /* else */
                [&] { store(v, sqrt(v)); });
            store(index[vout], v);
        });
    }
};
#endif // SIMPLE

#ifdef NOISE
#define NOISE_PERM_SIZE 256

static int NoisePerm[2 * NOISE_PERM_SIZE] = {
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140,
    36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120,
    234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
    88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168,  68, 175, 74, 165, 71, 
    134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 
    230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161,
    1, 216, 80, 73, 209, 76, 132, 187, 208,  89, 18, 169, 200, 196, 135, 130, 
    116, 188, 159, 86, 164, 100, 109, 198, 173, 186,  3, 64, 52, 217, 226, 250,
    124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 
    47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152,  2, 44,
    154, 163, 70, 221, 153, 101, 155, 167,  43, 172, 9, 129, 22, 39, 253,  19, 
    98, 108, 110, 79, 113, 224, 232, 178, 185,  112, 104, 218, 246, 97, 228, 251,
    34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249,
    14, 239, 107, 49, 192, 214,  31, 181, 199, 106, 157, 184, 84, 204, 176, 115,
    121, 50, 45, 127,  4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 
    243, 141, 128, 195, 78, 66, 215, 61, 156, 180, 151, 160, 137, 91, 90, 15,
    131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99,
    37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252,
    219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 
    136, 171, 168,  68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158,
    231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245,
    40, 244, 102, 143, 54,  65, 25, 63, 161,  1, 216, 80, 73, 209, 76, 132, 187,
    208,  89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 
    198, 173, 186,  3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118,
    126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
    223, 183, 170, 213, 119, 248, 152,  2, 44, 154, 163, 70, 221, 153, 101, 155, 
    167,  43, 172, 9, 129, 22, 39, 253,  19, 98, 108, 110, 79, 113, 224, 232,
    178, 185,  112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144,
    12, 191, 179, 162, 241,  81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214,
    31, 181, 199, 106, 157, 184,  84, 204, 176, 115, 121, 50, 45, 127,  4, 150,
    254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 
    66, 215, 61, 156, 180
};


struct SmoothStep : spmd_kernel
{
    vfloat _call(vfloat low, vfloat high, vfloat value) {
        vfloat v = clamp((value - low) / (high - low), 0.f, 1.f);
        return v * v * (-2.f * v + 3.f);
    }
};


struct Floor2Int : spmd_kernel
{
    vint _call(const vfloat& val) {
        return (vint)floor(val);
    }
};


struct Grad : spmd_kernel
{
#ifdef SPMD_NOISE_OPTIMIZATION
    vfloat _call(const vint& h, const vfloat& dx, const vfloat& dy, const vfloat& dz) {
#else
    vfloat _call(const vint& x, const vint& y, const vint& z, const vfloat& dx, const vfloat& dy, const vfloat& dz) {
        vint h = load((load((load(x[NoisePerm]) + y)[NoisePerm]) + z)[NoisePerm]);
        store(h, h & 15);
#endif
        vfloat u = spmd_ternary(h < 8 || h == 12 || h == 13, dx, dy);
        vfloat v = spmd_ternary(h < 4 || h == 12 || h == 13, dy, dz);
        return spmd_ternary(h & 1, -u, u) + spmd_ternary(h & 2, -v, v);
    }
};


struct NoiseWeight : spmd_kernel
{
    vfloat _call(const vfloat& t) {
        vfloat t3 = t*t*t;
        vfloat t4 = t3*t;
#ifdef SPMD_NOISE_OPTIMIZATION
        return fma(6.f*t4,t, fnma(15.f,t4,10.f*t3));
#else
        return 6.f*t4*t - 15.f*t4 + 10.f*t3;
#endif
    }
};


struct Lerp : spmd_kernel
{
    vfloat _call(const vfloat& t, const vfloat& low, const vfloat& high) {
#ifdef SPMD_NOISE_OPTIMIZATION
        return fnma(t, low, fma(t, high, low));
#else
        return (1.0f - t) * low + t * high;
#endif
    }
};

struct Noise : spmd_kernel
{
    vfloat _call(const vfloat& x, const vfloat& y, const vfloat& z)
    {
        // Compute noise cell coordinates and offsets
        vint ix = spmd_call<Floor2Int>(x), iy = spmd_call<Floor2Int>(y), iz = spmd_call<Floor2Int>(z);
        vfloat dx = x - ix, dy = y - iy, dz = z - iz;

        // Compute gradient weights
        store(ix, ix & (NOISE_PERM_SIZE - 1));
        store(iy, iy & (NOISE_PERM_SIZE - 1));
        store(iz, iz & (NOISE_PERM_SIZE - 1));

#ifdef SPMD_NOISE_OPTIMIZATION
        vint xx0 = load(ix[NoisePerm]);
        vint xx1 = load((ix + 1)[NoisePerm]);
        vint yy0 = load((xx0 + iy)[NoisePerm]);
        vint yy1 = load((xx0 + (iy + 1))[NoisePerm]);
        vint yy10 = load((xx1 + iy)[NoisePerm]);
        vint zz0 = load((yy0 + iz)[NoisePerm]);
        vint yy11 = load((xx1 + (iy + 1))[NoisePerm]);
        vint h0 = zz0 & 15;
        vint h1 = load((yy10 + iz)[NoisePerm]) & 15;
        vint h2 = load((yy1 + iz)[NoisePerm]) & 15;
        vint h3 = load((yy11 + iz)[NoisePerm]) & 15;
        vint h4 = load((yy0 + iz)[NoisePerm+1]) & 15;
        vint h5 = load((yy10 + iz)[NoisePerm+1]) & 15;
        vint h6 = load((yy1 + iz)[NoisePerm+1]) & 15;
        vint h7 = load((yy11 + iz)[NoisePerm+1]) & 15;

        vfloat w000 = spmd_call<Grad>(h0, dx, dy, dz);
        vfloat w100 = spmd_call<Grad>(h1, dx - 1, dy, dz);
        vfloat w010 = spmd_call<Grad>(h2, dx, dy - 1, dz);
        vfloat w110 = spmd_call<Grad>(h3, dx - 1, dy - 1, dz);
        vfloat w001 = spmd_call<Grad>(h4, dx, dy, dz - 1);
        vfloat w101 = spmd_call<Grad>(h5, dx - 1, dy, dz - 1);
        vfloat w011 = spmd_call<Grad>(h6, dx, dy - 1, dz - 1);
        vfloat w111 = spmd_call<Grad>(h7, dx - 1, dy - 1, dz - 1);
#else
        vfloat w000 = spmd_call<Grad>(ix, iy, iz, dx, dy, dz);
        vfloat w100 = spmd_call<Grad>(ix + 1, iy, iz, dx - 1, dy, dz);
        vfloat w010 = spmd_call<Grad>(ix, iy + 1, iz, dx, dy - 1, dz);
        vfloat w110 = spmd_call<Grad>(ix + 1, iy + 1, iz, dx - 1, dy - 1, dz);
        vfloat w001 = spmd_call<Grad>(ix, iy, iz + 1, dx, dy, dz - 1);
        vfloat w101 = spmd_call<Grad>(ix + 1, iy, iz + 1, dx - 1, dy, dz - 1);
        vfloat w011 = spmd_call<Grad>(ix, iy + 1, iz + 1, dx, dy - 1, dz - 1);
        vfloat w111 = spmd_call<Grad>(ix + 1, iy + 1, iz + 1, dx - 1, dy - 1, dz - 1);
#endif

        // Compute trilinear interpolation of weights
        vfloat wx = spmd_call<NoiseWeight>(dx), wy = spmd_call<NoiseWeight>(dy), wz = spmd_call<NoiseWeight>(dz);
        vfloat x00 = spmd_call<Lerp>(wx, w000, w100);
        vfloat x10 = spmd_call<Lerp>(wx, w010, w110);
        vfloat x01 = spmd_call<Lerp>(wx, w001, w101);
        vfloat x11 = spmd_call<Lerp>(wx, w011, w111);
        vfloat y0 = spmd_call<Lerp>(wy, x00, x10);
        vfloat y1 = spmd_call<Lerp>(wy, x01, x11);
        return spmd_call<Lerp>(wz, y0, y1);
    }
};

struct Turbulence : spmd_kernel
{
    vfloat _call(const vfloat& x, const vfloat& y, const vfloat& z, int octaves)
    {
        float omega = 0.6f;

        vfloat sum = 0.0f, lambda = 1.0f, o = 1.0f;
        for (int i = 0; i < octaves; ++i) {
            store(sum, sum + abs(o * spmd_call<Noise>(lambda * x, lambda * y, lambda * z)));
            store(lambda, lambda * 1.99f);
            store(o, o * omega);
        }
        return sum * 0.5f;
    }
};

struct noise : spmd_kernel
{
    void _call(float x0, float y0, float x1, 
               float y1, int width, int height, 
               float output[])
    {
        float dx = (x1 - x0) / width;
        float dy = (y1 - y0) / height;

        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i += programCount) {
                vfloat x = x0 + (i + programIndex) * dx;
                vfloat y = y0 + j * dy;

                lint index = (j * width + i + programIndex);
                store(index[output], spmd_call<Turbulence>(x, y, 0.6f, 8));
            }
        }
    }
};
#endif // NOISE

#endif // CPPSPMD

#ifdef ISPC

#ifdef SIMPLE
#include "simple.ispc.h"
#endif // SIMPLE

#ifdef NOISE
#include "noise.ispc.h"
#endif // NOISE

#endif // ISPC

#ifdef SIMPLE
int main()
{
    __declspec(align(32)) float vin[16], vout[16];
    for (int i = 0; i < 16; ++i)
        vin[i] = (float)i;

#ifdef SCALAR
    simple(vin, vout, 16);
#endif // SCALAR

#ifdef CPPSPMD
    spmd_call<simple>(vin, vout, 16);
#endif // CPPSPMD

#ifdef ISPC
    ispc::simple(vin, vout, 16);
#endif // ISPC

    for (int i = 0; i < 16; ++i)
        printf("%d: simple(%f) = %f\n", i, vin[i], vout[i]);
}
#endif // SIMPLE

#ifdef NOISE
/* Write a PPM image file with the image */
static void
writePPM(float *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width*height; ++i) {
        float v = buf[i] * 255.f;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        for (int j = 0; j < 3; ++j)
            fputc((char)v, fp);
    }
    fclose(fp);
}

int main()
{
    unsigned int width = 768;
    unsigned int height = 768;
    float x0 = -10;
    float x1 = 10;
    float y0 = -10;
    float y1 = 10;

    float *buf = new float[width*height];

    int num_iterations = 100;

    for (int i = 0; i < num_iterations; i++)
    {
#ifdef SCALAR
        noise(x0, y0, x1, y1, width, height, buf);
#endif

#ifdef CPPSPMD
        spmd_call<noise>(x0, y0, x1, y1, width, height, buf);
#endif // CPPSPMD

#ifdef ISPC
        ispc::noise(x0, y0, x1, y1, width, height, buf);
#endif // ISPC
    }

    writePPM(buf, width, height, "noise.ppm");
}
#endif // NOISE
