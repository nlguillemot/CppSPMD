#include <cstdio>
#include <cstdlib>

// which version to use
//#define SCALAR
#define CPPSPMD
//#define ISPC

// which test to run
//#define SIMPLE
//#define NOISE
//#define MANDELBROT
#define VOLUME

// Enable hand-written optimizations
#define SPMD_NOISE_OPTIMIZATION
#define SPMD_MANDELBROT_OPTIMIZATION

#ifdef SCALAR
#include <cmath>

#ifdef SIMPLE
/*
  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
*/

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
/*
  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
*/

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

#ifdef MANDELBROT
/*
  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
*/


static int mandel(float c_re, float c_im, int count) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

void mandelbrot(float x0, float y0, float x1, float y1,
                int width, int height, int maxIterations,
                int output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; ++i) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = (j * width + i);
            output[index] = mandel(x, y, maxIterations);
        }
    }
}
#endif // MANDELBROT

#ifdef VOLUME
/*
  Copyright (c) 2011, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
*/

#include <assert.h>
#include <math.h>
#include <algorithm>

// Just enough of a float3 class to do what we need in this file.
struct float3 {
    float3() { }
    float3(float xx, float yy, float zz) { x = xx; y = yy; z = zz; }

    float3 operator*(float f) const { return float3(x*f, y*f, z*f); }
    float3 operator-(const float3 &f2) const { 
        return float3(x-f2.x, y-f2.y, z-f2.z); 
    }
    float3 operator*(const float3 &f2) const { 
        return float3(x*f2.x, y*f2.y, z*f2.z); 
    }
    float3 operator+(const float3 &f2) const { 
        return float3(x+f2.x, y+f2.y, z+f2.z); 
    }
    float3 operator/(const float3 &f2) const { 
        return float3(x/f2.x, y/f2.y, z/f2.z); 
    }
    float operator[](int i) const { return (&x)[i]; }
    float &operator[](int i) { return (&x)[i]; }

    float x, y, z;
    float pad;  // match padding/alignment of ispc version 
}
#ifndef _MSC_VER
__attribute__ ((aligned(16)))
#endif
;

struct Ray {
    float3 origin, dir;
};


static void
generateRay(const float raster2camera[4][4], const float camera2world[4][4],
            float x, float y, Ray &ray) {
    // transform raster coordinate (x, y, 0) to camera space
    float camx = raster2camera[0][0] * x + raster2camera[0][1] * y + raster2camera[0][3];
    float camy = raster2camera[1][0] * x + raster2camera[1][1] * y + raster2camera[1][3];
    float camz = raster2camera[2][3];
    float camw = raster2camera[3][3];
    camx /= camw;
    camy /= camw;
    camz /= camw;

    ray.dir.x = camera2world[0][0] * camx + camera2world[0][1] * camy + camera2world[0][2] * camz;
    ray.dir.y = camera2world[1][0] * camx + camera2world[1][1] * camy + camera2world[1][2] * camz;
    ray.dir.z = camera2world[2][0] * camx + camera2world[2][1] * camy + camera2world[2][2] * camz;

    ray.origin.x = camera2world[0][3] / camera2world[3][3];
    ray.origin.y = camera2world[1][3] / camera2world[3][3];
    ray.origin.z = camera2world[2][3] / camera2world[3][3];
}


static bool
Inside(float3 p, float3 pMin, float3 pMax) {
    return (p.x >= pMin.x && p.x <= pMax.x &&
            p.y >= pMin.y && p.y <= pMax.y &&
            p.z >= pMin.z && p.z <= pMax.z);
}


static bool
IntersectP(const Ray &ray, float3 pMin, float3 pMax, float *hit0, float *hit1) {
    float t0 = -1e30f, t1 = 1e30f;

    float3 tNear = (pMin - ray.origin) / ray.dir;
    float3 tFar  = (pMax - ray.origin) / ray.dir;
    if (tNear.x > tFar.x) {
        float tmp = tNear.x;
        tNear.x = tFar.x;
        tFar.x = tmp;
    }
    t0 = std::max(tNear.x, t0);
    t1 = std::min(tFar.x, t1);

    if (tNear.y > tFar.y) {
        float tmp = tNear.y;
        tNear.y = tFar.y;
        tFar.y = tmp;
    }
    t0 = std::max(tNear.y, t0);
    t1 = std::min(tFar.y, t1);

    if (tNear.z > tFar.z) {
        float tmp = tNear.z;
        tNear.z = tFar.z;
        tFar.z = tmp;
    }
    t0 = std::max(tNear.z, t0);
    t1 = std::min(tFar.z, t1);
    
    if (t0 <= t1) {
        *hit0 = t0;
        *hit1 = t1;
        return true;
    }
    else
        return false;
}


static inline float Lerp(float t, float a, float b) {
    return (1.f - t) * a + t * b;
}


static inline int Clamp(int v, int low, int high) {
    return std::min(std::max(v, low), high);
}


static inline float D(int x, int y, int z, int nVoxels[3], float density[]) {
    x = Clamp(x, 0, nVoxels[0]-1);
    y = Clamp(y, 0, nVoxels[1]-1);
    z = Clamp(z, 0, nVoxels[2]-1);
    return density[z*nVoxels[0]*nVoxels[1] + y*nVoxels[0] + x];
}


static inline float3 Offset(float3 p, float3 pMin, float3 pMax) {
    return float3((p.x - pMin.x) / (pMax.x - pMin.x),
                  (p.y - pMin.y) / (pMax.y - pMin.y),
                  (p.z - pMin.z) / (pMax.z - pMin.z));
}


static inline float Density(float3 Pobj, float3 pMin, float3 pMax, 
                            float density[], int nVoxels[3]) {
    if (!Inside(Pobj, pMin, pMax)) 
        return 0;
    // Compute voxel coordinates and offsets for _Pobj_
    float3 vox = Offset(Pobj, pMin, pMax);
    vox.x = vox.x * nVoxels[0] - .5f;
    vox.y = vox.y * nVoxels[1] - .5f;
    vox.z = vox.z * nVoxels[2] - .5f;
    int vx = (int)(vox.x), vy = (int)(vox.y), vz = (int)(vox.z);
    float dx = vox.x - vx, dy = vox.y - vy, dz = vox.z - vz;

    // Trilinearly interpolate density values to compute local density
    float d00 = Lerp(dx, D(vx, vy, vz, nVoxels, density),     
                         D(vx+1, vy, vz, nVoxels, density));
    float d10 = Lerp(dx, D(vx, vy+1, vz, nVoxels, density),   
                         D(vx+1, vy+1, vz, nVoxels, density));
    float d01 = Lerp(dx, D(vx, vy, vz+1, nVoxels, density),   
                         D(vx+1, vy, vz+1, nVoxels, density));
    float d11 = Lerp(dx, D(vx, vy+1, vz+1, nVoxels, density), 
                         D(vx+1, vy+1, vz+1, nVoxels, density));
    float d0 = Lerp(dy, d00, d10);
    float d1 = Lerp(dy, d01, d11);
    return Lerp(dz, d0, d1);
}



static float
transmittance(float3 p0, float3 p1, float3 pMin,
              float3 pMax, float sigma_t, float density[], int nVoxels[3]) {
    float rayT0, rayT1;
    Ray ray;
    ray.origin = p1;
    ray.dir = p0 - p1;

    // Find the parametric t range along the ray that is inside the volume.
    if (!IntersectP(ray, pMin, pMax, &rayT0, &rayT1))
        return 1.;

    rayT0 = std::max(rayT0, 0.f);

    // Accumulate beam transmittance in tau
    float tau = 0;
    float rayLength = sqrtf(ray.dir.x * ray.dir.x + ray.dir.y * ray.dir.y +
                            ray.dir.z * ray.dir.z);
    float stepDist = 0.2f;
    float stepT = stepDist / rayLength;

    float t = rayT0;
    float3 pos = ray.origin + ray.dir * rayT0;
    float3 dirStep = ray.dir * stepT;
    while (t < rayT1) {
        tau += stepDist * sigma_t * Density(pos, pMin, pMax, density, nVoxels);
        pos = pos + dirStep;
        t += stepT;
    }

    return expf(-tau);
}


static float
distanceSquared(float3 a, float3 b) {
    float3 d = a-b;
    return d.x*d.x + d.y*d.y + d.z*d.z;
}


static float 
raymarch(float density[], int nVoxels[3], const Ray &ray) {
    float rayT0, rayT1;
    float3 pMin(.3f, -.2f, .3f), pMax(1.8f, 2.3f, 1.8f);
    float3 lightPos(-1.f, 4.f, 1.5f);

    if (!IntersectP(ray, pMin, pMax, &rayT0, &rayT1))
        return 0.;

    rayT0 = std::max(rayT0, 0.f);

    // Parameters that define the volume scattering characteristics and
    // sampling rate for raymarching
    float Le = .25f;           // Emission coefficient
    float sigma_a = 10;        // Absorption coefficient
    float sigma_s = 10;        // Scattering coefficient
    float stepDist = 0.025f;   // Ray step amount
    float lightIntensity = 40; // Light source intensity

    float tau = 0.f;  // accumulated beam transmittance
    float L = 0;      // radiance along the ray
    float rayLength = sqrtf(ray.dir.x * ray.dir.x + ray.dir.y * ray.dir.y +
                            ray.dir.z * ray.dir.z);
    float stepT = stepDist / rayLength;

    float t = rayT0;
    float3 pos = ray.origin + ray.dir * rayT0;
    float3 dirStep = ray.dir * stepT;
    while (t < rayT1) {
        float d = Density(pos, pMin, pMax, density, nVoxels);

        // terminate once attenuation is high
        float atten = expf(-tau);
        if (atten < .005f)
            break;

        // direct lighting
        float Li = lightIntensity / distanceSquared(lightPos, pos) * 
            transmittance(lightPos, pos, pMin, pMax, sigma_a + sigma_s,
                          density, nVoxels);
        L += stepDist * atten * d * sigma_s * (Li + Le);

        // update beam transmittance
        tau += stepDist * (sigma_a + sigma_s) * d;

        pos = pos + dirStep;
        t += stepT;
    }

    // Gamma correction
    return powf(L, 1.f / 2.2f);
}


void
volume_serial(float density[], int nVoxels[3], const float raster2camera[4][4],
              const float camera2world[4][4], 
              int width, int height, float image[]) {
    int offset = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++offset) {
            Ray ray;
            generateRay(raster2camera, camera2world, (float)x, (float)y, ray);
            image[offset] = raymarch(density, nVoxels, ray);
        }
    }
}
#endif // VOLUME

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
        return spmd_ternary(vbool(h & 1), -u, u) + spmd_ternary(vbool(h & 2), -v, v);
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

                lint index = j * width + i + programIndex;
                store(index[output], spmd_call<Turbulence>(x, y, 0.6f, 8));
            }
        }
    }
};
#endif // NOISE

#ifdef MANDELBROT
struct mandel : spmd_kernel
{
#ifndef SPMD_MANDELBROT_OPTIMIZATION
    vint _call(const vfloat& c_re, const vfloat& c_im, int count)
#else
    vint _call(const vfloat& c_re, const vfloat& c_im, const vint& count)
#endif
    {
        vfloat z_re = c_re, z_im = c_im;
        vint i_result;
        vint i;
#ifdef SPMD_MANDELBROT_OPTIMIZATION
        spmd_for([&] { i._value = _mm256_setzero_si256();  }, [&] { return i < count; }, [&] { i._value = (i + 1)._value; }, [&] {
#else
        spmd_for([&] { store(i, 0);  }, [&] { return i < count; }, [&] { store(i, i + 1); }, [&] {
#endif

#ifdef SPMD_MANDELBROT_OPTIMIZATION
            spmd_if(fma(z_re, z_re, z_im * z_im) > 4.0f, [&] {
                store(i_result, i);
#else
            spmd_if(z_re * z_re + z_im * z_im > 4.0f, [&] {
#endif
                spmd_break();
            });

#ifdef SPMD_MANDELBROT_OPTIMIZATION
            vfloat new_re = fma(z_re, z_re, - z_im * z_im);
#else
            vfloat new_re = z_re * z_re - z_im * z_im;
#endif
            vfloat new_im = 2.f * z_re * z_im;

#ifdef SPMD_MANDELBROT_OPTIMIZATION
            z_re._value = (c_re + new_re)._value;
            z_im._value = (c_im + new_im)._value;
#else
            spmd_unmasked([&] {
                store(z_re, c_re + new_re);
                store(z_im, c_im + new_im);
            });
#endif
        });

        return i_result;
    }
};

struct mandelbrot : spmd_kernel
{
    void _call(
        float x0, float y0,
        float x1, float y1,
        int width, int height,
        int maxIterations,
        int output[])
    {
        float dx = (x1 - x0) / width;
        float dy = (y1 - y0) / height;

        for (int j = 0; j < height; j++) {
            // Note that we'll be doing programCount computations in parallel,
            // so increment i by that much.  This assumes that width evenly
            // divides programCount.
            spmd_foreach(0, width, [&](const lint& i) {
                // Figure out the position on the complex plane to compute the
                // number of iterations at.  Note that the x values are
                // different across different program instances, since its
                // initializer incorporates the value of the programIndex
                // variable.
                vfloat x = x0 + i * dx;
                vfloat y = y0 + j * dy;

                lint index = j * width + i;
#ifdef SPMD_MANDELBROT_OPTIMIZATION
                *(__m256i*)index[output]._value = spmd_call<mandel>(x, y, maxIterations)._value;
#else
                store(index[output], spmd_call<mandel>(x, y, maxIterations));
#endif
            });
        }
    }
};
#endif // MANDELBROT

#ifdef VOLUME
// Just enough of a float3 class to do what we need in this file.
struct vfloat3
{
    vfloat x;
    vfloat y;
    vfloat z;

    vfloat3() = default;

    vfloat3(const vfloat& xx, const vfloat& yy, const vfloat& zz)
        : x(xx), y(yy), z(zz)
    { }

    vfloat3 operator*(const vfloat& f) const {
        return vfloat3(x * f, y * f, z * f); 
    }
    vfloat3 operator-(const vfloat3& f2) const {
        return vfloat3(x - f2.x, y - f2.y, z - f2.z);
    }
    vfloat3 operator*(const vfloat3& f2) const {
        return vfloat3(x * f2.x, y * f2.y, z * f2.z);
    }
    vfloat3 operator+(const vfloat3& f2) const {
        return vfloat3(x + f2.x, y + f2.y, z + f2.z);
    }
    vfloat3 operator/(const vfloat3& f2) const {
        return vfloat3(x / f2.x, y / f2.y, z / f2.z);
    }
    const vfloat& operator[](int i) const { return (&x)[i]; }
    vfloat& operator[](int i) { return (&x)[i]; }
};

#ifdef _MSC_VER
__declspec(align(16))
#endif
struct float3 {
    float3() = default;
    float3(float xx, float yy, float zz)
        : x(xx), y(yy), z(zz)
    { }

    operator vfloat3() const
    {
        return vfloat3(x, y, z);
    }

    float3 operator*(float f) const {
        return float3(x*f, y*f, z*f); 
    }
    float3 operator-(const float3 &f2) const {
        return float3(x - f2.x, y - f2.y, z - f2.z);
    }
    float3 operator*(const float3 &f2) const {
        return float3(x*f2.x, y*f2.y, z*f2.z);
    }
    float3 operator+(const float3 &f2) const {
        return float3(x + f2.x, y + f2.y, z + f2.z);
    }
    float3 operator/(const float3 &f2) const {
        return float3(x / f2.x, y / f2.y, z / f2.z);
    }
    const float& operator[](int i) const { return (&x)[i]; }
    float& operator[](int i) { return (&x)[i]; }

    float x, y, z;
    float pad;  // match padding/alignment of ispc version 
}
#ifndef _MSC_VER
__attribute__((aligned(16)))
#endif
;

template<class T>
struct vfloat3_mixin : T
{
    using T::store;

    vfloat3& store(vfloat3& dst, const vfloat3& src)
    {
        store(dst.x, src.x);
        store(dst.y, src.y);
        store(dst.z, src.z);
        return dst;
    }
};

struct vRay
{
    vfloat3 origin, dir;
};

struct generateRay : spmd_kernel
{
    void _call(const float raster2camera[4][4], 
               const float camera2world[4][4],
               const vfloat& x, const vfloat& y, vRay &ray) {
        // transform raster coordinate (x, y, 0) to camera space
        vfloat camw = raster2camera[3][3];
        vfloat camx = (raster2camera[0][0] * x + raster2camera[0][1] * y + raster2camera[0][3]) / camw;
        vfloat camy = (raster2camera[1][0] * x + raster2camera[1][1] * y + raster2camera[1][3]) / camw;
        vfloat camz = raster2camera[2][3] / camw;

        store(ray.dir.x, camera2world[0][0] * camx + camera2world[0][1] * camy + camera2world[0][2] * camz);
        store(ray.dir.y, camera2world[1][0] * camx + camera2world[1][1] * camy + camera2world[1][2] * camz);
        store(ray.dir.z, camera2world[2][0] * camx + camera2world[2][1] * camy + camera2world[2][2] * camz);

        store(ray.origin.x, camera2world[0][3] / camera2world[3][3]);
        store(ray.origin.y, camera2world[1][3] / camera2world[3][3]);
        store(ray.origin.z, camera2world[2][3] / camera2world[3][3]);
    }
};

struct Inside : spmd_kernel
{
    vbool _call(const vfloat3& p, const vfloat3& pMin, const vfloat3& pMax) {
        return (p.x >= pMin.x && p.x <= pMax.x &&
                p.y >= pMin.y && p.y <= pMax.y &&
                p.z >= pMin.z && p.z <= pMax.z);
    }
};

struct IntersectP : spmd_kernel
{
    vbool _call(const vRay& ray, const vfloat3& pMin, const vfloat3& pMax, vfloat &hit0, vfloat &hit1) {
        vfloat t0 = -1e30f, t1 = 1e30f;

        vfloat3 tNear = (pMin - ray.origin) / ray.dir;
        vfloat3 tFar  = (pMax - ray.origin) / ray.dir;
        spmd_if(tNear.x > tFar.x, [&] {
            vfloat tmp = tNear.x;
            store(tNear.x, tFar.x);
            store(tFar.x, tmp);
        });
        store(t0, max(tNear.x, t0));
        store(t1, min(tFar.x, t1));

        spmd_if(tNear.y > tFar.y, [&] {
            vfloat tmp = tNear.y;
            store(tNear.y, tFar.y);
            store(tFar.y, tmp);
        });

        store(t0, max(tNear.y, t0));
        store(t1, min(tFar.y, t1));

        spmd_if (tNear.z > tFar.z, [&] {
            vfloat tmp = tNear.z;
            store(tNear.z, tFar.z);
            store(tFar.z, tmp);
        });
        store(t0, max(tNear.z, t0));
        store(t1, min(tFar.z, t1));
    
        vbool result = t0 <= t1;
        spmd_if(result, [&] {
            store(hit0, t0);
            store(hit1, t1);
        });
        
        return result;
    }
};

struct Lerp : spmd_kernel
{
    vfloat _call(const vfloat& t, const vfloat& a, const vfloat& b) {
        return (1.f - t) * a + t * b;
    }
};

struct D : spmd_kernel
{
    vfloat _call(const vint& x, const vint& y, const vint& z, int nVoxels[3], 
                 float density[]) {
        vint xx = clamp(x, 0, nVoxels[0]-1);
        vint yy = clamp(y, 0, nVoxels[1]-1);
        vint zz = clamp(z, 0, nVoxels[2]-1);

        return load((zz * nVoxels[0] * nVoxels[1] + yy * nVoxels[0] + xx)[density]);
    }
};

struct Offset : spmd_kernel
{
    vfloat3 _call(const vfloat3& p, const vfloat3& pMin, const vfloat3& pMax) {
        return (p - pMin) / (pMax - pMin);
    }
};

struct Density : spmd_kernel
{
    vfloat _call(const vfloat3& Pobj, const vfloat3& pMin, const vfloat3& pMax, 
                 float density[], int nVoxels[3]) {
        
        vfloat result;
        spmd_if(!spmd_call<Inside>(Pobj, pMin, pMax), [&] {
            store(result, 0);
        });

        if (!any(exec))
            return result;

        // Compute voxel coordinates and offsets for _Pobj_
        vfloat3 vox = spmd_call<Offset>(Pobj, pMin, pMax);
        store(vox.x, vox.x * nVoxels[0] - .5f);
        store(vox.y, vox.y * nVoxels[1] - .5f);
        store(vox.z, vox.z * nVoxels[2] - .5f);
        vint vx = (vint)(vox.x), vy = (vint)(vox.y), vz = (vint)(vox.z);
        vfloat dx = vox.x - vx, dy = vox.y - vy, dz = vox.z - vz;

        // Trilinearly interpolate density values to compute local density
        vfloat d00 = spmd_call<Lerp>(dx, spmd_call<D>(vx, vy, vz, nVoxels, density),
                                         spmd_call<D>(vx+1, vy, vz, nVoxels, density));
        vfloat d10 = spmd_call<Lerp>(dx, spmd_call<D>(vx, vy+1, vz, nVoxels, density),
                                         spmd_call<D>(vx+1, vy+1, vz, nVoxels, density));
        vfloat d01 = spmd_call<Lerp>(dx, spmd_call<D>(vx, vy, vz+1, nVoxels, density),
                                         spmd_call<D>(vx+1, vy, vz+1, nVoxels, density));
        vfloat d11 = spmd_call<Lerp>(dx, spmd_call<D>(vx, vy+1, vz+1, nVoxels, density),
                                         spmd_call<D>(vx+1, vy+1, vz+1, nVoxels, density));
        vfloat d0 = spmd_call<Lerp>(dy, d00, d10);
        vfloat d1 = spmd_call<Lerp>(dy, d01, d11);

        store(result, spmd_call<Lerp>(dz, d0, d1));
        return result;
    }
};

struct transmittance : vfloat3_mixin<spmd_kernel>
{
    vfloat _call(const float3& p0, const vfloat3& p1, const float3& pMin,
                 const float3& pMax, float sigma_t, 
                 float density[], int nVoxels[3]) {
        vfloat rayT0, rayT1;
        vRay ray{ p1, vfloat3(p0) - p1 };

        vfloat result;

        // Find the parametric t range along the ray that is inside the volume.
        spmd_if(!spmd_call<IntersectP>(ray, pMin, pMax, rayT0, rayT1), [&] {
            store(result, 1.0f);
        });

        if (!any(exec))
            return result;

        store(rayT0, max(rayT0, 0.f));

        // Accumulate beam transmittance in tau
        vfloat tau = 0.0f;
        vfloat rayLength = sqrt(ray.dir.x * ray.dir.x + ray.dir.y * ray.dir.y +
                                ray.dir.z * ray.dir.z);
        float stepDist = 0.2f;
        vfloat stepT = stepDist / rayLength;

        vfloat t = rayT0;
        vfloat3 pos = ray.origin + ray.dir * rayT0;
        vfloat3 dirStep = ray.dir * stepT;
        spmd_while([&] { return t < rayT1; }, [&] {
            store(tau, tau + stepDist * sigma_t * spmd_call<Density>(pos, pMin, pMax, density, nVoxels));
            store(pos, pos + dirStep);
            store(t, t + stepT);
        });

        store(result, exp(-tau));
        return result;
    }
};

struct distanceSquared : spmd_kernel
{
    vfloat _call(const vfloat3& a, const vfloat3& b) {
        vfloat3 d = a - b;
        return d.x*d.x + d.y*d.y + d.z*d.z;
    }
};

struct raymarch : vfloat3_mixin<spmd_kernel>
{
    vfloat _call(float density[], int nVoxels[3], const vRay& ray) {
        vfloat rayT0, rayT1;
        float3 pMin = {.3, -.2, .3}, pMax = {1.8, 2.3, 1.8};
        float3 lightPos = { -1, 4, 1.5 };

        vfloat result;

        spmd_if(!spmd_call<IntersectP>(ray, pMin, pMax, rayT0, rayT1), [&] {
            store(result, 0.0f);
        });
        
        if (!any(exec))
            return result;

        store(rayT0, max(rayT0, 0.f));

        // Parameters that define the volume scattering characteristics and
        // sampling rate for raymarching
        float Le = 0.25f;             // Emission coefficient
        float sigma_a = 10.0f;        // Absorption coefficient
        float sigma_s = 10.0f;        // Scattering coefficient
        float stepDist = 0.025f;      // Ray step amount
        float lightIntensity = 40.0f; // Light source intensity

        vfloat tau = 0.f;  // accumulated beam transmittance
        vfloat L = 0.0f;   // radiance along the ray
        vfloat rayLength = sqrt(ray.dir.x * ray.dir.x + ray.dir.y * ray.dir.y +
                                ray.dir.z * ray.dir.z);
        vfloat stepT = stepDist / rayLength;

        vfloat t = rayT0;
        vfloat3 pos = ray.origin + ray.dir * rayT0;
        vfloat3 dirStep = ray.dir * stepT;
        spmd_while([&] { return t < rayT1; }, [&] {
            vfloat d = spmd_call<Density>(pos, pMin, pMax, density, nVoxels);

            // terminate once attenuation is high
            vfloat atten = exp(-tau);
            spmd_if(atten < 0.005f, [&] {
                spmd_break();
            });

            if (!any(exec))
                return;

            // direct lighting
            vfloat Li = lightIntensity / spmd_call<distanceSquared>(lightPos, pos) * 
                spmd_call<transmittance>(lightPos, pos, pMin, pMax, sigma_a + sigma_s,
                                         density, nVoxels);
            store(L, L + stepDist * atten * d * sigma_s * (Li + Le));

            // update beam transmittance
            store(tau, tau + stepDist * (sigma_a + sigma_s) * d);

            store(pos, pos + dirStep);
            store(t, t + stepT);
        });

        // Gamma correction
        store(result, pow(L, 1.f / 2.2f));
        return result;
    }
};

/* Utility routine used by both the task-based and the single-core entrypoints.
   Renders a tile of the image, covering [x0,x0) * [y0, y1), storing the
   result into the image[] array.
 */
struct volume_tile : spmd_kernel
{
    /* Utility routine used by both the task-based and the single-core entrypoints.
       Renders a tile of the image, covering [x0,x0) * [y0, y1), storing the
       result into the image[] array.
     */
    void _call(int x0, int y0, int x1,
         int y1, float density[], int nVoxels[3], 
         const float raster2camera[4][4],
         const float camera2world[4][4], 
         int width, int height, float image[]) {
        // Work on 4x4=16 pixel big tiles of the image.  This function thus
        // implicitly assumes that both (x1-x0) and (y1-y0) are evenly divisble
        // by 4.
        for (int y = y0; y < y1; y += 4) {
            for (int x = x0; x < x1; x += 4) {
                spmd_foreach(0, 16, [&](const lint& o) {
                    // These two arrays encode the mapping from [0,15] to
                    // offsets within the 4x4 pixel block so that we render
                    // each pixel inside the block
                    const int xoffsets[16] = { 0, 1, 0, 1, 2, 3, 2, 3,
                                               0, 1, 0, 1, 2, 3, 2, 3 };
                    const int yoffsets[16] = { 0, 0, 1, 1, 0, 0, 1, 1,
                                               2, 2, 3, 3, 2, 2, 3, 3 };

                    // Figure out the pixel to render for this program instance
                    vint xo = x + load(o[xoffsets]), yo = y + load(o[yoffsets]);

                    // Use viewing parameters to compute the corresponding ray
                    // for the pixel
                    vRay ray;
                    spmd_call<generateRay>(raster2camera, camera2world, vfloat(xo), vfloat(yo), ray);

                    // And raymarch through the volume to compute the pixel's
                    // value
                    vint offset = yo * width + xo;
                    store(offset[image], spmd_call<raymarch>(density, nVoxels, ray));
                });
            }
        }
    }
};

struct volume : spmd_kernel
{
    void _call(float density[], int nVoxels[3], 
                const float raster2camera[4][4],
                const float camera2world[4][4], 
                int width, int height, float image[]) {
        spmd_call<volume_tile>(0, 0, width, height, density, nVoxels, raster2camera, 
                               camera2world, width, height,  image);
    }
};
#endif // VOLUME

#endif // CPPSPMD

#ifdef ISPC

#ifdef SIMPLE
#include "simple.ispc.h"
#endif // SIMPLE

#ifdef NOISE
#include "noise.ispc.h"
#endif // NOISE

#ifdef MANDELBROT
#include "mandelbrot.ispc.h"
#endif // MANDELBROT

#ifdef VOLUME
#include "volume.ispc.h"
#endif // VOLUME

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

    int num_runs = 1;

    for (int i = 0; i < num_runs; i++)
    {
#ifdef SCALAR
        noise(x0, y0, x1, y1, width, height, buf);
#endif // SCALAR

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

#ifdef MANDELBROT
/* Write a PPM image file with the image of the Mandelbrot set */
static void
writePPM(int *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width*height; ++i) {
        // Map the iteration count to colors by just alternating between
        // two greys.
        char c = (buf[i] & 0x1) ? 240 : 20;
        for (int j = 0; j < 3; ++j)
            fputc(c, fp);
    }
    fclose(fp);
    printf("Wrote image file %s\n", fn);
}

int main()
{
    unsigned int width = 768;
    unsigned int height = 512;
    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;

    int maxIterations = 256;
    int *buf = new int[width*height];

    int num_runs = 1;

    for (int i = 0; i < num_runs; i++)
    {
#ifdef SCALAR
        mandelbrot(x0, y0, x1, y1, width, height, maxIterations, buf);
#endif // SCALAR

#ifdef CPPSPMD
        spmd_call<mandelbrot>(x0, y0, x1, y1, width, height, maxIterations, buf);
#endif // CPPSPMD

#ifdef ISPC
        ispc::mandelbrot(x0, y0, x1, y1, width, height, maxIterations, buf);
#endif // ISPC
    }

    writePPM(buf, width, height, "mandelbrot.ppm");
}
#endif // MANDELBROT

#ifdef VOLUME
/* Write a PPM image file with the image */
static void
writePPM(float *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width*height; ++i) {
        float v = buf[i] * 255.f;
        if (v < 0.f) v = 0.f;
        else if (v > 255.f) v = 255.f;
        unsigned char c = (unsigned char)v;
        for (int j = 0; j < 3; ++j)
            fputc(c, fp);
    }
    fclose(fp);
    printf("Wrote image file %s\n", fn);
}

/* Load image and viewing parameters from a camera data file.
   FIXME: we should add support to be able to specify viewing parameters
   in the program here directly. */
static void
loadCamera(const char *fn, int *width, int *height, float raster2camera[4][4],
           float camera2world[4][4]) {
    FILE *f = fopen(fn, "r");
    if (!f) {
        perror(fn);
        exit(1);
    }
    if (fscanf(f, "%d %d", width, height) != 2) {
        fprintf(stderr, "Unexpected end of file in camera file\n");
        exit(1);
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (fscanf(f, "%f", &raster2camera[i][j]) != 1) {
                fprintf(stderr, "Unexpected end of file in camera file\n");
                exit(1);
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (fscanf(f, "%f", &camera2world[i][j]) != 1) {
                fprintf(stderr, "Unexpected end of file in camera file\n");
                exit(1);
            }
        }
    }
    fclose(f);
}


/* Load a volume density file.  Expects the number of x, y, and z samples
   as the first three values (as integer strings), then x*y*z
   floating-point values (also as strings) to give the densities.  */
static float *
loadVolume(const char *fn, int n[3]) {
    FILE *f = fopen(fn, "r");
    if (!f) {
        perror(fn);
        exit(1);
    }

    if (fscanf(f, "%d %d %d", &n[0], &n[1], &n[2]) != 3) {
        fprintf(stderr, "Couldn't find resolution at start of density file\n");
        exit(1);
    }

    int count = n[0] * n[1] * n[2];
    float *v = new float[count];
    for (int i = 0; i < count; ++i) {
        if (fscanf(f, "%f", &v[i]) != 1) {
            fprintf(stderr, "Unexpected end of file at %d'th density value\n", i);
            exit(1);
        }
    }

    return v;
}


int main(int argc, char *argv[]) {
    // Load viewing data and the volume density data
    int width, height;
    float raster2camera[4][4], camera2world[4][4];
    loadCamera("volume_assets/camera.dat", &width, &height, raster2camera, camera2world);
    float *image = new float[width*height];

    int n[3];
    float *density = loadVolume("volume_assets/density_lowres.vol", n);

#ifdef SCALAR
    volume(density, n, raster2camera, camera2world,
           width, height, image);
#endif // SCALAR

#ifdef CPPSPMD
    spmd_call<volume>(density, n, raster2camera, camera2world,
                      width, height, image);
#endif // CPPSPMD

#ifdef ISPC
    ispc::volume(density, n, raster2camera, camera2world,
                 width, height, image);
#endif // ISPC

    writePPM(image, width, height, "volume.ppm");
}
#endif // VOLUME
