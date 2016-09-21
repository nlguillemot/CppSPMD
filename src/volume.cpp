#include "common.h"

#include "spmd_avx2-i32x8.h"

#ifdef SCALAR
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
volume(float density[], int nVoxels[3], const float raster2camera[4][4],
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
#endif // SCALAR

#ifdef CPPSPMD
#ifdef _MSC_VER
__declspec(align(16))
#endif
struct float3 {
    float3() = default;
    float3(float xx, float yy, float zz)
        : x(xx), y(yy), z(zz)
    { }

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

    vfloat3(const float3& f)
        : x(f.x), y(f.y), z(f.z)
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
            spmd_return();
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
            spmd_return();
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
        return d.x * d.x + d.y * d.y + d.z * d.z;
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
            spmd_return();
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
struct volume_tile : public spmd_kernel
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
                    spmd_call<generateRay>(raster2camera, camera2world,
                                           vfloat(xo), vfloat(yo), ray);

                    // And raymarch through the volume to compute the pixel's
                    // value
                    vint offset = yo * width + xo;
                    store(offset[image],
                          spmd_call<raymarch>(density, nVoxels, ray));
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
#endif // CPPSPMD

#ifdef ISPC
# include "volume.ispc.h"
#endif // ISPC

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
    float *density = loadVolume("volume_assets/density_highres.vol", n);

    int num_runs = 1;

    start_timer();

    for (int i = 0; i < num_runs; i++)
    {
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

        end_run();
    }

    stop_timer(num_runs);

    writePPM(image, width, height, "volume.ppm");
}
