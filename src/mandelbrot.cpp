#include "common.h"

// Enable hand-written optimizations
#define SPMD_MANDELBROT_OPTIMIZATION

#ifdef SCALAR
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
#endif // SCALAR

#ifdef CPPSPMD
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
#endif // CPPSPMD

#ifdef ISPC
# include "mandelbrot.ispc.h"
#endif // ISPC

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

    int num_runs = 10;

    start_timer();

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

        end_run();
    }

    stop_timer(num_runs);

    writePPM(buf, width, height, "mandelbrot.ppm");
}
