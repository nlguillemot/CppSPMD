
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#include "spmd_avx2-i32x8.h"

#if !defined(WIN32)
# define __stdcall
#endif

#if 0
extern "C" int __stdcall QueryPerformanceCounter(uint64_t* lpPerformanceCount);
extern "C" int __stdcall QueryPerformanceFrequency(uint64_t* lpFrequency);

uint64_t g_start_time;
uint64_t g_total_time;
#endif

inline void start_timer()
{
#if 0
    QueryPerformanceCounter(&g_start_time);
    g_total_time = 0;
#endif
}

inline void end_run()
{
#if 0
    uint64_t end_time;
    QueryPerformanceCounter(&end_time);
    g_total_time += end_time - g_start_time;
    QueryPerformanceCounter(&g_start_time);
#endif
}

inline void stop_timer(int num_runs)
{
#if 0
    uint64_t freq;
    QueryPerformanceFrequency(&freq);
    double sec = double(g_total_time) / freq;
    printf("%d runs in %.3lf seconds, %.3lf seconds per run\n", num_runs, sec, sec / num_runs);
#endif
}

/* Write a PPM image file with the image of the Mandelbrot set */
inline void writePPM(int *buf, int width, int height, const char *fn) {
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

/* Write a PPM image file with the image */
inline void writePPM(float *buf, int width, int height, const char *fn) {
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
