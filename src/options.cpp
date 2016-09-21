#include "common.h"

#include "spmd_avx2-i32x8.h"

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

#define BINOMIAL_NUM 64

// Cumulative normal distribution function
static inline float
CND(float X) {
    float L = fabsf(X);

    float k = 1.f / (1.f + 0.2316419f * L);
    float k2 = k*k;
    float k3 = k2*k;
    float k4 = k2*k2;
    float k5 = k3*k2;

    const float invSqrt2Pi = 0.39894228040f;
    float w = (0.31938153f * k - 0.356563782f * k2 + 1.781477937f * k3 +
               -1.821255978f * k4 + 1.330274429f * k5);
    w *= invSqrt2Pi * expf(-L * L * .5f);

    if (X > 0.f)
        w = 1.f - w;
    return w;
}


void
black_scholes(float Sa[], float Xa[], float Ta[],
              float ra[], float va[],
              float result[], int count) {
    for (int i = 0; i < count; ++i) {
        float S = Sa[i], X = Xa[i];
        float T = Ta[i], r = ra[i];
        float v = va[i];

        float d1 = (logf(S/X) + (r + v * v * .5f) * T) / (v * sqrtf(T));
        float d2 = d1 - v * sqrtf(T);

        result[i] = S * CND(d1) - X * expf(-r * T) * CND(d2);
    }
}


void
binomial_put(float Sa[], float Xa[], float Ta[],
             float ra[], float va[],
             float result[], int count) {
    float V[BINOMIAL_NUM];

    for (int i = 0; i < count; ++i) {
        float S = Sa[i], X = Xa[i];
        float T = Ta[i], r = ra[i];
        float v = va[i];

        float dt = T / BINOMIAL_NUM;
        float u = expf(v * sqrtf(dt));
        float d = 1.f / u;
        float disc = expf(r * dt);
        float Pu = (disc - d) / (u - d);

        for (int j = 0; j < BINOMIAL_NUM; ++j) {
            float upow = powf(u, (float)(2*j-BINOMIAL_NUM));
            V[j] = std::max(0.f, X - S * upow);
        }

        for (int j = BINOMIAL_NUM-1; j >= 0; --j)
            for (int k = 0; k < j; ++k)
                V[k] = ((1 - Pu) * V[k] + Pu * V[k + 1]) / disc;

        result[i] = V[0];
    }
}

#endif // SCALAR

#ifdef CPPSPMD
#define BINOMIAL_NUM 64

// Cumulative normal distribution function
struct CND : spmd_kernel
{
    vfloat _call(const vfloat& X) {
        vfloat L = abs(X);

        vfloat k = 1.0f / (1.0f + 0.2316419f * L);
        vfloat k2 = k*k;
        vfloat k3 = k2*k;
        vfloat k4 = k2*k2;
        vfloat k5 = k3*k2;

        const float invSqrt2Pi = 0.39894228040f;
        vfloat w = (0.31938153f * k - 0.356563782f * k2 + 1.781477937f * k3 +
                    -1.821255978f * k4 + 1.330274429f * k5)
            * invSqrt2Pi * exp(-L * L * .5f);

        spmd_if(X > 0.f, [&] {
            store(w, 1.0f - w);
        });

        return w;
    }
};

struct black_scholes : spmd_kernel
{
    void _call(float Sa[], float Xa[], float Ta[],
               float ra[], float va[],
               float result[], int count) {
        spmd_foreach(0, count, [&](const lint& i) {
            vfloat S = load(i[Sa]), X = load(i[Xa]), T = load(i[Ta]), r = load(i[ra]), v = load(i[va]);

            vfloat d1 = (log(S/X) + (r + v * v * .5f) * T) / (v * sqrt(T));
            vfloat d2 = d1 - v * sqrt(T);

            store(i[result], S * spmd_call<CND>(d1) - X * exp(-r * T) * spmd_call<CND>(d2));
        });
    }
};

struct do_binomial_put : spmd_kernel
{
    vfloat _call(const vfloat& S, const vfloat& X, const vfloat& T, const vfloat& r, const vfloat& v) {
        vfloat V[BINOMIAL_NUM];

        vfloat dt = T / BINOMIAL_NUM;
        vfloat u = exp(v * sqrt(dt));
        vfloat d = 1.0f / u;
        vfloat disc = exp(r * dt);
        vfloat Pu = (disc - d) / (u - d);

        for (int j = 0; j < BINOMIAL_NUM; ++j) {
            vfloat upow = pow(u, (vfloat)(2*j-BINOMIAL_NUM));
            store(j[V], max(0., X - S * upow));
        }

        for (int j = BINOMIAL_NUM-1; j >= 0; --j)
            for (int k = 0; k < j; ++k)
                store(k[V], ((1 - Pu) * V[k] + Pu * V[k + 1] / disc));
        return V[0];
    }
};

struct binomial_put : spmd_kernel
{
    void _call(float Sa[], float Xa[], float Ta[],
               float ra[], float va[],
               float result[], int count) {
        spmd_foreach (0, count, [&](const lint& i) {
            vfloat S = load(i[Sa]), X = load(i[Xa]), T = load(i[Ta]), r = load(i[ra]), v = load(i[va]);
            store(i[result], spmd_call<do_binomial_put>(S, X, T, r, v));
        });
    }
};
#endif // CPPSPMD

#ifdef ISPC
# include "options.ispc.h"
#endif // ISPC

int main(int argc, char *argv[]) {
    int nOptions = 128*1024;

    float *S = new float[nOptions];
    float *X = new float[nOptions];
    float *T = new float[nOptions];
    float *r = new float[nOptions];
    float *v = new float[nOptions];
    float *result = new float[nOptions];

    for (int i = 0; i < nOptions; ++i) {
        S[i] = 100;  // stock price
        X[i] = 98;   // option strike price
        T[i] = 2;    // time (years)
        r[i] = .02;  // risk-free interest rate
        v[i] = 5;    // volatility
    }

    int num_runs = 100;

    start_timer();

    // Binomial options pricing model
    for (int i = 0; i < num_runs; i++)
    {
#ifdef SCALAR
        binomial_put(S, X, T, r, v, result, nOptions);
#endif // SCALAR

#ifdef CPPSPMD
        spmd_call<binomial_put>(S, X, T, r, v, result, nOptions);
#endif // CPPSPMD

#ifdef ISPC
        ispc::binomial_put(S, X, T, r, v, result, nOptions);
#endif // ISPC

        end_run();
    }

    printf("Binomial options:\n");
    stop_timer(num_runs);

    FILE* binomial_csv = fopen("binomial.csv", "w");
    for (int i = 0; binomial_csv && i < nOptions; i++)
    {
        fprintf(binomial_csv, "%f\n", result[i]);
    }
    fclose(binomial_csv);

    num_runs = 1000;

    start_timer();

    // Black-Scholes options pricing model
    for (int i = 0; i < num_runs; i++)
    {
#ifdef SCALAR
        black_scholes(S, X, T, r, v, result, nOptions);
#endif // SCALAR

#ifdef CPPSPMD
        spmd_call<black_scholes>(S, X, T, r, v, result, nOptions);
#endif // CPPSPMD

#ifdef ISPC
        ispc::black_scholes(S, X, T, r, v, result, nOptions);
#endif // ISPC

        end_run();
    }

    printf("Black-Scholes:\n");
    stop_timer(num_runs);

    FILE* black_scholes_csv = fopen("black_scholes.csv", "w");
    for (int i = 0; black_scholes_csv && i < nOptions; i++)
    {
        fprintf(black_scholes_csv, "%f\n", result[i]);
    }
    fclose(black_scholes_csv);
}
