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
#endif // SCALAR

#ifdef CPPSPMD
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
#endif // CPPSPMD

#ifdef ISPC
# include "simple.ispc.h"
#endif // ISPC

int main()
{
    ALIGN(32) float vin[16], vout[16];
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
