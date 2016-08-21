#pragma once

#include <immintrin.h>

#include <cstdint>
#include <cassert>

struct spmd_kernel
{
    struct exec_t
    {
        __m256 _mask;
    };

    exec_t and(const exec_t& a, const exec_t& b)
    {
        return exec_t{ _mm256_and_ps(a._mask, b._mask) };
    }

    exec_t andnot(const exec_t& a, const exec_t& b)
    {
        return exec_t{ _mm256_andnot_ps(a._mask, b._mask) };
    }

    exec_t exec = exec_t{ _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ) };

    struct vbool
    {
        __m256 _value;

    private:
        // assignment must be masked
        vbool& operator=(const vbool&);
    };

    struct vfloat
    {
        __m256 _value;

        explicit vfloat(const __m256& v)
            : _value(v)
        { }

    private:
        // assignment must be masked
        vfloat& operator=(const vfloat&);
    };

    vfloat mul(const vfloat& a, const vfloat& b)
    {
        return vfloat{ _mm256_mul_ps(a._value, b._value) };
    }

    vfloat sqrt(const vfloat& v)
    {
        return vfloat{ _mm256_sqrt_ps(v._value) };
    }

    vfloat& store(vfloat& dst, const vfloat& src)
    {
        dst._value = _mm256_blendv_ps(dst._value, src._value, exec._mask);
        return dst;
    }

    vbool less(const vfloat& a, float b)
    {
        return vbool{ _mm256_cmp_ps(a._value, _mm256_set1_ps(b), _CMP_LT_OQ) };
    }

    // reference to a vfloat stored linearly in memory
    struct vfloat_ref
    {
        float* _value;

    private:
        // ref-ref assignment must be masked both ways
        vfloat_ref& operator=(const vfloat_ref&);
    };

    vfloat_ref& store(vfloat_ref& dst, const vfloat& src)
    {
        int mask = _mm256_movemask_ps(exec._mask);
        if (mask == 0b11111111)
        {
            // "all on" optimization: vector store
            _mm256_store_ps(dst._value, src._value);
        }
        else
        {
            // masked store
            _mm256_maskstore_ps(dst._value, _mm256_castps_si256(exec._mask), src._value);
        }
        return dst;
    }

    vfloat load(const vfloat_ref& src)
    {
        int mask = _mm256_movemask_ps(exec._mask);
        if (mask == 0b11111111)
        {
            // "all on" optimization: vector load
            return vfloat{ _mm256_load_ps(src._value) };
        }
        else
        {
            // masked load
            return vfloat{ _mm256_maskload_ps(src._value, _mm256_castps_si256(exec._mask)) };
        }
    }

    struct lint
    {
        __m256i _value;

        explicit lint(__m256i value)
            : _value(value)
        { }

        vfloat_ref operator[](float* ptr) const
        {
            return vfloat_ref{ ptr + _mm_cvtsi128_si32(_mm256_extracti128_si256(_value, 0)) };
        }

    private:
        // masked assignment can produce a non-linear value
        lint& operator=(const lint&);
    };

    const lint programIndex = lint{ _mm256_set_epi32(7,6,5,4,3,2,1,0) };
    const int programCount = 8;

    template<class IfBody>
    void spmd_if(const vbool& cond, const IfBody& ifBody)
    {
        // save old execution mask
        exec_t old_exec = exec;

        // apply "if" mask
        exec = and(exec, exec_t{ cond._value });

        // "all off" optimization
        int mask = _mm256_movemask_ps(exec._mask);
        if (mask != 0)
        {
            ifBody();
        }

        // restore execution mask
        exec = old_exec;
    }

    template<class IfBody, class ElseBody>
    void spmd_ifelse(const vbool& cond, const IfBody& ifBody, const ElseBody& elseBody)
    {
        // save old execution mask
        exec_t old_exec = exec;

        // apply "if" mask
        exec = and(exec, exec_t{ cond._value });

        // "all off" optimization
        int mask = _mm256_movemask_ps(exec._mask);
        if (mask != 0)
        {
            ifBody();
        }

        // invert mask for "else"
        exec = andnot(exec, old_exec);

        // "all off" optimization
        mask = _mm256_movemask_ps(exec._mask);
        if (mask != 0)
        {
            elseBody();
        }

        // restore execution mask
        exec = old_exec;
    }

    template<class ForeachBody>
    void spmd_foreach(int first, int last, const ForeachBody& foreachBody)
    {
        // could allow this, just too lazy right now.
        assert(first <= last);

        // number of loops that don't require extra masking
        int numFullLoops = ((last - first) / programCount) * programCount;
        // number of loops that require extra masking (if loop count is not a multiple of programCount)
        int numPartialLoops = (last - first) % programCount;

        // do every loop that doesn't need to be masked
        __m256i loopIndex = _mm256_add_epi32(programIndex._value, _mm256_set1_epi32(first));
        for (int i = 0; i < numFullLoops; i += programCount)
        {
            foreachBody(lint{ loopIndex });
            loopIndex = _mm256_add_epi32(loopIndex, _mm256_set1_epi32(programCount));
        }

        // do a partial loop if necessary (if loop count is not a multiple of programCount)
        if (numPartialLoops > 0)
        {
            // save old execution mask
            exec_t old_exec = exec;

            // apply mask for partial loop
            exec = and(exec, exec_t{ _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(numPartialLoops), programIndex._value)) });

            // do the partial loop
            foreachBody(lint{ loopIndex });

            // restore execution mask
            exec = old_exec;
        }
    }
};