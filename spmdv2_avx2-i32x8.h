#pragma once

#include <immintrin.h>

#include <cstdint>
#include <cassert>
#include <utility>

struct spmd_kernel
{
    struct vint;

    struct exec_t
    {
        __m256 _mask;
    };

    exec_t exec = exec_t{ _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ) };

    struct vbool
    {
        __m256 _value;

    private:
        // assignment must be masked
        vbool& operator=(const vbool&);
    };

    vbool& store(vbool& dst, const vbool& src)
    {
        dst._value = _mm256_blendv_ps(dst._value, src._value, exec._mask);
        return dst;
    }

    struct vfloat
    {
        __m256 _value;

        vfloat()
            : _value(_mm256_setzero_ps())
        { }

        explicit vfloat(const __m256& v)
            : _value(v)
        { }

        vfloat(float value)
            : _value(_mm256_set1_ps(value))
        { }

        vfloat(int value)
            : _value(_mm256_set1_ps((float)value))
        { }

    private:
        // assignment must be masked
        vfloat& operator=(const vfloat&);
    };

    vfloat& store(vfloat& dst, const vfloat& src)
    {
        dst._value = _mm256_blendv_ps(dst._value, src._value, exec._mask);
        return dst;
    }

    // reference to a vfloat stored linearly in memory
    struct vfloat_lref
    {
        float* _value;

    private:
        // ref-ref assignment must be masked both ways
        vfloat_lref& operator=(const vfloat_lref&);
    };

    // scatter
    vfloat_lref& store(vfloat_lref& dst, const vfloat& src)
    {
        int mask = _mm256_movemask_ps(exec._mask);
        if (mask == 0b11111111)
        {
            // "all on" optimization: vector store
            _mm256_storeu_ps(dst._value, src._value);
        }
        else
        {
            // masked store
            _mm256_maskstore_ps(dst._value, _mm256_castps_si256(exec._mask), src._value);
        }
        return dst;
    }

    // gather
    vfloat load(const vfloat_lref& src)
    {
        int mask = _mm256_movemask_ps(exec._mask);
        if (mask == 0b11111111)
        {
            // "all on" optimization: vector load
            return vfloat{ _mm256_loadu_ps(src._value) };
        }
        else
        {
            // masked load
            return vfloat{ _mm256_maskload_ps(src._value, _mm256_castps_si256(exec._mask)) };
        }
    }

    // reference to a vint stored randomly in memory
    struct vint_vref
    {
        int* _value;
        __m256i _vindex;

    private:
        // ref-ref assignment must be masked both ways
        vint_vref& operator=(const vint_vref&);
    };

    struct vint
    {
        __m256i _value;

        vint()
            : _value(_mm256_setzero_si256())
        { }

        explicit vint(const __m256i& value)
            : _value(value)
        { }

        vint(int value)
            : _value(_mm256_set1_epi32(value))
        { }

        vint(const vfloat& other)
            : _value(_mm256_cvtps_epi32(other._value))
        { }

        operator vbool() const
        {
            return vbool{ _mm256_castsi256_ps(
                _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_value, _mm256_setzero_si256()))) };
        }

        operator vfloat() const
        {
            return vfloat{ _mm256_cvtepi32_ps(_value) };
        }

        vint_vref operator[](int* ptr) const
        {
            return vint_vref{ ptr, _value };
        }

    private:
        // assignment must be masked
        vint& operator=(const vint&);
    };

    vint& store(vint& dst, const vint& src)
    {
        dst._value = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dst._value), _mm256_castsi256_ps(src._value), exec._mask));
        return dst;
    }

    // scatter
    vint_vref& store(vint_vref& dst, const vint& src)
    {
        __declspec(align(32)) int vindex[8];
        _mm256_store_si256((__m256i*)vindex, dst._vindex);

        __declspec(align(32)) int stored[8];
        _mm256_store_si256((__m256i*)stored, src._value);

        int mask = _mm256_movemask_ps(exec._mask);
        for (int i = 0; i < 8; i++)
        {
            if (mask & (1 << i))
                dst._value[vindex[i]] = stored[i];
        }
        return dst;
    }

    // gather
    vint load(const vint_vref& src)
    {
        return vint{ _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), src._value, src._vindex, _mm256_castps_si256(exec._mask), 4) };
    }

    struct lint
    {
        __m256i _value;

        explicit lint(__m256i value)
            : _value(value)
        { }

        operator vfloat() const
        {
            return vfloat{ _mm256_cvtepi32_ps(_value) };
        }

        vfloat_lref operator[](float* ptr) const
        {
            return vfloat_lref{ ptr + _mm_cvtsi128_si32(_mm256_extracti128_si256(_value, 0)) };
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
        exec = exec & exec_t{ cond._value };

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
        exec = exec & exec_t{ cond._value };

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
            exec = exec & exec_t{ _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(numPartialLoops), programIndex._value)) };

            // do the partial loop
            foreachBody(lint{ loopIndex });

            // restore execution mask
            exec = old_exec;
        }
    }

    template<class SPMDKernel, class... Args>
    auto spmd_call(Args&&... args)
    {
        SPMDKernel kernel;
        kernel.exec = exec;
        return kernel._call(std::forward<Args>(args)...);
    }
};

spmd_kernel::exec_t operator&(const spmd_kernel::exec_t& a, const spmd_kernel::exec_t& b)
{
    return spmd_kernel::exec_t{ _mm256_and_ps(a._mask, b._mask) };
}

spmd_kernel::exec_t andnot(const spmd_kernel::exec_t& a, const spmd_kernel::exec_t& b)
{
    return spmd_kernel::exec_t{ _mm256_andnot_ps(a._mask, b._mask) };
}

spmd_kernel::vbool operator||(const spmd_kernel::vbool& a, const spmd_kernel::vbool& b)
{
    return spmd_kernel::vbool{ _mm256_or_ps(a._value, b._value) };
}

spmd_kernel::vfloat operator*(const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b)
{
    return spmd_kernel::vfloat{ _mm256_mul_ps(a._value, b._value) };
}

spmd_kernel::vfloat operator/(const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b)
{
    return spmd_kernel::vfloat{ _mm256_div_ps(a._value, b._value) };
}

spmd_kernel::vfloat operator-(const spmd_kernel::vfloat& v)
{
    return spmd_kernel::vfloat{ _mm256_sub_ps(_mm256_xor_ps(v._value, v._value), v._value) };
}

spmd_kernel::vbool operator==(const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b)
{
    return spmd_kernel::vbool{ _mm256_cmp_ps(a._value, b._value, _CMP_EQ_OQ) };
}

spmd_kernel::vbool operator<(const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b)
{
    return spmd_kernel::vbool{ _mm256_cmp_ps(a._value, b._value, _CMP_LT_OQ) };
}

spmd_kernel::vfloat spmd_ternary(const spmd_kernel::vbool& cond, const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b)
{
    return spmd_kernel::vfloat{ _mm256_blendv_ps(b._value, a._value, cond._value) };
}

spmd_kernel::vfloat sqrt(const spmd_kernel::vfloat& v)
{
    return spmd_kernel::vfloat{ _mm256_sqrt_ps(v._value) };
}

spmd_kernel::vfloat abs(const spmd_kernel::vfloat& v)
{
    return spmd_kernel::vfloat{ _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v._value) };
}

spmd_kernel::vfloat floor(const spmd_kernel::vfloat& v)
{
    return spmd_kernel::vfloat{ _mm256_floor_ps(v._value) };
}

spmd_kernel::vfloat clamp(const spmd_kernel::vfloat& v, const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b)
{
    __m256 lomask = _mm256_cmp_ps(v._value, a._value, _CMP_LT_OQ);
    __m256 himask = _mm256_cmp_ps(v._value, b._value, _CMP_GT_OQ);
    __m256 okmask = _mm256_andnot_ps(_mm256_or_ps(lomask, himask), _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ));
    return spmd_kernel::vfloat{ _mm256_or_ps(_mm256_and_ps(okmask, v._value), _mm256_or_ps(_mm256_and_ps(lomask, a._value), _mm256_and_ps(himask, b._value))) };
}

spmd_kernel::lint operator+(int a, const spmd_kernel::lint& b)
{
    return spmd_kernel::lint{ _mm256_add_epi32(_mm256_set1_epi32(a), b._value) };
}

spmd_kernel::vfloat operator+(const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b)
{
    return spmd_kernel::vfloat{ _mm256_add_ps(a._value, b._value) };
}

spmd_kernel::vfloat operator-(const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b)
{
    return spmd_kernel::vfloat{ _mm256_sub_ps(a._value, b._value) };
}

spmd_kernel::vint operator&(const spmd_kernel::vint& a, const spmd_kernel::vint& b)
{
    return spmd_kernel::vint{ _mm256_and_si256(a._value, b._value) };
}

spmd_kernel::vbool operator==(const spmd_kernel::vint& a, const spmd_kernel::vint& b)
{
    return spmd_kernel::vbool{ _mm256_castsi256_ps(_mm256_cmpeq_epi32(a._value, b._value)) };
}

spmd_kernel::vbool operator<(const spmd_kernel::vint& a, const spmd_kernel::vint& b)
{
    return spmd_kernel::vbool{ _mm256_castsi256_ps(_mm256_cmpgt_epi32(b._value, a._value)) };
}

spmd_kernel::vint operator+(const spmd_kernel::vint& a, const spmd_kernel::vint& b)
{
    return spmd_kernel::vint{ _mm256_add_epi32(a._value, b._value) };
}

spmd_kernel::vbool operator==(const spmd_kernel::lint& a, const spmd_kernel::lint& b)
{
    return spmd_kernel::vbool{ _mm256_castsi256_ps(_mm256_cmpeq_epi32(a._value, b._value)) };
}

spmd_kernel::vbool operator<(const spmd_kernel::lint& a, const spmd_kernel::lint& b)
{
    return spmd_kernel::vbool{ _mm256_castsi256_ps(_mm256_cmpgt_epi32(b._value, a._value)) };
}

template<class SPMDKernel, class... Args>
auto spmd_call(Args&&... args)
{
    SPMDKernel kernel;
    return kernel._call(std::forward<Args>(args)...);
}