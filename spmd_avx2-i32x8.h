#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <utility>

#include "avx_mathfun_tweaked.h"

struct spmd_kernel
{
    struct vbool;
    struct vint;

    struct exec_t
    {
        __m256i _mask;

        exec_t() = default;

        explicit exec_t(const __m256i& mask)
            : _mask(mask)
        { }

        explicit exec_t(const vbool& b);

        static exec_t all_on()
        {
            return exec_t{ _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()) };
        }

        static exec_t all_off()
        {
            return exec_t{ _mm256_setzero_si256() };
        }
    };

    // forward declarations of non-member functions
    friend exec_t operator&(const exec_t& a, const exec_t& b);
    friend exec_t operator|(const exec_t& a, const exec_t& b);
    friend exec_t operator^(const exec_t& a, const exec_t& b);
    friend bool any(const exec_t& e);

    // the execution mask at the kernel-level
    // initial value is based on the execution mask of the caller
    // it is also updated spmd_return() is called
    exec_t _kernel_exec;

    // the execution mask at the current point of varying control flow inside a kernel
    // maintains execution mask of things like spmd_if(), spmd_for(), spmd_break()...
    // generally represents the state of the current loop's execution
    exec_t _internal_exec;

    // the OR of all lanes which hit a "spmd_continue" in the current loop
    exec_t _continue_mask;

    // current control flow's execution mask (= _internal_exec & _kernel_exec)
    exec_t exec;

    // this is basically the constructor
    // can't use a real constructor without requiring users
    // to eg. say "using spmd_kernel::spmd_kernel;", which is blah.
    void _init(const exec_t& kernel_exec)
    {
        _kernel_exec = kernel_exec;
        _internal_exec = exec_t::all_on();
        _continue_mask = exec_t::all_off();
        exec = kernel_exec;
    }

    // it is assumed that "false" is 0 (all zeros), and "true" is ~0 (all ones).
    // operations on vbool should maintain this invariant.
    struct vbool
    {
        __m256i _value;

        explicit vbool(const __m256i& value)
            : _value(value)
        { }

        vbool(bool value)
            : _value(_mm256_set1_epi32(value ? ~0 : 0))
        { }

    private:
        // assignment must be masked
        vbool& operator=(const vbool&);
    };

    friend vbool operator!(const vbool& v);

    vbool& store(vbool& dst, const vbool& src)
    {
        dst._value = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dst._value), _mm256_castsi256_ps(src._value), _mm256_castsi256_ps(exec._mask)));
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

        // this basically doesn't work because ints get implicitly converted to float anyways. :/:/
        explicit vfloat(int value)
            : _value(_mm256_set1_ps((float)value))
        { }

    private:
        // assignment must be masked
        vfloat& operator=(const vfloat&);
    };

    vfloat& store(vfloat& dst, const vfloat& src)
    {
        dst._value = _mm256_blendv_ps(dst._value, src._value, _mm256_castsi256_ps(exec._mask));
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
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(exec._mask));
        if (mask == 0b11111111)
        {
            // "all on" optimization: vector store
            _mm256_storeu_ps(dst._value, src._value);
        }
        else
        {
            // masked store
            _mm256_maskstore_ps(dst._value, exec._mask, src._value);
        }
        return dst;
    }

    // gather
    vfloat load(const vfloat_lref& src)
    {
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(exec._mask));
        if (mask == 0b11111111)
        {
            // "all on" optimization: vector load
            return vfloat{ _mm256_loadu_ps(src._value) };
        }
        else
        {
            // masked load
            return vfloat{ _mm256_maskload_ps(src._value, exec._mask) };
        }
    }

    // reference to a vint stored randomly in memory
    struct vfloat_vref
    {
        float* _value;
        __m256i _vindex;

    private:
        // ref-ref assignment must be masked both ways
        vfloat_vref& operator=(const vfloat_vref&);
    };

    // scatter
    vfloat_vref& store(vfloat_vref& dst, const vfloat& src)
    {
        __declspec(align(32)) int vindex[8];
        _mm256_store_si256((__m256i*)vindex, dst._vindex);

        __declspec(align(32)) float stored[8];
        _mm256_store_ps(stored, src._value);

        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(exec._mask));
        for (int i = 0; i < 8; i++)
        {
            if (mask & (1 << i))
                dst._value[vindex[i]] = stored[i];
        }
        return dst;
    }

    // gather
    vfloat load(const vfloat_vref& src)
    {
        return vfloat{ _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src._value, src._vindex, _mm256_castsi256_ps(exec._mask), 4) };
    }

    // reference to a vint stored linearly in memory
    struct vint_lref
    {
        int* _value;

    private:
        // ref-ref assignment must be masked both ways
        vint_lref& operator=(const vint_lref&);
    };

    // scatter
    vint_lref& store(vint_lref& dst, const vint& src)
    {
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(exec._mask));
        if (mask == 0b11111111)
        {
            // "all on" optimization: vector store
            _mm256_storeu_si256((__m256i*)dst._value, src._value);
        }
        else
        {
            // masked store
            _mm256_maskstore_epi32(dst._value, exec._mask, src._value);
        }
        return dst;
    }

    // gather
    vint load(const vint_lref& src)
    {
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(exec._mask));
        if (mask == 0b11111111)
        {
            // "all on" optimization: vector load
            return vint{ _mm256_loadu_si256((__m256i*)src._value) };
        }
        else
        {
            // masked load
            return vint{ _mm256_maskload_epi32(src._value, exec._mask) };
        }
    }

    // reference to a const vint stored linearly in memory
    struct cvint_lref
    {
        const int* _value;

    private:
        // ref-ref assignment must be masked both ways
        cvint_lref& operator=(const cvint_lref&);
    };

    // gather
    vint load(const cvint_lref& src)
    {
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(exec._mask));
        if (mask == 0b11111111)
        {
            // "all on" optimization: vector load
            return vint{ _mm256_loadu_si256((const __m256i*)src._value) };
        }
        else
        {
            // masked load
            return vint{ _mm256_maskload_epi32(src._value, exec._mask) };
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

        explicit vint(float value)
            : _value(_mm256_set1_epi32((int)value))
        { }

        explicit vint(const vfloat& other)
            : _value(_mm256_cvtps_epi32(other._value))
        { }

        explicit operator vbool() const
        {
            return vbool{
                _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_value, _mm256_setzero_si256())) };
        }

        explicit operator vfloat() const
        {
            return vfloat{ _mm256_cvtepi32_ps(_value) };
        }

        vint_vref operator[](int* ptr) const
        {
            return vint_vref{ ptr, _value };
        }

        vfloat_vref operator[](float* ptr) const
        {
            return vfloat_vref{ ptr, _value };
        }

    private:
        // assignment must be masked
        vint& operator=(const vint&);
    };


    vint& store(vint& dst, const vint& src)
    {
        dst._value = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dst._value), _mm256_castsi256_ps(src._value), _mm256_castsi256_ps(exec._mask)));
        return dst;
    }

    // scatter
    vint_vref& store(vint_vref& dst, const vint& src)
    {
        __declspec(align(32)) int vindex[8];
        _mm256_store_si256((__m256i*)vindex, dst._vindex);

        __declspec(align(32)) int stored[8];
        _mm256_store_si256((__m256i*)stored, src._value);

        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(exec._mask));
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
        return vint{ _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), src._value, src._vindex, exec._mask, 4) };
    }

    struct lint
    {
        __m256i _value;

        explicit lint(__m256i value)
            : _value(value)
        { }

        explicit operator vfloat() const
        {
            return vfloat{ _mm256_cvtepi32_ps(_value) };
        }

        explicit operator vint() const
        {
            return vint{ _value };
        }

        vfloat_lref operator[](float* ptr) const
        {
            return vfloat_lref{ ptr + _mm_cvtsi128_si32(_mm256_extracti128_si256(_value, 0)) };
        }

        vint_lref operator[](int* ptr) const
        {
            return vint_lref{ ptr + _mm_cvtsi128_si32(_mm256_extracti128_si256(_value, 0)) };
        }

        cvint_lref operator[](const int* ptr) const
        {
            return cvint_lref{ ptr + _mm_cvtsi128_si32(_mm256_extracti128_si256(_value, 0)) };
        }

    private:
        // masked assignment can produce a non-linear value
        lint& operator=(const lint&);
    };

    const lint programIndex = lint{ _mm256_set_epi32(7,6,5,4,3,2,1,0) };
    static const int programCount = 8;

    template<class IfBody>
    void spmd_if(const vbool& cond, const IfBody& ifBody)
    {
        // save old execution mask
        exec_t old_internal_exec = _internal_exec;

        // get the condition mask and use it to mask the internal control flow
        exec_t cond_exec(cond);
        exec_t pre_if_internal_exec = _internal_exec & cond_exec;
        
        _internal_exec = _internal_exec & cond_exec;
        exec = exec & cond_exec;

        if (any(exec)) // "all off" optimization
        {
            ifBody();
        }

        // propagate any lanes that were shut down inside the if
        // (assuming lanes haven't "come back to life" at the end of an "if")
        // eg: spmd_if(x, [&]{ spmd_break(); });
        _internal_exec = andnot(pre_if_internal_exec ^ _internal_exec, old_internal_exec);
        exec = _kernel_exec & _internal_exec;
    }

    template<class IfBody, class ElseBody>
    void spmd_ifelse(const vbool& cond, const IfBody& ifBody, const ElseBody& elseBody)
    {
        // Simple implementation, going for correctness first.
        // Maybe could be "optimized" by using andnot in between.
        // Also would be interesting to make it easier to chain if/elses,
        // though have to be careful to lazily evaluate the "cond" in "else if(cond)".
        spmd_if(cond, ifBody);
        spmd_if(!cond, elseBody);
    }

    template<class WhileCondBody, class WhileBody>
    void spmd_while(const WhileCondBody& whileCondBody, const WhileBody& whileBody)
    {
        // save old execution mask
        exec_t old_internal_exec = _internal_exec;

        // save the state of the previous loop (assuming there was one)
        // then start fresh for this loop
        exec_t old_continue_mask = _continue_mask;
        _continue_mask = exec_t::all_off();

        // start looping
        for (;;)
        {
            // compound the result of the loop condition into the execution mask
            exec_t cond_exec = exec_t(whileCondBody());
            _internal_exec = _internal_exec & cond_exec;
            exec = exec & cond_exec;

            // if no lanes are active anymore, stop looping
            if (!any(exec))
                break;

            whileBody();

            // reactivate all lanes that hit a "continue"
            _internal_exec = _internal_exec | _continue_mask;
            exec = _internal_exec & _kernel_exec;
            _continue_mask = exec_t::all_off();
        }

        // now that the loop is done,
        // can restore lanes that were "break"'d, or that failed the loop condition.
        _internal_exec = old_internal_exec;
        exec = _internal_exec & _kernel_exec;

        // restore the continue mask of the previous loop in the stack
        _continue_mask = old_continue_mask;
    }

    template<class ForInitBody, class ForCondBody, class ForIncrBody, class ForBody>
    void spmd_for(const ForInitBody& forInitBody, const ForCondBody& forCondBody, const ForIncrBody& forIncrBody, const ForBody& forBody)
    {
        // save old execution mask
        exec_t old_internal_exec = _internal_exec;

        // execute the initialization clause of the loop
        forInitBody();

        // save the state of the previous loop (assuming there was one)
        // then start fresh for this loop
        exec_t old_continue_mask = _continue_mask;
        _continue_mask = exec_t::all_off();

        // start looping
        for (;;)
        {
            // compound the result of the loop condition into the execution mask
            exec_t cond_exec = exec_t(forCondBody());
            _internal_exec = _internal_exec & cond_exec;
            exec = exec & cond_exec;

            // if no lanes are active anymore, stop looping
            if (!any(exec))
                break;

            // evaluate the loop body
            forBody();

            // reactivate all lanes that hit a "continue"
            _internal_exec = _internal_exec | _continue_mask;
            exec = _internal_exec & _kernel_exec;
            _continue_mask = exec_t::all_off();

            // evaluate the loop increment
            forIncrBody();
        }

        // now that the loop is done,
        // can restore lanes that were "break"'d, or that failed the loop condition.
        _internal_exec = old_internal_exec;
        exec = _internal_exec & _kernel_exec;

        // restore the continue mask of the previous loop in the stack
        _continue_mask = old_continue_mask;
    }

    template<class ForeachBody>
    void spmd_foreach(int first, int last, const ForeachBody& foreachBody)
    {
        // could probably allow this with a bit of effort, just too lazy right now.
        assert(first <= last);

        // save old execution mask
        exec_t old_internal_exec = _internal_exec;

        // save the state of the previous loop (assuming there was one)
        // then start fresh for this loop
        exec_t old_continue_mask = _continue_mask;
        _continue_mask = exec_t::all_off();

        // number of loops that don't require loop tail masking
        int num_full_simd_loops = ((last - first) / programCount);

        // number of loops that require loop tail masking
        // happens when the loop count is not a multiple of programCount)
        int num_partial_loops = (last - first) % programCount;

        // do every loop that doesn't need to be tail masked
        lint loopIndex = first + programIndex;
        for (int simd_loop_i = 0; simd_loop_i < num_full_simd_loops; simd_loop_i++)
        {
            // if no lanes are active anymore, stop looping
            if (!any(exec))
                break;

            // invoke the body with the current loop index
            foreachBody(loopIndex);

            // reactivate all lanes that hit a "continue"
            _internal_exec = _internal_exec | _continue_mask;
            exec = _internal_exec & _kernel_exec;
            _continue_mask = exec_t::all_off();
            
            // increment the index for each lane to the next work item
            loopIndex._value = (loopIndex + programCount)._value;
        }

        // do a partial loop if the loop wasn't a multiple of programCount and the loop isn't just a zombie by now
        if (num_partial_loops > 0 && any(exec))
        {
            // apply mask for partial loop
            exec_t partial_loop_mask = exec_t{ _mm256_cmpgt_epi32(_mm256_set1_epi32(num_partial_loops), programIndex._value) };
            _internal_exec = _internal_exec & partial_loop_mask;
            exec = exec & partial_loop_mask;

            // do the partial loop for the extra elements
            foreachBody(loopIndex);
        }

        // now that the loop is done,
        // can restore lanes that were "break"'d, or that failed the loop condition.
        _internal_exec = old_internal_exec;
        exec = _internal_exec & _kernel_exec;

        // restore the continue mask of the previous loop in the stack
        _continue_mask = old_continue_mask;
    }

    void spmd_break()
    {
        // this should only be called inside loops. A check for this would be good.

        // turn off all active lanes so nothing happens after the break.
        // this state will be restored after the current loop is done
        _internal_exec = exec_t::all_off();
        exec = exec_t::all_off();
    }

    void spmd_continue()
    {
        // this should only be called inside loops. A check for this would be good.

        // set currently active lanes as "continue"'d
        _continue_mask = _continue_mask | _internal_exec;

        // turn off all active lanes so nothing happens after the continue
        _internal_exec = exec_t::all_off();
        exec = exec_t::all_off();
    }

    void spmd_return()
    {
        // currently don't support returning *values* non-uniformly.
        // You have to manually accumulate your results before returning,
        // similarly to using an "out" variable,
        // then do a uniform return later.

        // Also, no effort is made to detect that everything is off and do a uniform return.
        // It's not possible to implement other than maybe macro magic, or longjmp/exceptions.
        // Currently depend on the user to implement that optimization where suitable.

        // turn off all active lanes so nothing happens after the return
        _kernel_exec = exec_t::all_off();
        exec = exec_t::all_off();
    }

    template<class UnmaskedBody>
    void spmd_unmasked(const UnmaskedBody& unmaskedBody)
    {
        // beware: since we don't immediately return when all lanes turn off, zombie execution is possible.

        // save old exec mask
        exec_t old_exec = exec;
        exec_t old_kernel_exec = _kernel_exec;
        exec_t old_internal_exec = _internal_exec;
        
        // totally unmask the execution
        _kernel_exec = exec_t::all_on();
        _internal_exec = exec_t::all_on();
        exec = exec_t::all_on();

        // run the unmasked body
        unmaskedBody();

        // restore execution mask with any new masks applied
        // eg: spmd_unmasked([&] {
        //         spmd_if(x, [&] {
        //             spmd_return();
        //         });
        //     });
        _kernel_exec = _kernel_exec & old_kernel_exec;
        _internal_exec = _internal_exec & old_internal_exec;
        exec = exec & old_exec;
    }

    template<class SPMDKernel, class... Args>
    decltype(auto) spmd_call(Args&&... args)
    {
        // pass on the execution mask, so masking works recursively
        SPMDKernel kernel;
        kernel._init(exec);
        return kernel._call(std::forward<Args>(args)...);
    }
};

using exec_t = spmd_kernel::exec_t;
using vbool = spmd_kernel::vbool;
using vfloat = spmd_kernel::vfloat;
using vfloat_lref = spmd_kernel::vfloat_lref;
using vfloat_vref = spmd_kernel::vfloat_vref;
using vint = spmd_kernel::vint;
using vint_lref = spmd_kernel::vint_lref;
using cvint_lref = spmd_kernel::cvint_lref;
using vint_vref = spmd_kernel::vint_vref;
using lint = spmd_kernel::lint;

vbool operator!(const vbool& v)
{
    return vbool{
        _mm256_xor_si256(
            _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()),
            v._value) };
}

exec_t::exec_t(const vbool& b)
{
    _mask = b._value;
}

exec_t operator&(const exec_t& a, const exec_t& b)
{
    return exec_t{ _mm256_and_si256(a._mask, b._mask) };
}

exec_t operator|(const exec_t& a, const exec_t& b)
{
    return exec_t{ _mm256_or_si256(a._mask, b._mask) };
}

exec_t operator^(const exec_t& a, const exec_t& b)
{
    return exec_t{ _mm256_xor_si256(a._mask, b._mask) };
}

bool any(const exec_t& e)
{
    return _mm256_movemask_ps(_mm256_castsi256_ps(e._mask)) != 0;
}

exec_t andnot(const exec_t& a, const exec_t& b)
{
    return exec_t{ _mm256_andnot_si256(a._mask, b._mask) };
}

vbool operator||(const vbool& a, const vbool& b)
{
    return vbool{ _mm256_or_si256(a._value, b._value) };
}

vbool operator&&(const vbool& a, const vbool& b)
{
    return vbool{ _mm256_and_si256(a._value, b._value) };
}

vfloat operator+(const vfloat& a, const vfloat& b)
{
    return vfloat{ _mm256_add_ps(a._value, b._value) };
}

vfloat operator-(const vfloat& a, const vfloat& b)
{
    return vfloat{ _mm256_sub_ps(a._value, b._value) };
}

vfloat operator+(const vfloat& a, float b)
{
    return a + vfloat(b);
}

vfloat operator+(float a, const vfloat& b)
{
    return vfloat(a) + b;
}

vfloat operator-(const vfloat& a, const vint& b)
{
    return a - vfloat(b);
}

vfloat operator-(const vint& a, const vfloat& b)
{
    return vfloat(a) - b;
}

vfloat operator-(const vfloat& a, int b)
{
    return a - vfloat(b);
}

vfloat operator-(int a, const vfloat& b)
{
    return vfloat(a) - b;
}

vfloat operator-(const vfloat& a, float b)
{
    return a - vfloat(b);
}

vfloat operator-(float a, const vfloat& b)
{
    return vfloat(a) - b;
}

vfloat operator*(const vfloat& a, const vfloat& b)
{
    return vfloat{ _mm256_mul_ps(a._value, b._value) };
}

vfloat operator*(const vfloat& a, int b)
{
    return a * vfloat(b);
}

vfloat operator*(int a, const vfloat& b)
{
    return vfloat(a) * b;
}

vfloat operator*(const vfloat& a, float b)
{
    return a * vfloat(b);
}

vfloat operator*(float a, const vfloat& b)
{
    return vfloat(a) * b;
}

vfloat operator/(const vfloat& a, const vfloat& b)
{
    return vfloat{ _mm256_div_ps(a._value, b._value) };
}

vfloat operator/(const vfloat& a, int b)
{
    return a / vfloat(b);
}

vfloat operator/(int a, const vfloat& b)
{
    return vfloat(a) / b;
}

vfloat operator/(const vfloat& a, float b)
{
    return a / vfloat(b);
}

vfloat operator/(float a, const vfloat& b)
{
    return vfloat(a) / b;
}

vfloat operator-(const vfloat& v)
{
    return vfloat{ _mm256_sub_ps(_mm256_xor_ps(v._value, v._value), v._value) };
}

vbool operator==(const vfloat& a, const vfloat& b)
{
    return vbool{ _mm256_castps_si256(_mm256_cmp_ps(a._value, b._value, _CMP_EQ_OQ)) };
}

vbool operator<(const vfloat& a, const vfloat& b)
{
    return vbool{ _mm256_castps_si256(_mm256_cmp_ps(a._value, b._value, _CMP_LT_OQ)) };
}

vbool operator>(const vfloat& a, const vfloat& b)
{
    return vbool{ _mm256_castps_si256(_mm256_cmp_ps(a._value, b._value, _CMP_GT_OQ)) };
}

vbool operator<=(const vfloat& a, const vfloat& b)
{
    return vbool{ _mm256_castps_si256(_mm256_cmp_ps(a._value, b._value, _CMP_LE_OQ)) };
}

vbool operator>=(const vfloat& a, const vfloat& b)
{
    return vbool{ _mm256_castps_si256(_mm256_cmp_ps(a._value, b._value, _CMP_GE_OQ)) };
}

vfloat spmd_ternary(const vbool& cond, const vfloat& a, const vfloat& b)
{
    return vfloat{ _mm256_blendv_ps(b._value, a._value, _mm256_castsi256_ps(cond._value)) };
}

vfloat sqrt(const vfloat& v)
{
    return vfloat{ _mm256_sqrt_ps(v._value) };
}

vfloat abs(const vfloat& v)
{
    return vfloat{ _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v._value) };
}

vfloat floor(const vfloat& v)
{
    return vfloat{ _mm256_floor_ps(v._value) };
}

vfloat max(const vfloat& a, const vfloat& b)
{
    return vfloat{ _mm256_max_ps(a._value, b._value) };
}

vfloat min(const vfloat& a, const vfloat& b)
{
    return vfloat{ _mm256_min_ps(a._value, b._value) };
}

vfloat exp(const vfloat& v)
{
    return vfloat{ exp256_ps(v._value) };
}

vfloat log(const vfloat& v)
{
    return vfloat{ log256_ps(v._value) };
}

vfloat pow(const vfloat& a, const vfloat& b)
{
    // log256_ps(x) with x <= 0 returns nan
    // so handle the case of negative bases by fallback to stdlib
    __m256 fallback = _mm256_cmp_ps(a._value, b._value, _CMP_LE_OQ);
    int fallbackmask = _mm256_movemask_ps(fallback);
    if (fallbackmask != 0)
    {
        __declspec(align(32)) float aa[8];
        __declspec(align(32)) float bb[8];
        __declspec(align(32)) float cc[8];
        _mm256_store_ps(aa, a._value);
        _mm256_store_ps(bb, b._value);
        for (int i = 0; i < 8; i++)
        {
            if (fallbackmask & (1 << i))
                cc[i] = powf(aa[i], bb[i]);
        }
        if (fallbackmask == 0xFF)
        {
            return vfloat{ _mm256_load_ps(cc) };
        }
        else
        {
            vfloat nonfallback = exp(b * log(a));
            return vfloat{ _mm256_blendv_ps(nonfallback._value, _mm256_load_ps(cc), fallback) };
        }
    }
    else
    {
        return exp(b * log(a));
    }
}

vfloat clamp(const vfloat& v, const vfloat& a, const vfloat& b)
{
    __m256 lomask = _mm256_cmp_ps(v._value, a._value, _CMP_LT_OQ);
    __m256 himask = _mm256_cmp_ps(v._value, b._value, _CMP_GT_OQ);
    __m256 okmask = _mm256_andnot_ps(_mm256_or_ps(lomask, himask), _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ));
    return vfloat{ _mm256_or_ps(_mm256_and_ps(okmask, v._value), _mm256_or_ps(_mm256_and_ps(lomask, a._value), _mm256_and_ps(himask, b._value))) };
}

vint clamp(const vint& v, const vint& a, const vint& b)
{
    __m256i lomask = _mm256_cmpgt_epi32(a._value, v._value);
    __m256i himask = _mm256_cmpgt_epi32(v._value, b._value);
    __m256i okmask = _mm256_andnot_si256(_mm256_or_si256(lomask, himask), _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()));
    return vint{ _mm256_or_si256(_mm256_and_si256(okmask, v._value), _mm256_or_si256(_mm256_and_si256(lomask, a._value), _mm256_and_si256(himask, b._value))) };
}

vfloat fma(const vfloat& a, const vfloat& b, const vfloat& c)
{
    return vfloat{ _mm256_fmadd_ps(a._value, b._value, c._value) };
}

vfloat fms(const vfloat& a, const vfloat& b, const vfloat& c)
{
    return vfloat{ _mm256_fmsub_ps(a._value, b._value, c._value) };
}

vfloat fnma(const vfloat& a, const vfloat& b, const vfloat& c)
{
    return vfloat{ _mm256_fnmadd_ps(a._value, b._value, c._value) };
}

vfloat fnms(const vfloat& a, const vfloat& b, const vfloat& c)
{
    return vfloat{ _mm256_fnmsub_ps(a._value, b._value, c._value) };
}

lint operator+(int a, const lint& b)
{
    return lint{ _mm256_add_epi32(_mm256_set1_epi32(a), b._value) };
}

lint operator+(const lint& a, int b)
{
    return lint{ _mm256_add_epi32(a._value, _mm256_set1_epi32(b)) };
}

vfloat operator+(float a, const lint& b)
{
    return vfloat(a) + vfloat(b);
}

vfloat operator+(const lint& a, float b)
{
    return vfloat(a) + vfloat(b);
}

vfloat operator*(const lint& a, float b)
{
    return vfloat(a) * vfloat(b);
}

vfloat operator*(float b, const lint& a)
{
    return vfloat(a) * vfloat(b);
}

vint operator&(const vint& a, const vint& b)
{
    return vint{ _mm256_and_si256(a._value, b._value) };
}

vbool operator==(const vint& a, const vint& b)
{
    return vbool{ _mm256_cmpeq_epi32(a._value, b._value) };
}

vbool operator<(const vint& a, const vint& b)
{
    return vbool{ _mm256_cmpgt_epi32(b._value, a._value) };
}

vbool operator<=(const vint& a, const vint& b)
{
    return !vbool{ _mm256_cmpgt_epi32(a._value, b._value) };
}

vbool operator>=(const vint& a, const vint& b)
{
    return !vbool{ _mm256_cmpgt_epi32(b._value, a._value) };
}

vint operator+(const vint& a, const vint& b)
{
    return vint{ _mm256_add_epi32(a._value, b._value) };
}

vint operator+(const vint& a, int b)
{
    return a + vint(b);
}

vint operator+(int a, const vint& b)
{
    return vint(a) + b;
}

vint operator*(const vint& a, const vint& b)
{
    // should this return a vlong?
    return vint{ _mm256_mullo_epi32(a._value, b._value) };
}

vint operator*(const vint& a, int b)
{
    return a * vint(b);
}

vint operator*(int a, const vint& b)
{
    return vint(a) * b;
}

vbool operator==(const lint& a, const lint& b)
{
    return vbool{ _mm256_cmpeq_epi32(a._value, b._value) };
}

vbool operator==(const lint& a, int b)
{
    return vint(a) == vint(b);
}

vbool operator==(int a, const lint& b)
{
    return vint(a) == vint(b);
}

vbool operator<(const lint& a, const lint& b)
{
    return vbool{ _mm256_cmpgt_epi32(b._value, a._value) };
}

vbool operator>(const lint& a, const lint& b)
{
    return vbool{ _mm256_cmpgt_epi32(a._value, b._value) };
}

vbool operator<=(const lint& a, const lint& b)
{
    return !vbool{ _mm256_cmpgt_epi32(a._value, b._value) };
}

vbool operator>=(const lint& a, const lint& b)
{
    return !vbool{ _mm256_cmpgt_epi32(b._value, a._value) };
}

float extract(const vfloat& v, int instance)
{
    // surely there's a better way to extract
    __declspec(align(32)) float values[8];
    _mm256_store_ps(values, v._value);
    return values[instance];
}

int extract(const vint& v, int instance)
{
    // surely there's a better way to extract
    __declspec(align(32)) int values[8];
    _mm256_store_si256((__m256i*)values, v._value);
    return values[instance];
}

int extract(const lint& v, int instance)
{
    // surely there's a better way to extract
    __declspec(align(32)) int values[8];
    _mm256_store_si256((__m256i*)values, v._value);
    return values[instance];
}

bool extract(const vbool& v, int instance)
{
    // might be interesting to also have a way to return ~0 for true
    // cast vbool to vint then extract?
    
    // surely there's a better way to extract
    __declspec(align(32)) int values[8];
    _mm256_store_si256((__m256i*)values, v._value);
    return values[instance] != 0;
}

template<class SPMDKernel, class... Args>
decltype(auto) spmd_call(Args&&... args)
{
    // This is a spmd_call from outside a spmd_kernel.
    // just call the kernel with an "all on" execution mask.
    SPMDKernel kernel;
    kernel._init(exec_t::all_on());
    return kernel._call(std::forward<Args>(args)...);
}