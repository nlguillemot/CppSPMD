#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cassert>
#include <utility>

struct spmd_kernel
{
    struct vbool;
    struct vint;

    struct exec_t
    {
        __m256i _mask;

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
    friend bool any(const exec_t& e);

    // the execution mask at entry of the kernel
    exec_t _kernel_exec;

    // the execution mask at the current point of varying control flow
    exec_t _internal_exec;

    // the OR of all lanes which hit a "spmd_break" in the current loop
    exec_t _break_mask;

    // the OR of all lanes which hit a "spmd_continue" in the current loop
    exec_t _continue_mask;

    // current control flow's execution mask (= _kernel_exec & _internal_exec)
    exec_t exec;

    // this is basically the constructor
    // can't use a real constructor without requiring users
    // to eg. say "using spmd_kernel::spmd_kernel;", which is blah.
    void _init(const exec_t& kernel_exec)
    {
        _kernel_exec = kernel_exec;
        _internal_exec = exec_t::all_on();
        exec = kernel_exec;
    }

    struct vbool
    {
        __m256i _value;

        explicit vbool(const __m256i& value)
            : _value(value)
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

        vfloat(int value)
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
            return vbool{
                _mm256_xor_si256(
                    _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_value, _mm256_setzero_si256())) };
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

        operator vfloat() const
        {
            return vfloat{ _mm256_cvtepi32_ps(_value) };
        }

        vfloat_lref operator[](float* ptr) const
        {
            return vfloat_lref{ ptr + _mm_cvtsi128_si32(_mm256_extracti128_si256(_value, 0)) };
        }

        vint_lref operator[](int* ptr) const
        {
            return vint_lref{ ptr + _mm_cvtsi128_si32(_mm256_extracti128_si256(_value, 0)) };
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
        // apply "if" mask
        exec_t cond_exec(cond);
        exec_t pre_if_internal_exec = _internal_exec & cond_exec;
        exec_t pre_if_exec = exec & cond_exec;
        
        _internal_exec = pre_if_internal_exec;
        exec = pre_if_exec;

        if (any(exec)) // "all off" optimization
        {
            ifBody();
        }

        // propagate any lanes that were shut down inside the if
        // (assuming lanes haven't "come back to life" at the end of an "if")
        // eg: spmd_if(x, [&]{ spmd_break(); });
        _internal_exec = andnot(pre_if_internal_exec ^ _internal_exec, _internal_exec);
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

    template<class ForInitBody, class ForCondBody, class ForIncrBody, class ForBody>
    void spmd_for(const ForInitBody& forInitBody, const ForCondBody& forCondBody, const ForIncrBody& forIncrBody, const ForBody& forBody)
    {
        // save old execution mask
        exec_t old_exec = exec;
        exec_t old_internal_exec = _internal_exec;

        // execute the initialization clause of the loop
        forInitBody();

        // save the state of the previous loop (assuming there was one)
        // then start fresh for this loop
        exec_t old_continue_mask = _continue_mask;
        exec_t old_break_mask = _break_mask;
        
        _continue_mask = exec_t::all_off();
        _break_mask = exec_t::all_off();

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

        // if any lanes hit a break statement, they were shut off in the loop.
        // now that the loop is done, their execution can be restored.
        _internal_exec = _internal_exec | _break_mask;
        exec = _internal_exec & _kernel_exec;

        // restore the continue/break of the previous loop in the stack
        _continue_mask = old_continue_mask;
        _break_mask = old_break_mask;
    }

    template<class ForeachBody>
    void spmd_foreach(int first, int last, const ForeachBody& foreachBody)
    {
        // could allow this, just too lazy right now.
        assert(first <= last);

        // number of loops that don't require loop tail masking
        int numFullLoops = ((last - first) / programCount) * programCount;

        // number of loops that require loop tail masking
        // happens when the loop count is not a multiple of programCount)
        int numPartialLoops = (last - first) % programCount;

        // do every loop that doesn't need to be tail masked
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

    void spmd_break()
    {
        // this should only be called inside loops. A check for this would be good.

        // set currently active lanes as "break"'d
        _break_mask = _break_mask | _internal_exec;

        // turn off all active lanes so nothing happens after the break.
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
        // if all lanes are off, that means the control flow shouldn't even get here
        // since we don't immediately return when all lanes turn off, zombie execution is possible.
        if (!any(exec))
            return;

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

spmd_kernel::vbool operator!(const spmd_kernel::vbool& v)
{
    return spmd_kernel::vbool{
        _mm256_xor_si256(
            _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()),
            v._value) };
}

spmd_kernel::exec_t::exec_t(const spmd_kernel::vbool& b)
{
    _mask = _mm256_cmpeq_epi32(b._value, _mm256_setzero_si256());
}

spmd_kernel::exec_t operator&(const spmd_kernel::exec_t& a, const spmd_kernel::exec_t& b)
{
    return spmd_kernel::exec_t{ _mm256_and_si256(a._mask, b._mask) };
}

spmd_kernel::exec_t operator|(const spmd_kernel::exec_t& a, const spmd_kernel::exec_t& b)
{
    return spmd_kernel::exec_t{ _mm256_or_si256(a._mask, b._mask) };
}

bool any(const spmd_kernel::exec_t& e)
{
    return _mm256_movemask_ps(_mm256_castsi256_ps(e._mask)) != 0;
}

spmd_kernel::exec_t andnot(const spmd_kernel::exec_t& a, const spmd_kernel::exec_t& b)
{
    return spmd_kernel::exec_t{ _mm256_andnot_si256(a._mask, b._mask) };
}

spmd_kernel::vbool operator||(const spmd_kernel::vbool& a, const spmd_kernel::vbool& b)
{
    return spmd_kernel::vbool{ _mm256_or_si256(a._value, b._value) };
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
    return spmd_kernel::vbool{ _mm256_castps_si256(_mm256_cmp_ps(a._value, b._value, _CMP_EQ_OQ)) };
}

spmd_kernel::vbool operator<(const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b)
{
    return spmd_kernel::vbool{ _mm256_castps_si256(_mm256_cmp_ps(a._value, b._value, _CMP_LT_OQ)) };
}

spmd_kernel::vbool operator>(const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b)
{
    return spmd_kernel::vbool{ _mm256_castps_si256(_mm256_cmp_ps(a._value, b._value, _CMP_GT_OQ)) };
}

spmd_kernel::vfloat spmd_ternary(const spmd_kernel::vbool& cond, const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b)
{
    return spmd_kernel::vfloat{ _mm256_blendv_ps(b._value, a._value, _mm256_castsi256_ps(cond._value)) };
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

spmd_kernel::vfloat fma(const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b, const spmd_kernel::vfloat& c)
{
    return spmd_kernel::vfloat{ _mm256_fmadd_ps(a._value, b._value, c._value) };
}

spmd_kernel::vfloat fms(const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b, const spmd_kernel::vfloat& c)
{
    return spmd_kernel::vfloat{ _mm256_fmsub_ps(a._value, b._value, c._value) };
}

spmd_kernel::vfloat fnma(const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b, const spmd_kernel::vfloat& c)
{
    return spmd_kernel::vfloat{ _mm256_fnmadd_ps(a._value, b._value, c._value) };
}

spmd_kernel::vfloat fnms(const spmd_kernel::vfloat& a, const spmd_kernel::vfloat& b, const spmd_kernel::vfloat& c)
{
    return spmd_kernel::vfloat{ _mm256_fnmsub_ps(a._value, b._value, c._value) };
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
    return spmd_kernel::vbool{ _mm256_cmpeq_epi32(a._value, b._value) };
}

spmd_kernel::vbool operator<(const spmd_kernel::vint& a, const spmd_kernel::vint& b)
{
    return spmd_kernel::vbool{ _mm256_cmpgt_epi32(b._value, a._value) };
}

spmd_kernel::vint operator+(const spmd_kernel::vint& a, const spmd_kernel::vint& b)
{
    return spmd_kernel::vint{ _mm256_add_epi32(a._value, b._value) };
}

spmd_kernel::vbool operator==(const spmd_kernel::lint& a, const spmd_kernel::lint& b)
{
    return spmd_kernel::vbool{ _mm256_cmpeq_epi32(a._value, b._value) };
}

spmd_kernel::vbool operator<(const spmd_kernel::lint& a, const spmd_kernel::lint& b)
{
    return spmd_kernel::vbool{ _mm256_cmpgt_epi32(b._value, a._value) };
}

spmd_kernel::vbool operator>(const spmd_kernel::lint& a, const spmd_kernel::lint& b)
{
    return spmd_kernel::vbool{ _mm256_cmpgt_epi32(a._value, b._value) };
}

template<class SPMDKernel, class... Args>
auto spmd_call(Args&&... args)
{
    // This is a spmd_call from outside a spmd_kernel.
    // just call the kernel with an "all on" execution mask.
    SPMDKernel kernel;
    kernel._init(spmd_kernel::exec_t::all_on());
    return kernel._call(std::forward<Args>(args)...);
}