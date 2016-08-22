#pragma once

#include <emmintrin.h>

#include <cassert>

struct vint;

struct exec_t
{
    __m128 _mask;
};

exec_t operator&(const exec_t& a, const exec_t& b)
{
    return exec_t{ _mm_and_ps(a._mask, b._mask) };
}

exec_t operator~(const exec_t& a)
{
    return exec_t{ _mm_andnot_ps(a._mask, _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps())) };
}

static exec_t exec = exec_t{ _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps()) };

struct vbool
{
    __m128 _value;
    
    vbool& operator=(const vbool& other)
    {
        _value = _mm_or_ps(_mm_and_ps(exec._mask, other._value), _mm_andnot_ps(exec._mask, _value));
        return *this;
    }
};

vbool operator||(const vbool& a, const vbool& b)
{
    return vbool{ _mm_or_ps(a._value, b._value) };
}

struct vfloat
{
    __m128 _value;

    explicit vfloat(const __m128& value)
        : _value(value)
    { }

    vfloat(float value)
        : _value(_mm_set1_ps(value))
    { }

    vfloat(int value)
        : _value(_mm_set1_ps((float)value))
    { }

    vfloat& operator=(const vfloat& other)
    {
        _value = _mm_or_ps(_mm_and_ps(exec._mask, other._value), _mm_andnot_ps(exec._mask, _value));
        return *this;
    }

    vfloat& operator+=(const vfloat& other)
    {
        _value = _mm_or_ps(_mm_and_ps(exec._mask, _mm_add_ps(_value, other._value)), _mm_andnot_ps(exec._mask, _value));
        return *this;
    }

    vfloat& operator*=(const vfloat& other)
    {
        _value = _mm_or_ps(_mm_and_ps(exec._mask, _mm_mul_ps(_value, other._value)), _mm_andnot_ps(exec._mask, _value));
        return *this;
    }
};

vfloat operator*(const vfloat& a, const vfloat& b)
{
    return vfloat{ _mm_mul_ps(a._value, b._value) };
}

vfloat operator/(const vfloat& a, const vfloat& b)
{
    return vfloat{ _mm_div_ps(a._value, b._value) };
}

vfloat operator+(const vfloat& a, const vfloat& b)
{
    return vfloat{ _mm_add_ps(a._value, b._value) };
}

vfloat operator-(const vfloat& a, const vfloat& b)
{
    return vfloat{ _mm_sub_ps(a._value, b._value) };
}

vfloat operator-(const vfloat& a)
{
    return vfloat{ _mm_sub_ps(_mm_xor_ps(a._value,a._value), a._value) };
}

vfloat abs(const vfloat& a)
{
    return vfloat{ _mm_andnot_ps(_mm_set1_ps(-0.0f), a._value) };
}

vfloat sqrt(const vfloat& a)
{
    return vfloat{ _mm_sqrt_ps(a._value) };
}

vfloat floor(const vfloat& a)
{
    __m128 fval = _mm_cvtepi32_ps(_mm_cvtps_epi32(a._value));
    return vfloat{ _mm_sub_ps(fval, _mm_and_ps(_mm_cmplt_ps(a._value, fval), _mm_set1_ps(1.0f))) };
}

vfloat clamp(const vfloat& v, const vfloat& a, const vfloat& b)
{
    __m128 lomask = _mm_cmplt_ps(v._value, a._value);
    __m128 himask = _mm_cmpgt_ps(v._value, b._value);
    __m128 okmask = _mm_andnot_ps(_mm_or_ps(lomask, himask), _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps()));
    return vfloat{ _mm_or_ps(_mm_and_ps(okmask, v._value), _mm_or_ps(_mm_and_ps(lomask, a._value), _mm_and_ps(himask, b._value))) };
}

vbool operator<(const vfloat& a, const vfloat& b)
{
    return vbool{ _mm_cmplt_ps(a._value, b._value) };
}

vfloat spmd_ternary(const vbool& cond, const vfloat& a, const vfloat& b)
{
    return vfloat{ _mm_or_ps(_mm_and_ps(cond._value, a._value), _mm_andnot_ps(cond._value, b._value)) };
}

// reference to a vint stored randomly in memory
struct vint_vref
{
    int* _value;
    __m128i _vindex;

    // scatter
    vint_vref& operator=(const vint& other);

    // gather
    operator vint() const;
};

struct vint
{
    __m128i _value;

    explicit vint(const __m128i value)
        : _value(value)
    { }

    vint(int value)
        : _value(_mm_set1_epi32(value))
    { }

    vint(const vfloat& other)
        : _value(_mm_cvtps_epi32(other._value))
    { }

    operator vbool() const
    {
        return vbool{ _mm_castsi128_ps(
            _mm_andnot_si128(
                _mm_cmpeq_epi32(_value, _mm_setzero_si128()),
                _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()))) };
    }

    operator vfloat() const
    {
        return vfloat{ _mm_cvtepi32_ps(_value) };
    }

    vint& operator=(const vint& other)
    {
        _value = _mm_castps_si128(
            _mm_or_ps(
                _mm_and_ps(
                    exec._mask, 
                    _mm_castsi128_ps(other._value)),
                _mm_andnot_ps(
                    exec._mask, 
                    _mm_castsi128_ps(_value))));
        return *this;
    }

    vint& operator&=(int other)
    {
        _value = _mm_castps_si128(
            _mm_or_ps(
                _mm_and_ps(
                    exec._mask, 
                    _mm_castsi128_ps(_mm_and_si128(_value, _mm_set1_epi32(other)))),
                _mm_andnot_ps(
                    exec._mask, 
                    _mm_castsi128_ps(_value))));
        return *this;
    }

    vint_vref operator[](int* ptr) const
    {
        return vint_vref{ ptr, _value };
    }
};

// scatter
vint_vref& vint_vref::operator=(const vint& src)
{
    __declspec(align(16)) int vindex[4];
    _mm_store_si128((__m128i*)vindex, _vindex);

    __declspec(align(16)) int stored[4];
    _mm_store_si128((__m128i*)stored, src._value);

    int mask = _mm_movemask_ps(exec._mask);
    for (int i = 0; i < 4; i++)
    {
        if (mask & (1 << i))
            _value[vindex[i]] = stored[i];
    }

    return *this;
}

// gather
vint_vref::operator vint() const
{
    __declspec(align(16)) int vindex[4];
    _mm_store_si128((__m128i*)vindex, _vindex);

    __declspec(align(16)) int loaded[4];

    int mask = _mm_movemask_ps(exec._mask);
    for (int i = 0; i < 4; i++)
    {
        if (mask & (1 << i))
            loaded[i] = _value[vindex[i]];
    }

    return vint{ _mm_load_si128((__m128i*)loaded) };
}

vint operator+(const vint& a, const vint& b)
{
    return vint{ _mm_add_epi32(a._value, b._value) };
}

vint operator&(const vint& a, const vint& b)
{
    return vint{ _mm_and_si128(a._value, b._value) };
}

vbool operator<(const vint& a, const vint& b)
{
    return vbool{ _mm_castsi128_ps(_mm_cmplt_epi32(a._value, b._value)) };
}

vbool operator==(const vint& a, const vint& b)
{
    return vbool{ _mm_castsi128_ps(_mm_cmpeq_epi32(a._value, b._value)) };
}

// reference to a vfloat stored linearly in memory
struct vfloat_lref
{
    float* _value;

    // scatter
    vfloat_lref& operator=(const vfloat& other)
    {
        int mask = _mm_movemask_ps(exec._mask);
        if (mask == 0b1111)
        {
            // "all on" optimization: vector store
            _mm_storeu_ps(_value, other._value);
        }
        else
        {
            // hand-written masked store
            __declspec(align(16)) float stored[4];
            _mm_storeu_ps(stored, other._value);

            for (int i = 0; i < 4; i++)
            {
                if (mask & (1 << i))
                    _value[i] = stored[i];
            }
        }
        return *this;
    }

    // gather
    operator vfloat() const
    {
        int mask = _mm_movemask_ps(exec._mask);
        if (mask == 0b1111)
        {
            // "all on" optimization: vector load
            return vfloat{ _mm_loadu_ps(_value) };
        }
        else
        {
            // hand-written masked load
            __declspec(align(16)) float loaded[4];
            for (int i = 0; i < 4; i++)
            {
                if (mask & (1 << i))
                    loaded[i] = _value[i];
            }

            return vfloat{ _mm_loadu_ps(loaded) };
        }
    }
};

struct lint
{
    __m128i _value;

    operator vfloat() const
    {
        return vfloat{ _mm_cvtepi32_ps(_value) };
    }

    vfloat_lref operator[](float* ptr) const
    {
        return vfloat_lref{ ptr + _mm_cvtsi128_si32(_value) };
    }
};

lint operator+(const lint& a, int b)
{
    return lint{ _mm_add_epi32(a._value, _mm_set1_epi32(b)) };
}

lint operator+(int a, const lint& b)
{
    return lint{ _mm_add_epi32(_mm_set1_epi32(a), b._value) };
}

vbool operator<(const lint& a, const lint& b)
{
    return vbool{ _mm_castsi128_ps(_mm_cmplt_epi32(a._value, b._value)) };
}

vbool operator==(const lint& a, const lint& b)
{
    return vbool{ _mm_castsi128_ps(_mm_cmpeq_epi32(a._value, b._value)) };
}


static const lint programIndex = lint{ _mm_set_epi32(3,2,1,0) };
static const int programCount = 4;

template<class IfBody>
void spmd_if(const vbool& cond, const IfBody& ifBody)
{
    // save old execution mask
    exec_t old_exec = exec;

    // apply "if" mask
    exec = exec & exec_t{ cond._value };

    // "all off" optimization
    int mask = _mm_movemask_ps(exec._mask);
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
    int mask = _mm_movemask_ps(exec._mask);
    if (mask != 0)
    {
        ifBody();
    }

    // invert mask for "else"
    exec = ~exec & old_exec;
    
    // "all off" optimization
    mask = _mm_movemask_ps(exec._mask);
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
    __m128i loopIndex = _mm_add_epi32(programIndex._value, _mm_set1_epi32(first));
    for (int i = 0; i < numFullLoops; i += programCount)
    {
        foreachBody(lint{ loopIndex });
        loopIndex = _mm_add_epi32(loopIndex, _mm_set1_epi32(programCount));
    }

    // do a partial loop if necessary (if loop count is not a multiple of programCount)
    if (numPartialLoops > 0)
    {
        // save old execution mask
        exec_t old_exec = exec;

        // apply mask for partial loop
        exec = exec & exec_t{ _mm_castsi128_ps(_mm_cmplt_epi32(programIndex._value, _mm_set1_epi32(numPartialLoops))) };

        // do the partial loop
        foreachBody(lint{ loopIndex });

        // restore execution mask
        exec = old_exec;
    }
}