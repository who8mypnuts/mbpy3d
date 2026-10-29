"""Kernel functions used by the DSL tests (must live in a file for numba caching)."""
import math

from mb3d.kernel.dsl import Bool, ConstRealArray, Int, Real, RealArray, kfunc, local_real

SCALE = 2.5
STEPS = 3


@kfunc
def box_fold(x: Real, fold: Real) -> Real:
    return abs(x + fold) - abs(x - fold) - x


@kfunc
def int_pow_loop(base: Real, n: Int) -> Real:
    r = 1.0
    for i in range(n):
        r *= base
    return r


@kfunc
def classify(x: Real) -> Int:
    if x < 0.0:
        return -1
    elif x == 0.0:
        return 0
    else:
        return 1


@kfunc
def sum_array(a: RealArray[4], p: ConstRealArray[2]) -> Real:
    s = 0.0
    i = 0
    while i < 4:
        s += a[i] * p[0]
        i += 1
    return s + p[1] * SCALE


@kfunc
def poly(x: Real, p: ConstRealArray[2], flag: Bool) -> Real:
    st = local_real(4)
    st[0] = x
    st[1] = x * x
    st[2] = math.sqrt(abs(x)) + math.atan2(x, 1.0)
    st[3] = 1.0 if flag else 2.0
    v = sum_array(st, p)
    k = classify(x)
    return v + float(k) + int_pow_loop(x, STEPS) + box_fold(x, 1.0) + min(x, 0.5) + max(k, 2)
