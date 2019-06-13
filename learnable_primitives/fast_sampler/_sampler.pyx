import numpy as np
import cython
from libc.math cimport sqrt, sin, cos, floor, M_PI_2, M_PI
from libc.float cimport FLT_MIN
cimport numpy as cnp


cdef inline float sign(float x):
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    if x == 0:
        return 0.0


@cython.wraparound(False)
cdef inline float _step_eta(
    float D_eta,
    float a1,
    float a2,
    float a3,
    float e1,
    float e2,
    float eta,
    float omega
):
    cdef float t1, t2, t3, t
    t1 = (a1**2) * sin(eta)**4
    t1 *= abs(cos(eta))**(2*e1)
    t1 *= abs(cos(omega))**(2*e2)
    t1 *= sign(cos(eta))**2
    t1 *= sign(cos(omega))**2
    t1 *= sign(cos(eta))**2

    t2 = (a2**2)*(sin(eta)**4)
    t2 *= abs(sin(omega))**(2*e2)
    t2 *= abs(cos(eta))**(2*e1)
    t2 *= sign(sin(omega))**2
    t2 *= sign(cos(eta))**2
    t2 *= sign(cos(eta))**2

    t3 = (a3**2)*(cos(eta)**4)
    t3 *= abs(sin(eta))**(2*e1)
    t3 *= sign(sin(eta))**2
    t3 *= sign(sin(eta))**2

    t = sqrt(1.0 / (t1 + t2 + t3))*sin(eta)*cos(eta)
    return abs(D_eta * (t / e1))


@cython.wraparound(False)
cpdef float step_eta(
    float D_eta,
    float a1,
    float a2,
    float a3,
    float e1,
    float e2,
    float eta,
    float omega
):
    cdef float step
    step = _step_eta(
        D_eta,
        a1,
        a2,
        a3,
        e1,
        e2,
        eta,
        omega
    )
    return step


@cython.wraparound(False)
cdef inline float _step_omega(
    float D_omega,
    float a1,
    float a2,
    float a3,
    float e1,
    float e2,
    float eta,
    float omega
):
    cdef float t1, t2, t3, t
    t1 = (a1**2) * sin(omega)**4
    t1 *= abs(cos(omega))**(2*e2)
    t1 *= sign(cos(omega))**2
    t1 *= sign(cos(omega))**2

    t2 = (a2**2) * cos(omega)**4
    t2 *= abs(sin(omega))**(2*e2)
    t2 *= sign(sin(omega))**2
    t2 *= sign(sin(omega))**2

    t = sqrt(abs(cos(eta))**(-2*e1) / (t1 + t2))
    t *= D_omega * sin(omega) * cos(omega)
    return abs(t / (e2 * sign(cos(eta))))


@cython.wraparound(False)
cpdef float step_omega(
    float D_omega,
    float a1,
    float a2,
    float a3,
    float e1,
    float e2,
    float eta,
    float omega
):
    cdef float step
    step = _step_omega(
        D_omega,
        a1,
        a2,
        a3,
        e1,
        e2,
        eta,
        omega
    )
    return step


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int update_etas(
    float eta_initial,
    float D_eta,
    float a1,
    float a2,
    float a3,
    float e1,
    float e2,
    float eta,
    float omega,
    float[:] etas
):
    # counter to keep track of the number of etas added in the list
    cdef int i = 0
    cdef float eta_value = eta_initial
    while eta_value < M_PI/2:
        # Update etas with the value
        etas[i] = eta_value
        if eta_value == 0.0:
            eta_value += 0.01
            continue
        # increment counter by 1
        eta_update = _step_eta(D_eta, a1, a2, a3, e1, e2, eta_value, omega)
        eta_value += eta_update
        i += 1
    return i


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int update_omegas(
    float omega_initial,
    float D_omega,
    float a1,
    float a2,
    float a3,
    float e1,
    float e2,
    float eta,
    float omega,
    float[:] omegas
):
    # counter to keep track of the number of omegas added in the list
    cdef int i = 0
    cdef float omega_value = omega_initial
    while omega_value <= M_PI:
        # Update etas with the value
        omegas[i] = omega_value
        if omega_value == 0.0:
            omega_value += 0.01
            continue
        # increment counter by 1
        omega_update = _step_omega(D_omega, a1, a2, a3, e1, e2,
                                   eta, omega_value)
        omega_value += omega_update
        i += 1
    return i


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int etas_is_not_acceptable(float[:] etas, int N):
    cdef float C = 0.1
    cdef int c1, c2, c3, i
    cdef float max_d = 0.0
    cdef float current_val = 0.0
    for i in range(1, N-1):
        current_val = etas[i] - etas[i-1]
        if current_val > max_d:
            max_d = current_val

    c1 = max_d > C
    c2 = <int>floor(abs(etas[0] + M_PI_2)) > C
    c3 = floor(abs(M_PI_2 - etas[N-1])) > 0.01
    return c1 or c2 or c3


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int omegas_is_not_acceptable(float[:] omegas, int N):
    cdef float C = 0.1
    cdef int c1, c2, c3, i
    cdef float max_d = 0.0
    cdef float current_val = 0.0
    for i in range(1, N-1):
        current_val = omegas[i] - omegas[i-1]
        if current_val > max_d:
            max_d = current_val

    c1 = max_d > C
    c2 = <int>floor(abs(omegas[0] + M_PI)) > C
    c3 = floor(abs(M_PI - omegas[N-1])) > 0.01
    return c1 or c2 or c3


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float32_t, ndim=1] collect_etas(
    float eta_initial,
    float D_eta,
    float a1,
    float a2,
    float a3,
    float e1,
    float e2,
    float eta,
    float omega
):
    cdef float D_eta_value = D_eta
    cdef cnp.ndarray[cnp.float32_t, ndim=1] etas = \
            np.empty(shape=(100000,), dtype=np.float32)
    # Do the first update
    cdef int N
    N = update_etas(eta_initial, D_eta_value, a1, a2, a3, e1, e2,
                    eta, omega, etas)

    # while (etas_is_not_acceptable(etas, N) == 1) and (N>500 and N<200):
    while (etas_is_not_acceptable(etas, N) == 1):
        D_eta_value = D_eta_value / 2.0
        # Update etas with the new D_eta_value
        N = update_etas(eta_initial, D_eta_value, a1, a2, a3, e1, e2,
                        eta, omega, etas)
    return etas[:N]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float32_t, ndim=1] collect_omegas(
    float omega_initial,
    float D_omega,
    float a1,
    float a2,
    float a3,
    float e1,
    float e2,
    float eta,
    float omega
):
    cdef float D_omega_value = D_omega
    cdef cnp.ndarray[cnp.float32_t, ndim=1] omegas = \
            np.empty(shape=(100000,), dtype=np.float32)
    # Do the first update
    cdef int N
    N = update_omegas(omega_initial, D_omega, a1, a2, a3, e1, e2,
                      eta, omega, omegas)
    while (omegas_is_not_acceptable(omegas, N) == 1):
        D_omega_value = D_omega_value / 2.0
        # Update omegas with the new D_omega_value
        N = update_omegas(omega_initial, D_omega_value, a1, a2, a3, e1, e2,
                          eta, omega, omegas)

    return omegas[:N]


cdef inline float fexp(float x, float p):
    return sign(x) * (abs(x)**p)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void xy(float theta, float a1, float a2, float e, float[2] C):
    C[0] = a1 * fexp(cos(theta), e)
    C[1] = a2 * fexp(sin(theta), e)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float distance(float[2] a, float[2] b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _sample_superellipse_divide_conquer_inner(
    float a1,
    float a2,
    float e,
    float theta_A,
    float theta_B,
    float[2] A,
    float[2] B,
    int N,
    float[::1] thetas,
    int * idx
):
    if N <= 0:
        return

    cdef float theta
    cdef float dA
    cdef float dB
    cdef float[2] C
    cdef int nA
    cdef int nB

    theta = (theta_A + theta_B)/2
    xy(theta, a1, a2, e, C)
    dA = distance(A, C)
    dB = distance(C, B)
    nA = int((dA/(dA+dB))*(N-1))
    nB = N - nA - 1

    _sample_superellipse_divide_conquer_inner(
        a1, a2, e,
        theta_A, theta,
        A, C,
        nA,
        thetas,
        idx
    )
    thetas[idx[0]] = theta
    idx[0] += 1
    _sample_superellipse_divide_conquer_inner(
        a1, a2, e,
        theta, theta_B,
        C, B,
        nB,
        thetas,
        idx
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sample_superellipse_divide_conquer(
    float a1,
    float a2,
    float e,
    float theta_A,
    float theta_B,
    float[::1] thetas
):
    # Decleare some useful variables
    cdef int N = thetas.shape[0]
    cdef float[2] A
    cdef float[2] B
    cdef int i = 0  # counter for the added elements in the list

    # Compute the two points on the superellipse
    xy(theta_A, a1, a2, e, A)
    xy(theta_B, a1, a2, e, B)

    # Add etas_init in the beginnint
    thetas[i] = theta_A
    i += 1
    _sample_superellipse_divide_conquer_inner(
        a1, a2, e, theta_A, theta_B, A, B, N-2, thetas, &i
    )
    # Add etas_end in the end
    thetas[i] = theta_B


def fast_sample(
    float a1,
    float a2,
    float a3,
    float e1,
    float e2,
    int N
):
    # Allocate memory for etas and omegas
    etas = np.empty((201,), dtype=np.float32)
    omegas = np.empty((201,), dtype=np.float32)
    sample_superellipse_divide_conquer(
        a1, a3, e1, M_PI_2, -M_PI_2, etas
    )
    sample_superellipse_divide_conquer(
        a1, a2, e2, M_PI, -M_PI, omegas
    )

    # Do the random sampling
    idxs = np.random.choice(
        etas.size*omegas.size, N, replace=False
    )
    idxs_unraveled = np.unravel_index(idxs, (etas.size, omegas.size))

    etas = etas[idxs_unraveled[0]]
    omegas = omegas[idxs_unraveled[1]]

    return etas, omegas


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_sample_on_batch(
    float[:, :, ::1] shapes,
    float[:, :, ::1] epsilons,
    int N
):
    # Declare some variables
    cdef int B = shapes.shape[0]
    cdef int M = shapes.shape[1]
    cdef int buffer_size = 201

    # Allocate memory for the etas and omegas
    cdef cnp.ndarray[cnp.float32_t, ndim=3] etas = \
        np.zeros((B, M, N), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] omegas = \
        np.zeros((B, M, N), dtype=np.float32)
    with nogil:
        sample_on_batch(
            &shapes[0, 0, 0],
            &epsilons[0, 0, 0],
            <float *>etas.data,
            <float *>omegas.data,
            B, M, N,
            buffer_size,
            0
        )

    return etas, omegas

cdef extern from "sampling.hpp" nogil:
    void sample_on_batch(
        float *shapes,
        float *epsilons,
        float *etas,
        float *omegas,
        int B,
        int M,
        int N,
        int buffer_size,
        int seed
    )
