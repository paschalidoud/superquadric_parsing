import os
import pickle

import numpy as np
from sympy import solve, symbols, diff, cos, sin, sign, Function, S

from .fast_sampler import step_eta as fast_step_eta,\
    step_omega as fast_step_omega, collect_etas as fast_collect_etas,\
    collect_omegas as fast_collect_omegas, fast_sample, fast_sample_on_batch


class my_sign(Function):
    @classmethod
    def eval(cls, x):
        if x.is_positive:
            return S.One
        elif x.is_negative:
            return S.NegativeOne
        else:
            return None

    def fdiff(self, argindex=1):
        return S.Zero


class EqualDistanceSamplerSQ(object):
    a1, a2, a3 = symbols('a1 a2 a3', real=True)
    e1, e2 = symbols("epsilon1 epsilon2", real=True)
    eta, omega = symbols('eta omega', real=True)
    D_eta, D_omega = symbols("D_eta D_omega", real=True)
    d_eta, d_omega = symbols("delta_eta delta_omega", real=True)
    step_expressions = None

    EXPRESSION_FILE_ENV = "EXPRESSION_FILE"

    def __init__(self, n_samples, D_eta=0.005, D_omega=0.005,
                 omega_initial=-np.pi+0.001, eta_initial=-np.pi/2+0.001):
        self._n_samples = n_samples
        self.D_eta_value = D_eta
        self.D_omega_value = D_omega

        self.omega_initial_value = omega_initial
        self.eta_initial_value = eta_initial

        # Cache the etas and omegas into buckets so that we don't compute them
        # each time
        self._etas = {}
        self._omegas = {}

    @property
    def n_samples(self):
        return self._n_samples

    def step_omega_with_sympy(self, **kwargs):
        step = EqualDistanceSamplerSQ.get_step_expressions()[1].subs({
            self.D_omega: kwargs.get("D_omega", self.D_omega_value),
            self.e1: kwargs.get("eps1", 0.25),
            self.e2: kwargs.get("eps2", 0.25),
            self.a1: kwargs.get("a1", 0.25),
            self.a2: kwargs.get("a2", 0.25),
            self.a3: kwargs.get("a3", 0.25),
            self.eta: kwargs.get("eta", 0),
            self.omega: kwargs.get("omega", 0)
        })
        return abs(float(step))

    def step_omega(self, **kwargs):
        D_omega = kwargs.get("D_omega", self.D_omega_value)
        e1 = kwargs.get("eps1", 0.25)
        e2 = kwargs.get("eps2", 0.25)
        a1 = kwargs.get("a1", 0.25)
        a2 = kwargs.get("a2", 0.25)
        a3 = kwargs.get("a3", 0.25)
        eta = kwargs.get("eta", 0)
        omega = kwargs.get("omega", 0)

        t1 = (a1**2) * np.sin(omega)**4
        t1 *= np.abs(np.cos(omega))**(2*e2)
        t1 *= np.sign(np.cos(omega))**2
        t1 *= np.sign(np.cos(omega))**2

        t2 = (a2**2) * np.cos(omega)**4
        t2 *= np.abs(np.sin(omega))**(2*e2)
        t2 *= np.sign(np.sin(omega))**2
        t2 *= np.sign(np.sin(omega))**2

        t = np.sqrt(np.abs(np.cos(eta))**(-2*e1) / (t1 + t2))
        t *= D_omega * np.sin(omega) * np.cos(omega)
        step = t / (e2 * np.sign(np.cos(eta)))
        return abs(float(step))

    def step_omega_superellipsoid(self, **kwargs):
        D_omega = kwargs.get("D_omega", self.D_omega_value)
        e1 = kwargs.get("eps1", 0.25)
        e2 = kwargs.get("eps2", 0.25)
        a1 = kwargs.get("a1", 0.25)
        a2 = kwargs.get("a2", 0.25)
        a3 = kwargs.get("a3", 0.25)
        eta = kwargs.get("eta", 0)
        omega = kwargs.get("omega", 0)

        t1 = (a1**2) * np.sin(omega)**4
        t1 *= np.abs(np.cos(omega))**(2*e2)
        t1 *= np.sign(np.cos(omega))**2
        t1 *= np.sign(np.cos(omega))**2

        t2 = (a2**2) * np.cos(omega)**4
        t2 *= np.abs(np.sin(omega))**(2*e2)
        t2 *= np.sign(np.sin(omega))**2
        t2 *= np.sign(np.sin(omega))**2

        t = np.sqrt(1 / (t1 + t2))
        t *= (D_omega * np.sin(omega)*np.cos(omega)) / e2
        return abs(float(t))

    def step_omega_superellipsoid_zero_omega(self, **kwargs):
        D_omega = kwargs.get("D_omega", self.D_omega_value)
        e1 = kwargs.get("eps1", 0.25)
        e2 = kwargs.get("eps2", 0.25)
        a1 = kwargs.get("a1", 0.25)
        a2 = kwargs.get("a2", 0.25)
        a3 = kwargs.get("a3", 0.25)
        eta = kwargs.get("eta", 0)
        omega = kwargs.get("omega", 0)

        t = (D_omega * (omega ** (-e2 + 1))) / (a2 * e2)
        return abs(float(t))

    def step_eta_with_sympy(self, **kwargs):
        step = EqualDistanceSamplerSQ.get_step_expressions()[0].subs({
            self.D_eta: kwargs.get("D_eta", self.D_eta_value),
            self.e1: kwargs.get("eps1", 0.25),
            self.e2: kwargs.get("eps2", 0.25),
            self.a1: kwargs.get("a1", 0.25),
            self.a2: kwargs.get("a2", 0.25),
            self.a3: kwargs.get("a3", 0.25),
            self.eta: kwargs.get("eta", 0),
            self.omega: kwargs.get("omega", 0)
        })
        return abs(float(step))

    def step_eta(self, **kwargs):
        D_eta = kwargs.get("D_eta", self.D_eta_value)
        e1 = kwargs.get("eps1", 0.25)
        e2 = kwargs.get("eps2", 0.25)
        a1 = kwargs.get("a1", 0.25)
        a2 = kwargs.get("a2", 0.25)
        a3 = kwargs.get("a3", 0.25)
        eta = kwargs.get("eta", 0)
        omega = kwargs.get("omega", 0)

        t1 = (a1**2) * np.sin(eta)**4
        t1 *= np.abs(np.cos(eta))**(2*e1)
        t1 *= np.abs(np.cos(omega))**(2*e2)
        t1 *= np.sign(np.cos(eta))**2
        t1 *= np.sign(np.cos(omega))**2
        t1 *= np.sign(np.cos(eta))**2

        t2 = (a2**2)*(np.sin(eta)**4)
        t2 *= np.abs(np.sin(omega))**(2*e2)
        t2 *= np.abs(np.cos(eta))**(2*e1)
        t2 *= np.sign(np.sin(omega))**2
        t2 *= np.sign(np.cos(eta))**2
        t2 *= np.sign(np.cos(eta))**2

        t3 = (a3**2)*(np.cos(eta)**4)
        t3 *= np.abs(np.sin(eta))**(2*e1)
        t3 *= np.sign(np.sin(eta))**2
        t3 *= np.sign(np.sin(eta))**2

        t = np.sqrt(1.0 / (t1 + t2 + t3))*np.sin(eta)*np.cos(eta)
        step = D_eta * (t / e1)
        return abs(float(step))

    def step_eta_superellipsoid(self, **kwargs):
        D_eta = kwargs.get("D_eta", self.D_eta_value)
        e1 = kwargs.get("eps1", 0.25)
        e2 = kwargs.get("eps2", 0.25)
        a1 = kwargs.get("a1", 0.25)
        a2 = kwargs.get("a2", 0.25)
        a3 = kwargs.get("a3", 0.25)
        eta = kwargs.get("eta", 0)
        omega = kwargs.get("omega", 0)

        t1 = (a3**2)*(np.cos(eta)**4)
        t1 *= np.abs(np.sin(eta))**(2*e1)
        t1 *= np.sign(np.sin(eta))**2
        t1 *= np.sign(np.sin(eta))**2

        t2 = np.sin(eta)**4
        t2 *= np.abs(np.cos(eta))**(2*e1)
        t2 *= np.sign(np.cos(eta))**2
        t2 *= np.sign(np.cos(eta))**2

        t = np.sqrt(1 / (t1 + t2))
        t *= (D_eta * np.sin(eta) * np.cos(eta)) / e1
        return abs(float(t))

    def collect_omegas(self, **kwargs):
        def inner(omega_initial_value, **kwargs):
            omega_value = omega_initial_value
            omegas = []
            while omega_value <= np.pi:
                omegas.append(omega_value)
                kwargs.update(omega=omega_value)
                omega_value += fast_step_omega(
                    D_omega=kwargs.get("D_omega", self.D_omega_value),
                    e1=kwargs.get("eps1", 0.25),
                    e2=kwargs.get("eps2", 0.25),
                    a1=kwargs.get("a1", 0.25),
                    a2=kwargs.get("a2", 0.25),
                    a3=kwargs.get("a3", 0.25),
                    eta=kwargs.get("eta", 0),
                    omega=kwargs.get("omega", 0)
                )
            return omegas

        def omegas_is_not_acceptable(omegas):
            C = 0.1
            # Make sure that consecutive points will have a minimum distance
            diff = np.array(omegas[1:]) - np.array(omegas[:-1])
            c1 = np.max(diff) > C
            c2 = np.round(np.abs(omegas[0] + np.pi), 3) > C
            c3 = np.abs(omegas[-1] - np.pi) > 0.01
            return c1 or c2 or c3

        key = self._get_key(**kwargs)
        if key not in self._omegas:
            omegas = inner(self.omega_initial_value, **kwargs)
            D_omega = kwargs.get("D_omega", self.D_omega_value)
            while omegas_is_not_acceptable(omegas):
                D_omega = D_omega / 2
                kwargs.update(D_omega=D_omega)
                omegas = inner(self.omega_initial_value, **kwargs)
            self._omegas[key] = np.array(omegas)

        return self._omegas[key]

    def collect_etas(self, **kwargs):
        def inner(eta_initial_value, **kwargs):
            eta_value = eta_initial_value
            etas = []
            while eta_value < np.pi/2:
                etas.append(eta_value)
                kwargs.update(eta=eta_value)
                eta_value += fast_step_eta(
                    D_eta=kwargs.get("D_eta", self.D_eta_value),
                    e1=kwargs.get("eps1", 0.25),
                    e2=kwargs.get("eps2", 0.25),
                    a1=kwargs.get("a1", 0.25),
                    a2=kwargs.get("a2", 0.25),
                    a3=kwargs.get("a3", 0.25),
                    eta=kwargs.get("eta", 0),
                    omega=kwargs.get("omega", 0),
                )
            return etas

        def etas_is_not_acceptable(etas):
            C = 0.1
            # Make sure that consecutive points will have a minimum distance
            diff = np.array(etas[1:]) - np.array(etas[:-1])
            c1 = np.max(diff) > C
            # Make sure that the first sampled eta will be close to np.pi/2
            c2 = np.round(np.abs(etas[0] + np.pi/2), 3) > C
            # Make sure that the last sampled eta will be close to np.pi/2
            c3 = np.abs(etas[-1] - np.pi/2) > 0.01
            return c1 or c2 or c3

        key = self._get_key(**kwargs)
        if key not in self._etas:
            etas = inner(self.eta_initial_value, **kwargs)
            D_eta = kwargs.get("D_eta", self.D_eta_value)
            while etas_is_not_acceptable(etas):
                D_eta = D_eta / 2
                kwargs.update(D_eta=D_eta)
                etas = inner(self.eta_initial_value, **kwargs)
            self._etas[key] = np.array(etas)

        return self._etas[key]

    def _get_etas_omegas(self, **kwargs):
        etas = self.collect_etas(**kwargs)
        omegas = self.collect_omegas(**kwargs)
        return etas, omegas

    def _get_fast_etas_omegas(self, **kwargs):
        etas = fast_collect_etas(
            eta_initial=self.eta_initial_value,
            D_eta=kwargs.get("D_eta", self.D_eta_value),
            e1=kwargs.get("eps1", 0.25),
            e2=kwargs.get("eps2", 0.25),
            a1=kwargs.get("a1", 0.25),
            a2=kwargs.get("a2", 0.25),
            a3=kwargs.get("a3", 0.25),
            eta=kwargs.get("eta", 0),
            omega=kwargs.get("omega", 0)
        )
        omegas = fast_collect_omegas(
            omega_initial=self.omega_initial_value,
            D_omega=kwargs.get("D_omega", self.D_omega_value),
            e1=kwargs.get("eps1", 0.25),
            e2=kwargs.get("eps2", 0.25),
            a1=kwargs.get("a1", 0.25),
            a2=kwargs.get("a2", 0.25),
            a3=kwargs.get("a3", 0.25),
            eta=kwargs.get("eta", 0),
            omega=kwargs.get("omega", 0)
        )
        return etas, omegas

    def _sample_superellipse_divide_conquer(self, theta_lim, a1, a2, e, N):
        def xy(theta):
            return np.array([
                a1*fexp(np.cos(theta), e),
                a2*fexp(np.sin(theta), e)
            ])

        def get_sample(wA, wB, A, B, N, ident=""):
            if N <= 0:
                return
            w = (wA + wB)/2
            C = xy(w)
            dA = np.sqrt(((A-C)**2).sum())
            dB = np.sqrt(((C-B)**2).sum())
            nA = int((dA/(dA+dB))*(N-1))
            nB = N-nA-1
            for t in get_sample(wA, w, A, C, nA, ident+"l"):
                yield t
            yield w
            for t in get_sample(w, wB, C, B, nB, ident+"r"):
                yield t

        w1, w2 = theta_lim
        return [w1] + list(get_sample(w1, w2, xy(w1), xy(w2), N-2)) + [w2]

    def _get_etas_divide_conquer(self, **kwargs):
        e1 = kwargs.get("eps1", 0.25)
        e2 = kwargs.get("eps2", 0.25)
        a1 = kwargs.get("a1", 0.25)
        a2 = kwargs.get("a2", 0.25)
        a3 = kwargs.get("a3", 0.25)

        etas = self._sample_superellipse_divide_conquer(
            [np.pi/2, -np.pi/2],
            a1, a3,
            e1,
            201
        )
        return np.array(etas)

    def _get_omegas_divide_conquer(self, **kwargs):
        e1 = kwargs.get("eps1", 0.25)
        e2 = kwargs.get("eps2", 0.25)
        a1 = kwargs.get("a1", 0.25)
        a2 = kwargs.get("a2", 0.25)
        a3 = kwargs.get("a3", 0.25)

        omegas = self._sample_superellipse_divide_conquer(
            [np.pi, -np.pi],
            a1, a2,
            e2,
            201
        )
        return np.array(omegas)

    def _get_etas_omegas_divide_conquer(self, **kwargs):
        return self._get_etas_divide_conquer(**kwargs),  \
            self._get_omegas_divide_conquer(**kwargs)

    def _sample(self, **kwargs):
        etas, omegas = self._get_etas_omegas_divide_conquer(**kwargs)

        # Do random sampling
        idxs = np.random.choice(
            etas.size*omegas.size, self.n_samples, replace=False
        )
        idxs_unraveled = np.unravel_index(idxs, (etas.size, omegas.size))
        etas = etas[idxs_unraveled[0]]
        omegas = omegas[idxs_unraveled[1]]

        return etas, omegas

    def sample(self, **kwargs):
        return fast_sample(
            a1=kwargs.get("a1", 0.25),
            a2=kwargs.get("a2", 0.25),
            a3=kwargs.get("a3", 0.25),
            e1=kwargs.get("eps1", 0.25),
            e2=kwargs.get("eps2", 0.25),
            N=self.n_samples
        )

    def sample_on_batch(self, shapes, epsilons):
        return fast_sample_on_batch(
            shapes,
            epsilons,
            self.n_samples
        )

    def _get_key(self, **kwargs):
            a1 = kwargs.get("a1", 0.25)
            a2 = kwargs.get("a2", 0.25)
            a3 = kwargs.get("a3", 0.25)
            e1 = kwargs.get("eps1", 0.25)
            e2 = kwargs.get("eps2", 0.25)

            # Since a1-3 values are from 0.01-0.51 this gives a maximum of 11
            # values for each. We do the same for e1, e2 whic are from 0.2-1.8
            # thus 11**5 total values
            return (
                int(20*a1),
                int(20*a2),
                int(20*a3),
                int(10*(e1-0.2)/1.6),
                int(10*(e2-0.2)/1.6)
            )

    @staticmethod
    def fexp(x, p):
        return my_sign(x)*(abs(x)**p)

    @staticmethod
    def get_step_expressions():
        if EqualDistanceSamplerSQ.step_expressions is None:
            expression_file = os.environ.get(
                EqualDistanceSamplerSQ.EXPRESSION_FILE_ENV,
                None
            )
            try:
                with open(expression_file) as f:
                    EqualDistanceSamplerSQ.step_expressions = pickle.load(f)
            except:
                # The three scaling factors along the three axes
                a1 = EqualDistanceSamplerSQ.a1
                a2 = EqualDistanceSamplerSQ.a2
                a3 = EqualDistanceSamplerSQ.a3
                # The shape of the SQ
                e1 = EqualDistanceSamplerSQ.e1
                e2 = EqualDistanceSamplerSQ.e2
                # Angles for the longtitude and latitude
                omega = EqualDistanceSamplerSQ.omega
                eta = EqualDistanceSamplerSQ.eta
                # We follow Sec. 4.1 from paper Equal-Distance Sampling of
                # SuperEllipse Models from Pilu and Fishe, to perform the
                # sampling on the surface of the SQ
                d_eta = EqualDistanceSamplerSQ.d_eta
                d_omega = EqualDistanceSamplerSQ.d_omega
                D_eta = EqualDistanceSamplerSQ.D_eta
                D_omega = EqualDistanceSamplerSQ.D_omega

                # Compute (symbollically) the parametric formulation for the
                # superellispoid along each axis
                t1 = a1 * EqualDistanceSamplerSQ.fexp(cos(eta), e1)
                t1 = t1 * EqualDistanceSamplerSQ.fexp(cos(omega), e2)
                t2 = a2 * EqualDistanceSamplerSQ.fexp(cos(eta), e1)
                t2 = t2 * EqualDistanceSamplerSQ.fexp(sin(omega), e2)
                t3 = a3 * EqualDistanceSamplerSQ.fexp(sin(eta), e1)

                step_eta_expr = solve(
                    (diff(t1, eta)*d_eta)**2 + (diff(t2, eta)*d_eta)**2 +
                    (diff(t3, eta)*d_eta)**2 - D_eta**2,
                    d_eta
                )[1]
                step_omega_expr = solve(
                    (diff(t1, omega)*d_omega)**2 +
                    (diff(t2, omega)*d_omega)**2 +
                    (diff(t3, omega)*d_omega)**2 - D_omega**2,
                    d_omega
                )[1]
                EqualDistanceSamplerSQ.step_expressions = (
                    step_eta_expr,
                    step_omega_expr
                )
                if expression_file:
                    with open(expression_file, "w") as f:
                        pickle.dump(EqualDistanceSamplerSQ.step_expressions, f)
        return EqualDistanceSamplerSQ.step_expressions


class CuboidSampler(object):
    def __init__(self, n_samples):
        self._n_samples = n_samples

    @property
    def n_samples(self):
        return self._n_samples

    def sample(self, a1, a2, a3):
        pass

    def sample_on_batch(self, shapes, epsilons):
        pass


def get_sampler(use_cuboids, n_samples, D_eta, D_omega):
    if use_cuboids:
        sampler = CuboidSampler(n_samples)
    else:
        sampler = EqualDistanceSamplerSQ(
            n_samples,
            D_eta,
            D_omega
        )
    return sampler


def fexp(x, p):
    return np.sign(x)*(np.abs(x)**p)


def sq_surface(a1, a2, a3, e1, e2, eta, omega):
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)
    return x, y, z


def visualize_points_on_sq_mesh(e, **kwargs):
    print(kwargs)
    e1 = kwargs.get("eps1", 0.25)
    e2 = kwargs.get("eps2", 0.25)
    a1 = kwargs.get("a1", 0.25)
    a2 = kwargs.get("a2", 0.25)
    a3 = kwargs.get("a3", 0.25)
    Kx = kwargs.get("Kx", 0.0)
    Ky = kwargs.get("Ky", 0.0)

    shapes = np.array([[[a1, a2, a3]]], dtype=np.float32)
    epsilons = np.array([[[e1, e2]]], dtype=np.float32)
    etas, omegas = e.sample_on_batch(shapes, epsilons)
    x, y, z = sq_surface(a1, a2, a3, e1, e2, etas.ravel(), omegas.ravel())

    # Apply tapering
    fx = Kx * z / a3 + 1
    fy = Ky * z / a3 + 1
    fz = 1

    x = x * fx
    y = y * fy
    z = z * fz

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlim([-0.65, 0.65])
    ax.set_ylim([-0.65, 0.65])
    ax.set_zlim([-0.65, 0.65])
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    plt.show()


if __name__ == "__main__":
    e = EqualDistanceSamplerSQ(600)
    visualize_points_on_sq_mesh(e, **{
        'a1': 0.15,
        'a2': 0.15,
        'a3': 0.35,
        'eps1': 0.20715195,
        'eps2': 1.3855394,
        'Kx': 0.0,
        'Ky': 0.0
    })
