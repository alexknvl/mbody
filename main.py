#!/usr/bin/env python
from __future__ import division, print_function, unicode_literals

from numpy import *
from wire_loops import wire_loop_field, TestWireLoop
from const import *
from numpy.random import rand, normal, randint
from numpy.linalg import norm

import unittest

def random_point_on_disk(maxRadius=1, coordinates='polar'):
    assert maxRadius > 0
    # For explanation @see
    # http://www.ece.virginia.edu/mv/edu/prob/stat/random-number-generation.pdf
    # Since the total area of a disk is pi * r^2,
    # the PDF is p(x) = 2 * pi * r * dr / pi * R^2 == 2 * r * dr / R^2
    # and the CDF is F(x) = r^2 / R^2, thus F^-1(y) = sqrt(R^2 * y)
    angle = 2 * pi * rand()
    radius = sqrt(rand()) * maxRadius

    assert 0 <= angle  < 2 * pi
    assert 0 <= radius < maxRadius

    if coordinates == 'polar':
        return (angle, radius)
    elif coordinates == 'cartesian':
        return (radius * cos(angle), radius * sin(angle))
    else:
        assert False

def random_point_on_sphere_surface(radius=1, coordinates='spherical'):
    assert radius >= 0

    # For explanation @see
    # http://www.ece.virginia.edu/mv/edu/prob/stat/random-number-generation.pdf
    phi = 2 * pi * rand()
    theta = arccos(1 - 2 * rand())

    assert 0 <= phi   <  2 * pi
    assert 0 <= theta <= pi

    if coordinates == 'spherical':
        return (radius, theta, phi)
    elif coordinates == 'cylindrical':
        return (radius * sin(theta), phi, radius * cos(theta))
    elif coordinates == 'cartesian':
        return (radius * sin(theta) * cos(phi),
                radius * sin(theta) * sin(phi),
                radius * cos(theta))

def random_point_inside_sphere(maxRadius=1, coordinates='spherical'):
    assert maxRadius >= 0

    radius = maxRadius * pow(rand(), 1/3)
    assert 0 <= radius < maxRadius

    return random_point_on_sphere_surface(radius, coordinates)

def dd(f, x, n):
    eps = 1e-10
    return (f(x + n * eps) - f(x)) / eps

def jacobi_matrix(f, x):
    f0 = lambda x: f(x)[0]
    f1 = lambda x: f(x)[1]
    f2 = lambda x: f(x)[2]
    n0 = array([1, 0, 0])
    n1 = array([0, 1, 0])
    n2 = array([0, 0, 1])

    return matrix([[dd(f0, x, n0), dd(f0, x, n1), dd(f0, x, n2)],
                   [dd(f1, x, n0), dd(f1, x, n1), dd(f1, x, n2)],
                   [dd(f2, x, n0), dd(f2, x, n1), dd(f2, x, n2)]])

class TestJacobiMatrix(unittest.TestCase):
    def test_simple(self):
        def f(x):
            return array([x[1],x[2],x[0]])
        m = jacobi_matrix(f, [0, 0, 0])
        n = matrix([[0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]])

        for i in xrange(3):
            for j in xrange(3):
                self.assertAlmostEqual(m[i, j], n[i, j])

class TestMatrixMult(unittest.TestCase):
    def test_matrix_mult(self):
        m = matrix([[0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]])
        s = matrix([[1, 2, 3]])

        #print(array(s * m)[0])

def proj(u, v):
    assert dot(u, u) > 0
    return u * dot(v, u) / dot(u, u)


def nearly_equal(a, b, ndigits=5):
    return round(a - b, ndigits) == 0

def gram_schmidt(set):
    i = 0
    while i < len(set):
        # print(i, set[i])
        for j in xrange(i):
            set[i] = set[i] - proj(set[j], set[i])
            # print("-", j, set[j], set[i])

        n = norm(set[i])

        if nearly_equal(n, 0, 10):
            del set[i]
        else:
            #print(i, norm(set[i]))
            set[i] = set[i] / n
            #print(i, norm(set[i]))
            i += 1
    return set

class TestGramSchmidt(unittest.TestCase):
    def test_simple(self):
        set = [array([1, 0, 0]),
               array([0, 0, 1]),
               array([0, 1, 0])]
        new_set = gram_schmidt(set)

        assert len(new_set) == 3
        for i in xrange(3):
            for j in xrange(3):
                self.assertAlmostEqual(set[i][j], new_set[i][j])

    def test_simple_linear_dep(self):
        set = [array([1, 0, 0]),
               array([0, 0, 1]),
               array([0, 0, 1]),
               array([0, 1, 0])]
        a =   [array([1, 0, 0]),
               array([0, 0, 1]),
               array([0, 1, 0])]
        b = gram_schmidt(set)

        self.assertTrue(len(b) == 3)
        for i in xrange(3):
            for j in xrange(3):
                self.assertAlmostEqual(a[i][j], b[i][j])

    def test_linear_dep(self):
        set = [array([1, 0, 0]),
               array([0, 0, 1]),
               array([0.48, 0, 0.5]),
               array([0, 1, 0])]
        a =   [array([1, 0, 0]),
               array([0, 0, 1]),
               array([0, 1, 0])]
        b = gram_schmidt(set)

        self.assertTrue(len(b) == 3)
        for i in xrange(3):
            for j in xrange(3):
                self.assertAlmostEqual(a[i][j], b[i][j])

    def test_another(self):
        set = [array([ 0.01442135,  0.00605002,  0.9998777 ]),
            array([1, 0, 0]),
            array([0, 1, 0]),
            array([0, 0, 1])]
        b = gram_schmidt(set)

        self.assertTrue(len(b) == 3)

def complete_basis(vec):
    assert not isnan(sum(vec))
    assert not isinf(sum(vec))
    assert norm(vec) > 0

    set = [vec.copy(),
           array([1, 0, 0]),
           array([0, 1, 0]),
           array([0, 0, 1])]
    #print(set)
    set = gram_schmidt(set)
    #print(set)

    assert len(set) == 3
    assert all(set[0] == vec/norm(vec))
    #print([norm(v) for v in set])
    assert all([nearly_equal(norm(v), 1, 8) for v in set])
    assert nearly_equal(dot(set[0], set[1]), 0, 8)
    assert nearly_equal(dot(set[1], set[2]), 0, 8)
    assert nearly_equal(dot(set[2], set[0]), 0, 8)

    return set

def random_orthogonal_vector(n):
    n, a, b = tuple(complete_basis(n))
    angle = 2 * pi * rand()
    return a * cos(angle) + b * sin(angle)

class Particle:
    __slots__ = ["mass", "position", "velocity", "spin", "adiabaticMode"]

    def __str__(self):
        return "position=%s, velocity=%s, spin=%s, adiabaticMode=%s" %\
            (self.position, self.velocity, self.spin, self.adiabaticMode)

def field(x):
    return wire_loop_field(0.1, x[0], x[1], x[2]-0.05) + \
           wire_loop_field(0.1, x[0], x[1], x[2]+0.05)

def field_and_derivatives(x):
    F = field(x)
    J = jacobi_matrix(field, x)
    return (F, J)

def calc_derivatives(m, s, v, x, O, J):
    return (cross(O, s), -kReducedPlanckConstant / m * array(s * J)[0], v)

always_adiabatic = True
two_states = True

def select_time_step(m, s, v, x, O, J):
    kMinStep = 1e-3
    kTimeDivider = 25
    kMinAdiabaticRatio = 25

    def good_float(f):
        return not isnan(f) and not isinf(f)

    t0 = 2 * kPi / norm(O)

    if always_adiabatic:
        return (kMinStep, True)

    o = array((t0 * J * matrix([v]).T)).T
    o = abs(o[0])
    t1 = 1/o[0]
    t2 = 1/o[1]
    t3 = 1/o[2]
    alpha = max([t1, t2, t3]) / t0

    if dot(O, O) == 0:
        return (kMinStep, True)
    else:
        ts = [t0 / kTimeDivider,
              t1 / kTimeDivider,
              t2 / kTimeDivider,
              t3 / kTimeDivider,
              kMinStep]
        ts = list(filter(good_float, ts))

        return (min(ts), alpha > kMinAdiabaticRatio)

def integration_step(p):
    s, v, x, m = p.spin, p.velocity, p.position, p.mass

    F, J = field_and_derivatives(p.position)
    O = F * kBohrMagneton / kReducedPlanckConstant
    J = J * kBohrMagneton / kReducedPlanckConstant

    # If we are in adiabatic mode already, we assume that
    # spin is always parallel to the field
    if p.adiabaticMode:
        s = s * (F / norm(F))

    dt, am = select_time_step(m, s, v, x, O, J)
    sd, vd, xd = calc_derivatives(m, s, v, x, O, J)

    if p.adiabaticMode:
        if am:
            # Adiabatic mode
            pass
        else:
            # Exiting adiabatic mode
            n = F / norm(F)
            o = random_orthogonal_vector(n)

            p.spin = p.spin * n + sqrt(1/4 - p.spin ** 2) * o
            p.adiabaticMode = False

    elif am:
        # Entering adiabatic mode
        p.spin = dot(p.spin, F) / norm(F)
        p.adiabaticMode = True
    else:
        # Non-adiabatic mode
        p.spin += sd * dt
        p.spin *= (1/2) / sqrt(dot(p.spin, p.spin))

    p.velocity += vd * dt
    p.position += xd * dt

if __name__ == "__main__":
    for i in xrange(100000):
        p = Particle()
        p.mass = kHydrogenMass

        x, y = random_point_on_disk(maxRadius=0.0005, coordinates='cartesian')
        z = 0.05 + 1e-7
        p.position = array([x, y, z])

        if two_states:
            sign = -1 if randint(2) == 0 else 1
            F = field(p.position)
            p.spin = 1/2 * F / norm(F) * sign
        else:
            p.spin = array(random_point_on_sphere_surface(
                radius=1/2, coordinates='cartesian'))

        kTemp = 1e-3
        p.velocity = normal(size=3) * sqrt(kBoltzmannConstant * kTemp / p.mass)

        p.adiabaticMode = False

        def in_cylinder():
            a = 0.05 < p.position[2] < 0.15
            b = norm(p.position[0:2]) < 0.10
            return a and b

        count = 0
        while in_cylinder():
            integration_step(p)

            # count += 1
            # if count % 5000:
            #     print(i, p)

        if p.position[2] > 0.15:
            print(p.position[0], p.position[1], p.position[2],
                  p.velocity[0], p.velocity[1], p.velocity[2])
