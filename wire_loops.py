from numpy import *
from ellip import ellipe, ellipk
from const import *

import unittest

def wire_loop_field(radius, x, y, z):
    rr = radius
    r1 = sqrt(x**2 + y**2)

    if r1 == 0:
        return array([0, 0, 2 * pi * rr ** 2 / pow(rr**2 + z**2, 1.5)]) * kMagneticConstant / (4 * kPi)

    theta = arctan2(y, x)
    alpha = sqrt(r1**2 + 2 * r1 * rr + rr**2 + z**2)
    beta = sqrt(r1**2 - 2 * r1 * rr + rr**2 + z**2)
    gamma = sqrt(x**2 + y**2 + z**2)

    k1 = (4*r1*rr)/alpha**2
    k2 = (-4*r1*rr)/beta**2

    ek1 = ellipe(k1)
    ek2 = ellipe(k2)
    kk1 = ellipk(k1)
    kk2 = ellipk(k2)

    cth = cos(theta)
    sth = sin(theta)

    return array([-((cth*z*(alpha*kk1*r1**2 + beta*kk2*r1**2 - 2*alpha*kk1*r1*rr + 2*beta*kk2*r1*rr + alpha*kk1*rr**2 + beta*kk2*rr**2 - alpha*ek1*(gamma**2 + rr**2) - beta*ek2*(gamma**2 + rr**2) + alpha*kk1*z**2 + beta*kk2*z**2))/
   (alpha**2*beta**2*r1)), -((sth*z*(alpha*kk1*r1**2 + beta*kk2*r1**2 - 2*alpha*kk1*r1*rr + 2*beta*kk2*r1*rr + alpha*kk1*rr**2 + beta*kk2*rr**2 - alpha*ek1*(gamma**2 + rr**2) - beta*ek2*(gamma**2 + rr**2) + alpha*kk1*z**2 +
     beta*kk2*z**2))/(alpha**2*beta**2*r1)), (alpha*kk1*r1**2 + beta*kk2*r1**2 - 2*alpha*kk1*r1*rr + 2*beta*kk2*r1*rr + alpha*kk1*rr**2 + beta*kk2*rr**2 + alpha*ek1*(-gamma**2 + rr**2) + beta*ek2*(-gamma**2 + rr**2) + alpha*kk1*z**2 +
   beta*kk2*z**2)/(alpha**2*beta**2)]) * kMagneticConstant / (4 * kPi)

class TestWireLoop(unittest.TestCase):
    def test_continuity(self):
        a = wire_loop_field(1, 1e-10, 1e-10, 1)
        b = wire_loop_field(1, 0, 0, 1)
        for i in xrange(3):
            self.assertAlmostEqual(a[i], b[i])

        a = wire_loop_field(1, 1e-10, 1e-10, 10)
        b = wire_loop_field(1, 0, 0, 10)
        for i in xrange(3):
            self.assertAlmostEqual(a[i], b[i])

    def test_field_in_the_centre(self):
        a = wire_loop_field(1, 0, 0, 0)
        b = array([0, 0, kMagneticConstant / 2])
        for i in xrange(3):
            self.assertAlmostEqual(a[i], b[i])

    def test_at_some_distance(self):
        a = wire_loop_field(10, 0, 0, 1) * 100
        b = array([0, 0, 0.0000061901020332917456])
        for i in xrange(3):
            self.assertAlmostEqual(a[i], b[i])

        a = wire_loop_field(10, 0, 0, -1) * 100
        b = array([0, 0, 0.0000061901020332917456])
        for i in xrange(3):
            self.assertAlmostEqual(a[i], b[i])
