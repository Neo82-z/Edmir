import unittest
import numpy as np
from review import constrained_attack

class AttackTest(unittest.TestCase):
    def test_two_costs_have_known_exact_fractional_boundary(self):
        # 2*(1-epsilon)+margin <= 1*(1+epsilon).
        cert,delta=constrained_attack({'a':1.,'b':2.},{'a':np.array([1.,0.]),'b':np.array([0.,2.])},
                                      np.eye(2),np.ones(2),'b')
        self.assertTrue(cert['success'])
        self.assertAlmostEqual(cert['epsilon_fractional'],(1+1e-7)/3,places=8)
        self.assertLessEqual(2*(1+delta[1])+1e-7,1+delta[0]+1e-9)

    def test_active_region_exclusion_is_not_reported_as_robustness(self):
        cert,delta=constrained_attack({'a':1.,'b':2.},{'a':np.array([1.,0.]),'b':np.array([0.,2.])},
                                      np.array([[-1.,0.],[0.,1.]]),np.zeros(2),'b')
        self.assertFalse(cert['success'])
        self.assertIsNone(delta)

    def test_no_witness_beyond_search_bound(self):
        cert,delta=constrained_attack({'a':1.,'b':4.},{'a':np.array([1.,0.]),'b':np.array([0.,4.])},
                                      np.eye(2),np.ones(2),'b')
        self.assertFalse(cert['success'])
        self.assertIsNone(delta)

if __name__=='__main__':unittest.main()
