#!/usr/bin/env python3
"""Verify ADF-lag1 math independently of archived publication-time alignment."""
import unittest
import numpy as np
from statsmodels.tsa.stattools import adfuller
from fast_daily_eg_shadow import fit_eg

class EGCoreMath(unittest.TestCase):
 def test_eg_residual_adf_lag1_matches_statsmodels(self):
  rng=np.random.default_rng(2408)
  for n in (63,126):
   x=rng.normal(size=(n,3))
   resid=np.zeros(n)
   for i in range(1,n):resid[i]=0.63*resid[i-1]+rng.normal(0,.20)
   y=1+0.25*x[:,0]-0.1*x[:,1]+1.2*x[:,2]+resid
   frame=np.column_stack((y,x))
   available,stable,stat=fit_eg(frame,-3)
   self.assertTrue(available)
   # Reference regress on exact OLS residuals produced by same ridge spec;
   # statsmodels ADF 'c' with maxlag=1 and autolag=None uses same design.
   X=np.column_stack((np.ones(n),x))
   b=np.linalg.inv(X.T@X+1e-10*np.eye(4))@(X.T@y)
   stats=adfuller(y-X@b,maxlag=1,regression='c',autolag=None)[0]
   self.assertAlmostEqual(stat,stats,places=8)

if __name__=='__main__':unittest.main(verbosity=2)
