import unittest
import numpy as np
from trading_prod.fast_frozen_signal_core import (
 _fit, eg_state, daily_state, calculate_pair, latest_completed_signal,
 _regime, SignalBlocked, EG63_CRITICAL
)

class FrozenMathematicsTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(137);n=200
  x=rng.normal(size=(n,3));y=1.1+0.4*x[:,0]-.15*x[:,1]+.1*x[:,2]+rng.normal(0,.12,n)
  self.data=np.column_stack((y,x));self.times=np.arange(n,dtype=float)*604800+1700000000
  self.states=[{'eg63':(True,True,-4.),'eg126':(True,True,-4.)}]*n
 def test_n_factor_intercept_ridge_exact(self):
  for f in (2,3,4,7):
   rng=np.random.default_rng(f);x=rng.normal(size=(100,f));y=1.1+x@rng.normal(size=f)+rng.normal(size=100)*.01
   a=np.column_stack((y,x));b,r=_fit(a);X=np.column_stack((np.ones(100),x));truth=np.linalg.solve(X.T@X+np.eye(f+1)*1e-10,X.T@y)
   self.assertLess(np.max(np.abs(b-truth)),1e-8)
   self.assertLess(np.max(np.abs(r-(y-X@truth))),1e-8)
 def test_weekly_primary_and_secondary_last52(self):
  v=calculate_pair('EURUSD',self.data,self.times,self.states)[-1]
  for mat,res,sig in [(self.data[-52:],v.primary_residual,v.primary_sigma),(np.diff(self.data[-53:],axis=0),v.secondary_residual,v.secondary_sigma)]:
   _,r=_fit(mat)
   self.assertAlmostEqual(sig,np.std(r,ddof=1),places=9)
   self.assertAlmostEqual(res,r[-1],places=9)
  self.assertAlmostEqual(v.primary_z,v.primary_residual/v.primary_sigma)
  self.assertAlmostEqual(v.secondary_fair_value,self.data[-1,0]-v.secondary_residual)
 def test_eg_matches_independent_adf_matrix_equations(self):
  t=eg_state(self.data,63,EG63_CRITICAL)
  self.assertEqual(len(t),3)
  self.assertEqual(t[0],bool(np.isfinite(t[2])))
 def test_regime_uses_prior_vol_only(self):
  self.assertEqual(_regime([.1]*103,.2)[0],0)
  self.assertEqual(_regime([.1]*104,.2)[0],3)
 def test_symmetric_stable_displacement_branch(self):
  ds=self.data.copy();ds[-1,0]+=4.0
  positive=calculate_pair('EURUSD',ds,self.times,self.states)[-1]
  self.assertEqual((positive.direction,positive.branch),(-1,1))
  ds[-1,0]-=8.0
  negative=calculate_pair('EURUSD',ds,self.times,self.states)[-1]
  self.assertEqual((negative.direction,negative.branch),(1,1))
 def test_unstable_secondary_requires_both_displacements(self):
  unstable=[{'eg63':(True,False,0.),'eg126':(True,False,0.)}]*len(self.data)
  s=calculate_pair('EURUSD',self.data,self.times,unstable)[-1]
  self.assertEqual((s.direction,s.branch),(0,0))
 def test_history_is_not_updated_by_only_nonzero_signal(self):
  s=calculate_pair('EURUSD',self.data,self.times,self.states)
  self.assertEqual(len(s),len(self.data)-51)
  self.assertEqual(s[-1].eg63_age_weeks,len(self.data))
 def test_nan_no_fake_factor_fill(self):
  w=self.data.copy();w[-1,-1]=float('nan')
  s=calculate_pair('EURUSD',w,self.times,self.states)[-1]
  self.assertEqual((s.direction,s.branch),(0,0))
 def test_no_future_week_peek(self):
  signal=latest_completed_signal('EURUSD',self.data,self.times,self.states,self.times[-1]-.01)
  self.assertEqual(signal.signal_week_end_epoch,self.times[-2])
  with self.assertRaisesRegex(SignalBlocked,'NO_COMPLETED'):
   latest_completed_signal('EURUSD',self.data,self.times,self.states,self.times[0])
 def test_missing_native_daily_state_fails_closed(self):
  state=list(self.states);state[-1]=None
  with self.assertRaisesRegex(SignalBlocked,'MISSING_NATIVE'):
   calculate_pair('EURUSD',self.data,self.times,state)

if __name__=='__main__':unittest.main()
