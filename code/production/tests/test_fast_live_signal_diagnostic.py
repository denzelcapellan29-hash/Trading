import unittest, json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fast_live_signal_diagnostic import collect_symbols, expr, derive_current_signals, aligned_pair_inputs
from trading_prod.fast_frozen_signal_core import SignalBlocked

HERE=Path(__file__).resolve().parents[1] / 'tools'
C=json.loads((HERE/'frozen_fast_pair_inputs.json').read_text())

class IngestTests(unittest.TestCase):
 def test_freeze_has_exact_97_no_missing_aliases(self):
  self.assertEqual(len(C['pairs']),31)
  self.assertEqual(len(collect_symbols(C)),97)
  self.assertEqual(set(p['fxcm_model_symbol'] for p in C['pairs'].values()), {'FX:'+x for x in C['pairs']})
 def test_all_31_frozen_expressions_parse_and_execute(self):
  for p in C['pairs'].values():
   for f in p['factors']:
    vals={a.lower():float(i+1) for i,a in enumerate(f['symbols'])}
    v=expr(f['expression'],vals)
    self.assertIsInstance(v,float)
 def test_block_code_execution_in_input_expr(self):
  for malicious in ('__import__("os").system("echo harm")','f_log(1,2)','1 ** 12','f_log(1).__class__'):
   with self.assertRaises((SignalBlocked,ValueError,SyntaxError)):
    expr(malicious,{'x':1.})
 def test_completed_week_never_uses_current_partial(self):
  pair='EURUSD';p=C['pairs'][pair];symbols={p['fxcm_model_symbol']}|{x for f in p['factors'] for x in f['symbols'].values()}
  # Fabricated source prices exercise only data plumbing (not actual FAST forecast).
  obs=1746500000.
  weekly=np.array([(obs-250*604800+i*604800,1.0+.0001*i) for i in range(250)],float)
  daily=np.array([(obs-280*86400+i*86400,1.0+.0001*i) for i in range(280)],float)
  charts={(s,tf):v for s in symbols for tf,v in [('D',daily),('W',weekly)]}
  # Incomplete last weekly candle may not become prior completed signal.
  wm,ends,eg=aligned_pair_inputs(pair,p,charts,obs)
  self.assertTrue(all(t<=obs for t in ends))
  self.assertEqual(wm.shape[1],len(p['factors'])+1)
 def test_missing_model_lookback_is_block(self):
  p=C['pairs']['EURUSD'];sym=p['fxcm_model_symbol']
  with self.assertRaisesRegex(SignalBlocked,'INSUFFICIENT_MODEL_LOOKBACK'):
   aligned_pair_inputs('EURUSD',p,{(sym,'D'):np.empty((0,2)),(sym,'W'):np.empty((0,2))},1700000000.)

if __name__=='__main__': unittest.main()
