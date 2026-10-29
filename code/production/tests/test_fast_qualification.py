import copy,csv,json,lzma,sys,unittest
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from qualify_fast_offline import verify_bundle,audit_sources
from fast_live_signal_diagnostic import require_volatility_warmup
from trading_prod.fast_frozen_signal_core import SignalBlocked,eg_state,EG63_CRITICAL,EG126_CRITICAL,classify_frozen_decision
from trading_prod.fast_clock import decision_clocks


class FastQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=ROOT.parent/'fx_fast'
        cls.manifest=list(csv.DictReader((cls.directory/'PINE_MANIFEST.csv').read_text(encoding='utf-8-sig').splitlines()))
        cls.registry=json.loads((ROOT/'tools/frozen_fast_pair_inputs.json').read_text())
        cls.bundle=(cls.directory/'frozen_pine_31pair_bundle.txt.xz').read_bytes()
        cls.index=json.loads((cls.directory/'frozen_pine_bundle_index.json').read_text())

    def files(self):
        payload=lzma.decompress(self.bundle)
        return {pair:payload[r['offset']:r['offset']+r['bytes']] for pair,r in self.index['files'].items()}

    def test_all_31_restored_sources_match_original_hash_roots(self):
        self.assertTrue(verify_bundle(self.bundle,self.index,self.manifest))
        self.assertEqual(audit_sources(self.files(),self.manifest,self.registry)['verified_factor_registries'],31)

    def test_source_mutation_is_rejected(self):
        files=self.files();files['EURUSD']+=b'\n// changed'
        with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):audit_sources(files,self.manifest,self.registry)

    def test_input_alias_mutation_is_rejected(self):
        registry=copy.deepcopy(self.registry)
        registry['pairs']['EURUSD']['factors'][0]['symbols']['EU03MY']='WRONG:SOURCE'
        with self.assertRaisesRegex(ValueError,'FACTOR_SYMBOL_MISMATCH'):
            audit_sources(self.files(),self.manifest,registry)

    def test_260_rows_cannot_supply_260_prior_volatility_estimates(self):
        with self.assertRaisesRegex(SignalBlocked,'INSUFFICIENT_FULL'):
            require_volatility_warmup(np.column_stack((np.ones(260),np.ones(260))))
        require_volatility_warmup(np.column_stack((np.ones(274),np.ones(274))))

    def test_missing_price_in_required_warmup_is_rejected(self):
        data=np.ones((300,3));data[-274,0]=float('nan')
        with self.assertRaisesRegex(SignalBlocked,'INCOMPLETE_FULL'):require_volatility_warmup(data)

    def test_decision_clocks_follow_both_dst_calendars(self):
        for date,hour,exit_hour in [('2026-03-09',9,21),('2026-03-30',8,21),
                                    ('2026-10-26',9,21),('2026-11-02',9,22)]:
            entry=datetime.fromisoformat(date+f'T{hour:02}:00:00+00:00')
            clocks=decision_clocks(entry)
            self.assertEqual(clocks['scheduled_exit_utc'].hour,exit_hour)
            self.assertEqual(clocks['scheduled_exit_utc'].weekday(),4)
            self.assertLess(clocks['prior_model_week_close_utc'],entry)

    def test_no_tuesday_holiday_or_late_entry_fallback(self):
        for value in ('2026-11-03T09:00:00+00:00','2026-11-02T09:01:00+00:00','2026-11-02T08:00:00+00:00'):
            with self.assertRaisesRegex(ValueError,'no fallback'):decision_clocks(datetime.fromisoformat(value))

    def test_adf_statistic_against_independent_augmented_qr_regressions(self):
        # QR least squares with augmented rows applies the fixed OLS ridge.
        # A separate QR ADF solve checks the lag layout, degrees of freedom,
        # standard error and statistic; it does not call the production _fit.
        for factors in range(2,8):
            for length,critical in [(63,EG63_CRITICAL),(126,EG126_CRITICAL)]:
                for walk in (False,True):
                    rng=np.random.default_rng(factors*1000+length+int(walk))
                    x=rng.normal(size=(length,factors))
                    noise=rng.normal(size=length)
                    if walk:noise=np.cumsum(noise)
                    y=1+x@rng.normal(size=factors)+noise
                    design=np.column_stack((np.ones(length),x))
                    beta=np.linalg.lstsq(np.vstack((design,np.eye(factors+1)*1e-5)),
                        np.r_[y,np.zeros(factors+1)],rcond=None)[0]
                    residual=y-design@beta
                    delta=np.diff(residual)
                    a=np.column_stack((np.ones(length-2),residual[1:-1],delta[:-1]))
                    dependent=delta[1:]
                    fit=np.linalg.lstsq(a,dependent,rcond=None)[0]
                    q,r=np.linalg.qr(a,mode='reduced')
                    invr=np.linalg.solve(r,np.eye(3))
                    variance=np.sum((dependent-a@fit)**2)/(len(dependent)-3)
                    expected=fit[1]/np.sqrt(variance*(invr@invr.T)[1,1])
                    actual=eg_state(np.column_stack((y,x)),length,critical)
                    with self.subTest(factors=factors,length=length,walk=walk):
                        self.assertTrue(actual[0]);self.assertAlmostEqual(actual[2],expected,places=7)
                        self.assertEqual(actual[1],expected<=critical)


class FrozenBoundaryTests(unittest.TestCase):
    def choose(self,**overrides):
        values=dict(primary_z=-2.,secondary_z=-3.,eg63_available=True,eg63_stable=True,
                    eg126_available=True,eg126_stable=True,eg63_age=12,eg126_age=26,
                    confidence_high=True,residual_acceleration_z=0.,vol_regime=1)
        values.update(overrides);return classify_frozen_decision(**values)

    def test_primary_minimum_is_strict_and_marginal_upper_is_inclusive(self):
        for sign in (-1,1):
            self.assertEqual(self.choose(primary_z=sign*1.5)['direction'],0)
            at=self.choose(primary_z=sign*1.625)
            self.assertTrue(at['primary_marginal']);self.assertEqual(at['multiplier'],.5)
            above=self.choose(primary_z=sign*(1.625+1e-8))
            self.assertFalse(above['primary_marginal'])

    def test_secondary_primary_gate_strict_secondary_gate_inclusive(self):
        self.assertEqual(self.choose(eg63_stable=False,primary_z=1.625)['direction'],0)
        for z in (-2.,2.):
            got=self.choose(eg63_stable=False,primary_z=1.625+1e-8,secondary_z=z)
            self.assertEqual(got['branch'],2);self.assertEqual(got['direction'],-1 if z>0 else 1)
            self.assertEqual(got['multiplier'],1.)
        self.assertEqual(self.choose(eg63_stable=False,secondary_z=2.-1e-8)['direction'],0)

    def test_mature_63_requires_26_week_confirmation_to_restore_size(self):
        self.assertEqual(self.choose(eg63_age=13,eg126_age=25)['multiplier'],.5)
        self.assertFalse(self.choose(eg63_age=13,eg126_age=26)['weak_mature'])
        self.assertTrue(self.choose(eg63_age=13,eg126_age=26,eg126_available=False)['weak_mature'])

    def test_low_confidence_adverse_acceleration_middle_reduction(self):
        self.assertEqual(self.choose(confidence_high=False,residual_acceleration_z=-1,vol_regime=2)['multiplier'],.5)
        self.assertEqual(self.choose(confidence_high=False,residual_acceleration_z=-1,vol_regime=1)['multiplier'],.75)

    def test_missing_eg_or_disallowed_direction_is_not_an_entry(self):
        self.assertEqual(self.choose(eg63_available=False)['direction'],0)
        self.assertEqual(self.choose(allow_long=False)['direction'],0)
        self.assertEqual(self.choose(primary_z=2.,allow_short=False)['direction'],0)
