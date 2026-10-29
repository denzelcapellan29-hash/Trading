import copy,csv,json,sys,tempfile,unittest
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from compare_fast_pine_reference import compare,read_reference,NUMERIC,BOOLEAN


class FastReferenceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.pairs=json.loads((ROOT/'tools/frozen_fast_pair_inputs.json').read_text())['pairs']
        self.snapshot={'freeze_id':'FX-FAST-2026-08-28','signals':{}}
        for pair in self.pairs:
            fields={key:1. for key in NUMERIC}
            fields.update({key:True for key in BOOLEAN})
            fields.update(direction=0,branch=0,confidence_score=1,confidence_high=False,
                          multiplier=None,base_planned_risk=None,
                          model_week_end_utc='2026-10-30T21:00:00+00:00')
            self.snapshot['signals'][pair]=fields

    def test_complete_matching_reference_never_grants_release(self):
        result=compare(self.snapshot,copy.deepcopy(self.snapshot),self.pairs)
        self.assertTrue(result['comparison_pass'])
        self.assertFalse(result['prospective_asof_verified'])
        self.assertFalse(result['orders_authorized'])

    def test_flat_pairs_cannot_be_omitted(self):
        missing=copy.deepcopy(self.snapshot);del missing['signals']['EURUSD']
        with self.assertRaisesRegex(ValueError,'COMPLETE_31'):compare(self.snapshot,missing,self.pairs)

    def test_missing_none_field_is_not_same_as_no_trade(self):
        ref=copy.deepcopy(self.snapshot);del ref['signals']['EURUSD']['multiplier']
        self.assertFalse(compare(self.snapshot,ref,self.pairs)['comparison_pass'])

    def test_all_mismatches_are_reported_including_exact_flags(self):
        ref=copy.deepcopy(self.snapshot)
        ref['signals']['EURUSD']['primary_z']+=.01
        ref['signals']['USDJPY']['eg63_stable']=0
        r=compare(self.snapshot,ref,self.pairs)
        self.assertEqual({(x['pair'],x['field']) for x in r['mismatches']},
                         {('EURUSD','primary_z'),('USDJPY','eg63_stable')})

    def test_active_trade_cannot_have_missing_multiplier(self):
        ref=copy.deepcopy(self.snapshot);ref['signals']['EURUSD']['direction']=1
        self.assertFalse(compare(ref,copy.deepcopy(ref),self.pairs)['comparison_pass'])

    def test_comparison_tolerance_does_not_relax_categorical_multiplier(self):
        ref=copy.deepcopy(self.snapshot);actual=copy.deepcopy(self.snapshot)
        for s in (ref,actual):s['signals']['EURUSD'].update(direction=1,branch=1,multiplier=1,base_planned_risk=.016)
        actual['signals']['EURUSD']['multiplier']=1+1e-8
        self.assertFalse(compare(actual,ref,self.pairs)['comparison_pass'])

    def test_unknown_daily_state_is_never_complete_even_if_both_match(self):
        ref=copy.deepcopy(self.snapshot);ref['signals']['EURUSD']['eg126_available']=False
        self.assertFalse(compare(ref,copy.deepcopy(ref),self.pairs)['comparison_pass'])

    def write_references(self,directory):
        columns=['time','EXPORT Signal Week End Timestamp']+list(NUMERIC.values())+list(BOOLEAN.values())
        for pair,state in self.snapshot['signals'].items():
            row={'time':'2026-11-02T08:00:00+00:00',
                 'EXPORT Signal Week End Timestamp':1793394000000}
            # Use the actual prior Friday from the fixture, rather than a
            # hardcoded epoch, so time-unit and timezone handling are exercised.
            row['EXPORT Signal Week End Timestamp']=int(datetime.fromisoformat(state['model_week_end_utc']).timestamp()*1000)
            row.update({title:state[key] for key,title in NUMERIC.items()})
            row.update({title:int(state[key]) for key,title in BOOLEAN.items()})
            with (directory/f'OANDA_{pair}, 60.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerow(row)

    def test_csv_selects_closed_08utc_bar_for_09utc_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);self.write_references(directory)
            ref=read_reference(directory,datetime(2026,11,2,9,tzinfo=timezone.utc),self.pairs)
            self.assertTrue(compare(self.snapshot,ref,self.pairs)['comparison_pass'])
            self.assertEqual(len(ref['receipts']),31)

    def test_csv_missing_decision_bar_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);self.write_references(directory)
            p=directory/'OANDA_EURUSD, 60.csv'
            p.write_text(p.read_text().replace('2026-11-02T08:00:00+00:00','2026-11-02T09:00:00+00:00'))
            with self.assertRaisesRegex(ValueError,'EXACT_SINGLE'):read_reference(directory,datetime(2026,11,2,9,tzinfo=timezone.utc),self.pairs)
