import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from trading_prod.equity.corridor_accept import classify_family,first_barrier_exit,make_pivots,TRAIN_NEAR_Q

class CorridorAcceptTests(unittest.TestCase):
    def test_corridor_classification(self):
        nodes=[{"level":99.0,"members":2,"source_diversity":2,"Q":2,"dist":-.6},{"level":101.0,"members":2,"source_diversity":2,"Q":2,"dist":.6}]
        self.assertEqual(classify_family(nodes)[0],"corridor")

    def test_stop_first_same_bar(self):
        O=np.array([100.]);H=np.array([103.]);L=np.array([97.])
        j,px,status=first_barrier_exit(O,H,L,0,1,102.,98.)
        self.assertEqual((j,px,status),(0,98.,"ambiguous_stop_first"))

    def test_pivots_are_causally_confirmed(self):
        H=np.array([1.,2.,4.,2.,1.,2.,1.]);L=H-.5;C=H-.2
        conf,inv,level,kind=make_pivots(H,L,C,2)
        self.assertTrue(any((conf==4)&(level==4.)))

    def test_train_near_gate_is_frozen_phase3_value(self):
        self.assertAlmostEqual(TRAIN_NEAR_Q,.357323996666185,15)

if __name__=="__main__":
    unittest.main()
