import pathlib,sys,unittest
import numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from trading_prod.equity.corridor_accept import classify_family,_first_exit,make_pivots

class CorridorAcceptTests(unittest.TestCase):
    def test_corridor_classification(self):
        nodes=[(99.,2,2,-.6,0.),(101.,2,2,.6,0.)]
        self.assertEqual(classify_family(nodes)[0],"corridor")
    def test_stop_first_same_bar(self):
        O=np.array([100.]);H=np.array([103.]);L=np.array([97.])
        j,px,status=_first_exit(O,H,L,0,1,102.,98.)
        self.assertEqual((j,px,status),(0,98.,"ambiguous_stop_first"))
    def test_pivots_are_causally_confirmed(self):
        H=np.array([1.,2.,4.,2.,1.,2.,1.]);L=H-.5;C=H-.2
        conf,inv,level,kind=make_pivots(H,L,C,2)
        self.assertTrue(any((conf==4)&(level==4.)))

if __name__=="__main__":unittest.main()
