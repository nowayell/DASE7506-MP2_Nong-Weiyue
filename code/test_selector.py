import unittest
from fractions import Fraction
import numpy as np
from selector import arithmetic,features
class Checks(unittest.TestCase):
 def test_safe_arithmetic(self):
  self.assertEqual(arithmetic('12*3'),36)
  self.assertEqual(arithmetic('1/3+1/6'),Fraction(1,2))
  with self.assertRaises(ValueError):arithmetic('__import__("os")')
 def test_group_equivalence(self):
  bank=[{'question':'one half','candidates':[{'text':t} for t in ('#### 0.5','#### 1/2','bad','bad')]}]
  x,m,r,a,g=features(bank,np.ones((1,4,1024)),'E4')
  self.assertEqual(g,[[[0,1],[2],[3]]]);self.assertEqual(int(m.sum()),3);self.assertTrue(np.isfinite(x).all())
if __name__=='__main__':unittest.main()
