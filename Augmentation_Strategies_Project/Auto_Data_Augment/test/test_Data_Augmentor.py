import unittest
import numpy as np
from ..Data_Augmentor import Data_Augmentor

class Test_Data_Augumentor(unittest.TestCase):
    def setUp(self):
        self.da_inst = Data_Augmentor.Data_Augumentor()

    def tearDown(self):
        del self.da_inst
        self.da_inst = None

    def test_prnt_policies():
        pv0_res = Data_Augumentor.aug_inst.policy_v0()
        psim_res = Data_Augumentor.aug_inst.policy_simple()
        # class return type chk
        self.assertIsInstance(p0_res, list)
        self.assertIsInstance(psim_res, list)
        # policy instance size 
        self.assertGreaterEqual(len(pv0_res), 0, msg) 
        self.assertGreaterEqual(len(psim_res), 0, msg) 
    
    def test_data_augment():
        dummy_img = np.random.random((3, 3, 4))
        aug_img = self.da_inst.data_augment(dummy_img)
        self.assertIsInstance(aug_img, np.array)

if __name__ == '__main__':
    unittest.main(verbosity=2)
