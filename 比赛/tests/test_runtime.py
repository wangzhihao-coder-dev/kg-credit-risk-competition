"""Regression for an observed OpenMP barrier hang after tree training."""
import subprocess
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class MixedRuntimeTest(unittest.TestCase):
    def test_tree_then_tensor_allocation_finishes(self):
        code='''
import sys
sys.path.insert(0,sys.argv[1])
from graph_models import configure_cpu_runtime
import numpy as np
import torch
from lightgbm import LGBMClassifier
configure_cpu_runtime()
x=np.arange(400).reshape(200,2);y=np.arange(200)%2
LGBMClassifier(n_jobs=1,n_estimators=2,verbosity=-1).fit(x,y)
t=torch.tensor(np.zeros((5000,23)),dtype=torch.float32)
assert t.shape==(5000,23)
print('finished')
'''
        result=subprocess.run([sys.executable,'-c',code,str(ROOT/'src')],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('finished',result.stdout)
if __name__=='__main__':unittest.main()
