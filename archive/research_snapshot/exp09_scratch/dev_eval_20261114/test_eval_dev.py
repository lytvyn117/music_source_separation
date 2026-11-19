import unittest
import numpy as np
import torch
from evaluate_dev import exp09_output, measure, paired, RATE


class Identity(torch.nn.Module):
    def forward(self, x):
        return x * .25, x * .75


class EvaluationTests(unittest.TestCase):
    def test_overlap_alignment_and_tail(self):
        rng = np.random.default_rng(5)
        for n in (7829, RATE * 6 + 317):
            x = rng.normal(0,.1,(2,n)).astype("float32")
            v, i = exp09_output(Identity(),x,torch.device("cpu"))
            np.testing.assert_allclose(v,x*.25,atol=2e-7,rtol=0)
            np.testing.assert_allclose(v+i,x,atol=2e-7,rtol=0)

    def test_metrics_know_leakage_and_retention(self):
        rng=np.random.default_rng(6)
        v,i=rng.normal(0,.1,(2,2,RATE*2)).astype("float32")
        r=measure(v+i,v,i,.75*v,i+.25*v)
        self.assertAlmostEqual(r["vocal_bleed_coeff"],.25,places=6)
        self.assertAlmostEqual(r["vocal_bleed_energy_pct"],6.25,places=5)
        self.assertAlmostEqual(r["vocal_retention"],.75,places=6)
        self.assertAlmostEqual(r["vocal_sdr_db"],12.0412,places=3)
        self.assertLess(r["mixture_max_abs"],2e-7)

    def test_equal_artist_weight_and_all_paired(self):
        rows=[{"artist_key":"a","exp09":{"vocal_sdr_db":v},"exp04":{"vocal_sdr_db":0}}
              for v in (1,1,1)]
        rows.append({"artist_key":"b","exp09":{"vocal_sdr_db":3},"exp04":{"vocal_sdr_db":0}})
        r=paired(rows,"exp04","vocal_sdr_db",.01)
        self.assertEqual(r["artist_macro_delta"],2)
        self.assertEqual(r["wins"],4)
