import unittest
import tempfile
from pathlib import Path
from src.hf_oof import acquire_oof_weights

class HFTest(unittest.TestCase):
    def test_disabled_does_not_access_network(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(acquire_oof_weights(d,[0,1],enabled=False),{})
    def test_reuses_two_local_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'oof';root.mkdir()
            for ext in ('pth','json'):(root/f'fold_0_model.{ext}').write_bytes(b'test')
            out=acquire_oof_weights(d,[0],enabled=True)
            self.assertEqual(set(out),{0})

if __name__=='__main__':unittest.main()
