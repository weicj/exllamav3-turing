"""Unit coverage for the QSA dense-prefill/sparse-decode policy."""

import os
import unittest
from unittest.mock import patch

from exllamav3.modules.attn import _bc_max_qlen, qsa_prefill_dense_enabled


class TestQSAPrefillPolicy(unittest.TestCase):

    def test_is_opt_in_and_excludes_decode_shapes(self):
        with patch.dict(os.environ, {}, clear = False):
            os.environ.pop("EXL3_QSA_PREFILL_DENSE", None)
            self.assertFalse(qsa_prefill_dense_enabled(_bc_max_qlen + 1))

        with patch.dict(os.environ, {"EXL3_QSA_PREFILL_DENSE": "1"}, clear = False):
            self.assertFalse(qsa_prefill_dense_enabled(_bc_max_qlen))
            self.assertTrue(qsa_prefill_dense_enabled(_bc_max_qlen + 1))
            self.assertTrue(qsa_prefill_dense_enabled(4096))


if __name__ == "__main__":
    unittest.main()
