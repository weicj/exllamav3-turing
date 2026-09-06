"""Policy coverage for deterministic TP MoE reduction defaults and overrides."""

import os
import unittest
from unittest.mock import patch

from exllamav3.modules.block_sparse_mlp import _moe_deterministic_reduce_settings


class TestMoEDeterministicReducePolicy(unittest.TestCase):

    def test_qwen_tp_default_uses_vectorized_fixed_order(self):
        with patch.dict(os.environ, {}, clear = False):
            os.environ.pop("EXL3_MOE_DETERMINISTIC_REDUCE", None)
            os.environ.pop("EXL3_MOE_DETERMINISTIC_REDUCE_MODE", None)
            self.assertEqual(
                _moe_deterministic_reduce_settings(True, True),
                (True, "expert_sum"),
            )

    def test_default_requires_tensor_parallel_qwen_policy(self):
        with patch.dict(os.environ, {}, clear = False):
            os.environ.pop("EXL3_MOE_DETERMINISTIC_REDUCE", None)
            os.environ.pop("EXL3_MOE_DETERMINISTIC_REDUCE_MODE", None)
            self.assertEqual(
                _moe_deterministic_reduce_settings(False, True),
                (False, "expert_sum"),
            )
            self.assertEqual(
                _moe_deterministic_reduce_settings(True, False),
                (False, "expert_sum"),
            )

    def test_environment_explicitly_overrides_architecture_default(self):
        with patch.dict(
            os.environ,
            {
                "EXL3_MOE_DETERMINISTIC_REDUCE": "0",
                "EXL3_MOE_DETERMINISTIC_REDUCE_MODE": "expert_index_add",
            },
            clear = False,
        ):
            self.assertEqual(
                _moe_deterministic_reduce_settings(True, True),
                (False, "expert_index_add"),
            )
