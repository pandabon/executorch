# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Structure modeled on backends/xnnpack/test/passes/test_remove_noop_expand_copy_pass.py

import unittest

import torch
from executorch.backends.xnnpack._passes.patchify_conv_to_linear_pass import (
    PatchifyConvToLinearPass,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import (
    get_transform_passes,
    get_xnnpack_edge_compile_config,
)
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops


def _count(ep, target):
    return sum(1 for n in ep.graph.nodes if n.op == "call_function" and n.target == target)


class TestPatchifyConvToLinearPass(unittest.TestCase):
    CONV = exir_ops.edge.aten.convolution.default

    def setUp(self):
        torch._dynamo.reset()

    @staticmethod
    def _conv(k, s, p=0, dtype=torch.bfloat16, bias=True):
        return torch.nn.Conv2d(3, 8, k, s, p, bias=bias).to(dtype).eval()

    def _rewritten(self, m, x):
        edge = to_edge(torch.export.export(m, (x,), strict=True))
        return edge.transform([PatchifyConvToLinearPass()]).exported_program()

    def test_rewrites_bf16_patchify_and_matches_conv(self):
        for bias in (True, False):
            m, x = self._conv(4, 4, bias=bias), torch.randn(2, 3, 16, 24).to(torch.bfloat16)
            ep = self._rewritten(m, x)
            self.assertEqual(_count(ep, self.CONV), 0)
            ref = m(x).float()
            err = (ep.module()(x).float() - ref).norm() / ref.norm()
            self.assertLess(err.item(), 1e-2)

    def test_keeps_non_patchify_and_non_bf16_convs(self):
        x = torch.randn(1, 3, 16, 16)
        for m, xx in (
            (self._conv(4, 2), x.bfloat16()),  # overlapping patches
            (self._conv(4, 4, p=1), x.bfloat16()),  # padding
            (self._conv(4, 4, dtype=torch.float32), x),  # f32: XNNPACK has conv
        ):
            self.assertEqual(_count(self._rewritten(m, xx), self.CONV), 1)

    def test_bf16_patchify_is_fully_delegated(self):
        m, x = self._conv(4, 4), torch.randn(1, 3, 16, 16).to(torch.bfloat16)
        edge = to_edge_transform_and_lower(
            torch.export.export(m, (x,), strict=True),
            partitioner=[XnnpackPartitioner(enable_bf16=True)],
            transform_passes=get_transform_passes(),
            compile_config=get_xnnpack_edge_compile_config(),
        )
        ep = edge.exported_program()
        self.assertEqual(_count(ep, self.CONV), 0)
        self.assertEqual(
            sum("executorch_call_delegate" in str(n.target) for n in ep.graph.nodes), 1
        )
        self.assertEqual(_count(ep, exir_ops.edge.aten.addmm.default), 0)  # no portable gemm left


if __name__ == "__main__":
    unittest.main()
