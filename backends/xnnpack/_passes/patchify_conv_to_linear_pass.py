# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Structure modeled on backends/xnnpack/_passes/conv1d_unsqueeze_pass.py
# (executorch release/1.6, last changed in 9cd0c12d0a): match conv nodes by
# weight rank/args and rewrite the constant weight in place. Registered in
# backends/xnnpack/utils/configs.py get_transform_passes(), next to
# RemoveNoopExpandCopyPass. The patchify -> reshape/permute + matmul identity
# is the standard ViT patch-embedding equivalence.

from typing import Optional, Sequence, Tuple

import torch
from executorch.backends.transforms.utils import set_param_tensor
from executorch.backends.xnnpack.utils.utils import get_param_tensor, is_param_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import (
    ExportedProgramPassBase,
    ExportedProgramPassResult,
    ExportPass,
)
from torch.export import ExportedProgram


def _as_pair(value: Sequence[int]) -> Optional[Tuple[int, int]]:
    """ATen allows a single value to be broadcast over the spatial dims."""
    value = list(value)
    if len(value) == 1:
        value = value * 2
    if len(value) != 2 or not all(isinstance(v, int) for v in value):
        return None
    return value[0], value[1]


class PatchifyConvToLinearPass(ExportedProgramPassBase):
    """
    XNNPACK has no bf16 convolution, so a bf16 conv2d stays a (slow, scalar)
    portable op. A "patchify" convolution (kernel == stride, no padding,
    dilation 1, groups 1, as in the ViT / SigLIP patch embedding) reads
    non-overlapping patches, so it is exactly a matrix multiply:

        x:[N,C,H,W] -> view [N,C,H/kh,kh,W/kw,kw]
                    -> permute (0,2,4,1,3,5) -> [N,H/kh,W/kw,C,kh,kw]
                    -> view [N,(H/kh)*(W/kw),C*kh*kw]
                    -> linear(W.view(O,C*kh*kw), b) -> [N,P,O]
                       (emitted in aten.linear's decomposed form, view +
                       permute_copy(W) + addmm/mm + view, tagged as a linear
                       source partition so XNNPACK delegates it)
                    -> permute (0,2,1) -> view [N,O,H/kh,W/kw]

    The (c,kh,kw) flatten order matches the conv weight layout [O,C,kh,kw], so
    this is the same sum of products (only the accumulation order inside the
    gemm may differ). The linear is then delegated like any other bf16 linear.

    Only dtypes in `dtypes` are rewritten (default: bf16, the one XNNPACK cannot
    do as a convolution), so f32/fp16/quantized convs keep using XNNPACK's own
    convolution. The weight must be a constant (param/buffer/constant) used only
    by this conv; it is reshaped to [O, C*kh*kw] ahead of time, in place (the
    graph signature is unchanged, so no placeholder is created or deleted).

    This must run BEFORE partitioning (the partitioner rejects bf16 convs, so
    the conv is never handed to the delegate's own passes, and a partitioner may
    not modify the graph). It is therefore part of get_transform_passes(), the
    list of passes to give to to_edge_transform_and_lower(transform_passes=...).
    """

    dtypes: Tuple[torch.dtype, ...] = (torch.bfloat16,)

    def _match(
        self, ep: ExportedProgram, node: torch.fx.Node
    ) -> Optional[Tuple[int, int, int, int]]:
        """Returns (kh, kw, Hp, Wp) if `node` is a rewritable patchify conv."""
        if node.op != "call_function" or (
            node.target != exir_ops.edge.aten.convolution.default
        ):
            return None
        x, w, b, stride, padding, dilation, transposed, out_pad, groups = node.args
        if transposed or groups != 1:
            return None
        if any(v != 0 for v in out_pad):
            return None
        if not isinstance(w, torch.fx.Node) or not is_param_node(ep, w):
            return None
        if len(w.users) != 1:  # shared weight: leave the conv alone
            return None
        if b is not None and not (
            isinstance(b, torch.fx.Node) and is_param_node(ep, b)
        ):
            return None
        x_val, w_val = x.meta.get("val"), w.meta.get("val")
        if x_val is None or w_val is None:
            return None
        if x_val.dtype not in self.dtypes or w_val.dtype != x_val.dtype:
            return None
        if x_val.dim() != 4 or w_val.dim() != 4:
            return None
        shape = list(x_val.shape)
        if not all(isinstance(s, int) for s in shape):  # static shapes only
            return None
        kh, kw = w_val.shape[2], w_val.shape[3]
        s, p, d = _as_pair(stride), _as_pair(padding), _as_pair(dilation)
        if s is None or p is None or d is None:
            return None
        if s != (kh, kw) or p != (0, 0) or d != (1, 1):
            return None
        _, c, h, w_ = shape
        if c != w_val.shape[1] or h % kh != 0 or w_ % kw != 0:
            return None
        return kh, kw, h // kh, w_ // kw

    def call(self, ep: ExportedProgram) -> ExportedProgramPassResult:
        graph_module = ep.graph_module
        graph = graph_module.graph
        ops = exir_ops.edge.aten
        modified = False

        for node in list(graph.nodes):
            m = self._match(ep, node)
            if m is None:
                continue
            kh, kw, hp, wp = m
            x, w, b = node.args[0], node.args[1], node.args[2]
            n, c = x.meta["val"].shape[0], x.meta["val"].shape[1]
            o = w.meta["val"].shape[0]

            # Weight [O,C,kh,kw] -> [O,C*kh*kw] (same (c,kh,kw) flatten order), in place.
            w_data = get_param_tensor(ep, w)
            set_param_tensor(
                ep, w, w_data.detach().reshape(o, c * kh * kw).contiguous()
            )
            w.meta["val"] = w.meta["val"].reshape(o, c * kh * kw)
            w_lin = w

            def mk(target, args):
                return graph.create_node("call_function", target, args=args)

            with graph.inserting_before(node):
                v1 = mk(ops.view_copy.default, (x, [n, c, hp, kh, wp, kw]))
                p1 = mk(ops.permute_copy.default, (v1, [0, 2, 4, 1, 3, 5]))
                # [N,P,K] patches: this is the linear's input; the 2D view below
                # and the gemm are the linear (same form aten.linear decomposes to).
                a3 = mk(ops.view_copy.default, (p1, [n, hp * wp, c * kh * kw]))
                v2 = mk(ops.view_copy.default, (a3, [n * hp * wp, c * kh * kw]))
                wt = mk(ops.permute_copy.default, (w_lin, [1, 0]))
                if b is None:
                    mm = mk(ops.mm.default, (v2, wt))
                else:
                    mm = mk(ops.addmm.default, (b, v2, wt))
                v3 = mk(ops.view_copy.default, (mm, [n, hp * wp, o]))
                p2 = mk(ops.permute_copy.default, (v3, [0, 2, 1]))
                out = mk(ops.view_copy.default, (p2, [n, o, hp, wp]))

            # XNNPACK's addmm/mm partitioning recognizes a linear through the
            # source-partition tag that decomposing aten.linear leaves behind
            # (see AddmmConfig / ConvertToLinearPass), so tag the gemm group
            # the same way. The name must be unique per conv.
            for lin_node in (v2, wt, mm, v3):
                lin_node.meta["source_fn_stack"] = [
                    (f"linear_patchify_{node.name}", torch.nn.functional.linear)
                ]

            node.replace_all_uses_with(out)
            graph.erase_node(node)
            modified = True

        if not modified:
            return ExportedProgramPassResult(ep, False)

        graph_module.recompile()
        # Retrace to regenerate meta["val"] for the new nodes.
        ep._graph_module = ExportPass().call(graph_module).graph_module
        return ExportedProgramPassResult(ep, True)
