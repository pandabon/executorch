# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator
from typing import List, Optional

import torch
from executorch.backends.xnnpack._passes.decompose_batch_norm import DecomposeBatchNorm
from executorch.backends.xnnpack._passes.fuse_batch_norm import FuseBatchNormPass
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
    XNNPartitionerConfig,
)
from executorch.backends.xnnpack.utils.utils import is_param_node
from executorch.exir.backend.utils import WhyNoPartition
from torch.export import ExportedProgram

logger = logging.getLogger(__name__)
why = WhyNoPartition(logger=logger)


class BatchNormConfig(XNNPartitionerConfig):
    target_name = "_native_batch_norm_legit_no_training.default"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        if not self.check_common_constraints(node, ep):
            return False

        bn = node
        input_node = node.all_input_nodes[0]

        can_decompose = DecomposeBatchNorm.can_decompose_batch_norm(node, ep, why)
        can_fuse = FuseBatchNormPass.can_fuse(input_node, bn, ep)

        if not can_fuse and not can_decompose:
            why(node, f"BatchNorm cannot be decomposed or fused with {input_node}")
            return False

        return True

    def get_node_and_deps(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        deps = [node]

        # weight, bias, running_mean, running_var
        deps.extend(node.all_input_nodes[1:5])

        # All the users of batchnorm node must be getitem ops. batchnorm
        # returns a 3-element tuple. Each user must only access the first
        # element of the tuple.
        if [
            (user.target == operator.getitem and user.args[1] == 0)
            for user in node.users
        ].count(False):
            return []

        deps.extend(list(node.users.keys()))
        return deps

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class LayerNormConfig(XNNPartitionerConfig):
    """
    aten.native_layer_norm over the last dim with static weight and bias (f32 and bf16).

    native_layer_norm returns (output, mean, rstd); only output 0 may be used, because
    the XNNPACK layer norm node produces just the normalized tensor. The node must have
    exactly one user, that getitem.
    """

    target_name = "native_layer_norm.default"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        if not self.check_common_constraints(node, ep):
            return False

        # args: (input, normalized_shape, weight, bias, eps)
        if len(node.args) != 5:
            why(node, "expected (input, normalized_shape, weight, bias, eps)")
            return False
        input_node, normalized_shape, weight, bias, _eps = node.args

        input_val = input_node.meta.get("val")  # pyre-ignore
        if input_val is None or input_val.dtype not in (torch.float32, torch.bfloat16):
            why(node, "layer norm input must be float32 or bfloat16")
            return False

        # XNNPACK normalizes the last dim only.
        if (
            len(normalized_shape) != 1  # pyre-ignore
            or input_val.dim() < 1
            or normalized_shape[0] != input_val.shape[-1]
        ):
            why(node, f"only normalization over the last dim is supported, got {normalized_shape}")
            return False

        # Static weight and bias of the input dtype.
        for name, arg in (("weight", weight), ("bias", bias)):
            if not isinstance(arg, torch.fx.Node) or not is_param_node(ep, arg):
                why(node, f"layer norm {name} must be a static parameter")
                return False
            val = arg.meta.get("val")
            if val is None or val.dtype != input_val.dtype or val.dim() != 1:
                why(node, f"layer norm {name} must be 1D and of the input dtype")
                return False

        # Only the first output is produced by the XNNPACK node: the one user must be getitem(0).
        users = list(node.users)
        if not (
            len(users) == 1
            and users[0].target == operator.getitem
            and users[0].args[1] == 0
        ):
            why(node, "layer norm mean/rstd outputs are not supported")
            return False

        return True

    def get_node_and_deps(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        # node, weight, bias and the getitem(0) users
        return [node, node.args[2], node.args[3]] + list(node.users.keys())  # pyre-ignore

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class MaxDimConfig(XNNPartitionerConfig):
    target_name = "max.dim"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        # We support max_dim as long as we don't return indices
        supported_dtypes = {torch.float32, torch.float16, torch.int8, torch.qint8}
        node_val = node.meta.get("val")
        output_0 = node_val[0]

        input_node = node.all_input_nodes[0]
        if len(input_node.meta.get("val").shape) != 4:
            why(node, f"Unsupported input rank {input_node.meta.get('val').shape}")
            return False
        # Don't check indicies dtype
        if output_0.dtype not in supported_dtypes:
            why(node, f"Unsupported output dtype {output_0.dtype}")
            return False

        max_input = node.all_input_nodes[0]
        if max_input.meta.get("val").dtype not in supported_dtypes:
            why(node, f"Unsupported input dtype {max_input.meta.get('val').dtype}")
            return False

        # Make sure that all users are getitems of the first output
        for user in node.users:
            if not (user.target == operator.getitem and user.args[1] == 0):
                why(node, "Unsupported user of max.dim")
                return False

        return True

    def get_node_and_deps(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        getitems = list(node.users)

        return [node] + getitems

    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        return None

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class PreluConfig(XNNPartitionerConfig):
    target_name = "prelu.default"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        if not self.check_common_constraints(node, ep):
            return False

        weight = node.all_input_nodes[1]
        is_param = is_param_node(ep, weight)
        if not is_param:
            why(node, "Prelu weight must be a parameter")
            return False
        return True

    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        return torch.ops.aten.prelu.default

    def get_node_and_deps(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        weight = node.all_input_nodes[1]

        return [node, weight]

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]
