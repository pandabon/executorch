# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, cast

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNGraph,
    XNNLayerNorm,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import check_or_raise, get_input_node


@register_node_visitor
class LayerNormVisitor(NodeVisitor):
    """
    aten.native_layer_norm(input, normalized_shape, weight, bias, eps) over the last
    dim, with static weight and bias. The node returns (output, mean, rstd); the
    partitioner only accepts it when its single user is getitem(0), so only that
    tensor is serialized.
    """

    target = "aten.native_layer_norm.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        # node.meta["val"] is a tuple (output, mean, rstd). Only the output is
        # serialized: the partitioner accepts exactly one user, a getitem(0), so the
        # XNNPACK output tensor is defined from that getitem node (it decides whether
        # the tensor is an external output of the delegate, which the layer norm
        # node itself cannot, as its only user lives inside the partition).
        users = list(node.users.keys())
        check_or_raise(
            len(users) == 1
            and users[0].target.__name__ == "getitem"  # pyre-ignore
            and users[0].args[1] == 0,
            "XNNPACK layer norm only supports a single getitem(0) user",
        )
        output_node = users[0]
        for inp in (get_input_node(node, 0), get_input_node(node, 2), get_input_node(node, 3)):
            self.define_tensor(inp, xnn_graph, vals_to_ids)
        self.define_tensor(output_node, xnn_graph, vals_to_ids)
        output_id = vals_to_ids[output_node]
        vals_to_ids[node] = output_id

        input_id = vals_to_ids[get_input_node(node, 0)]
        scale_id = vals_to_ids[get_input_node(node, 2)]
        bias_id = vals_to_ids[get_input_node(node, 3)]
        epsilon = cast(float, node.args[4])

        ser_node = XNode(
            xnode_union=XNNLayerNorm(
                epsilon=epsilon,
                input_id=input_id,
                scale_id=scale_id,
                bias_id=bias_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
