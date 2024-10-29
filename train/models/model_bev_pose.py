import torch
import roma
from typing import Optional
from torch import nn
from torch import Tensor
from .layers.block import Block
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, add_self_loops
import torchvision
from .layers.attention import MemEffAttention, Attention
from .model_bev_cnn import MultiUp
import math
from pathlib import Path

# from models.model_localization import MessEdgeConv


def round_to_multiple(num, mult):
    if num == 0:
        return num

    remainder = num % mult
    if remainder == 0:
        return num

    return num + mult - remainder


def linspace_mult(start, end, num, mult):
    ls = torch.linspace(start, end, num, dtype=int).tolist()
    return [ls[0]] + [round_to_multiple(f, mult) for f in ls[1:-1]] + [ls[-1]]


class PyramidTransformer(nn.Module):
    # def __init__(self, dims, sequences, attn_cls=MemEffAttention):
    def __init__(self, dims, sequences, attn_cls=Attention):
        super().__init__()
        assert len(dims) == len(sequences)
        self.sequences = sequences
        modules = []

        prev_dim = dims[0]
        for dim, seq in zip(dims, self.sequences):
            layer_modules = []
            if prev_dim != dim:
                layer_modules.append(nn.Linear(prev_dim, dim))

            layer_modules += [
                Block(
                    dim=dim,
                    num_heads=12,
                    mlp_ratio=4.0,
                    drop=0.2,
                    attn_drop=0.2,
                    attn_class=attn_cls,
                ),
                torch.nn.LayerNorm(dim, eps=1e-6),
            ]

            modules.append(nn.Sequential(*layer_modules))
            prev_dim = dim

        self.blocks = nn.ModuleList(modules)

    def forward(self, x: Tensor) -> Tensor:
        for block, seq in zip(self.blocks, self.sequences):
            x = block(x[:, :seq])
        return x


# trans = PyramidTransformer([96, 48, 24], [128, 64, 32])
# trans(torch.zeros(1, 128, 96))
# breakpoint()


class PoseEdgeConv(torch_geometric.nn.MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        seq_len,
        activation=torch.nn.LeakyReLU(inplace=True),
        **kwargs,
    ):
        super().__init__(**kwargs, node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_len * 2, self.in_channels))
        torch.nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.aggregator = nn.Sequential(
            PyramidTransformer(
                linspace_mult(self.in_channels, self.out_channels, 8, 48),
                [seq_len * 2] * 4 + linspace_mult(seq_len * 2, 32, 4, 8),
            ),
        )

    def message(self, x_i, x_j):
        # x_i is sink, x_j is source
        x = torch.cat([x_i, x_j], dim=1) + self.pos_embedding
        aggr = self.aggregator(x)
        return aggr

    def forward(self, x, edge_index):
        msg = self.message(x[edge_index[1]], x[edge_index[0]])
        return msg  # [:, 0]


class BEVEdgeConv(torch_geometric.nn.MessagePassing):
    def __init__(
        self,
        in_channels,
        seq_len,
        out_channels,
        activation=torch.nn.LeakyReLU(inplace=True),
        **kwargs,
    ):
        super().__init__(**kwargs, node_dim=0)

        self.in_channels = in_channels
        self.in_seq_len = seq_len
        self.out_channels = out_channels

        self.pos_embedding = nn.Parameter(
            torch.empty(1, self.in_seq_len * 2, self.in_channels)
        )

        torch.nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        pose_emb_size = 48
        self.pose_embedding = torch.nn.Sequential(
            torch.nn.Linear(17, pose_emb_size),
            # torch.nn.Linear(8, pose_emb_size),
            torch.nn.LayerNorm(pose_emb_size, eps=1e-6),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(pose_emb_size, pose_emb_size),
            torch.nn.LayerNorm(pose_emb_size, eps=1e-6),
        )
        self.aggregator = nn.Sequential(
            PyramidTransformer(
                linspace_mult(
                    self.in_channels + pose_emb_size, self.out_channels, 8, 48
                ),
                linspace_mult(self.in_seq_len * 2, 32, 8, 8),
            ),
        )

    def message(self, x_i, x_j, edge_attr):
        x = torch.cat([x_i, x_j], dim=1) + self.pos_embedding
        pose = self.pose_embedding(edge_attr).unsqueeze(1).repeat(1, x.shape[1], 1)
        x_p = torch.cat([x, pose], dim=2)
        aggr = self.aggregator(x_p)
        return aggr

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)


class RelPoseEdgeTransform(BaseTransform):
    r"""Saves the polar coordinates of linked nodes in its edge attributes
    (functional name: :obj:`polar`).

    Args:
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """

    def __init__(
        self,
        dropout_features: float = 1.0,
        cat: bool = True,
    ):
        self.cat = cat
        self.dropout_features = dropout_features

    def __call__(self, data: Data) -> Data:
        (row, col), pos, rot, pseudo = (
            data.edge_index,
            data.pos,
            data.rot,
            data.edge_attr,
        )

        angle_sink = rot[col]
        angle_source = rot[row]

        rel_pos = pos[col] - pos[row]
        rel_pos_source = roma.RotationUnitQuat(
            linear=roma.quat_inverse(angle_source)
        ).apply(rel_pos)
        angle_between_nodes = roma.quat_product(
            angle_source, roma.quat_inverse(angle_sink)
        )

        # dist += torch.randn(*dist.shape, device=dist.device) * 0.2
        # angle_to_other += torch.randn(*angle_to_other.shape, device=angle_to_other.device) * 0.2
        # angle_between += torch.randn(*angle_between.shape, device=dist.device) * 0.05

        edge_attr = torch.cat(
            [
                rel_pos_source,
                angle_between_nodes,
            ],
            dim=-1,
        )
        edge_mask = (
            torch.rand(row.size(0), device=data.edge_index.device)
            <= self.dropout_features
        )
        edge_attr[edge_mask] = 0.0
        edge_attr = torch.cat([edge_attr, edge_mask.unsqueeze(1)], dim=1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, edge_attr.type_as(pos)], dim=-1)
        else:
            data.edge_attr = edge_attr

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cat={self.cat})"


class BEVGNNModel(nn.Module):
    def __init__(
        self,
        comm_range: float,
        gnn_in_channels: int,
        gnn_in_seq_len: int,
        pose_gnn_out_channels: int,
        bev_gnn_out_channels: int,
        dec_out_channels: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.comm_range = comm_range
        self.gnn_in_seq_len = gnn_in_seq_len
        self.bev_gnn_out_channels = bev_gnn_out_channels

        self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        for param in self.encoder.parameters():
            param.requires_grad = False

        enc_out_seq_len = int((224 / self.encoder.patch_size) ** 2)
        self.enc_post = nn.Sequential(
            PyramidTransformer(
                linspace_mult(self.encoder.num_features, gnn_in_channels, 6, 48),
                linspace_mult(enc_out_seq_len, self.gnn_in_seq_len, 6, 8),
            ),
        )
        self.pose_gnn_a = PoseEdgeConv(
            gnn_in_channels, pose_gnn_out_channels, self.gnn_in_seq_len, aggr="add"
        )
        self.pose_gnn_a_decoder_post = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(pose_gnn_out_channels, 17),
        )

        self.pose_gnn_b = PoseEdgeConv(
            pose_gnn_out_channels,
            pose_gnn_out_channels,
            32,
            aggr="add",
        )
        self.pose_gnn_b_decoder_post = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(pose_gnn_out_channels, 17),
        )

        self.bev_gnn = BEVEdgeConv(
            gnn_in_channels, self.gnn_in_seq_len, bev_gnn_out_channels
        )
        self.bev_decoder = MultiUp(
            linspace_mult(bev_gnn_out_channels, 16, 7, 8), dec_out_channels
        )

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        bs, n_nodes = input["img_norm"].shape[:2]
        img_flat = input["img_norm"].flatten(0, 1)
        x = self.encoder.forward_features(img_flat)["x_norm_patchtokens"]
        x = self.enc_post(x)

        graphs = torch_geometric.data.Batch()
        graphs.batch = torch.repeat_interleave(torch.arange(bs), n_nodes, dim=0).to(
            input["img_norm"].device
        )
        graphs.x = x[:, : self.gnn_in_seq_len]
        graphs.pos = input["pos"].flatten(0, 1)
        graphs.rot = input["rot_quat"].flatten(0, 1)
        edge_index_pose = torch_geometric.nn.pool.radius_graph(
            graphs.pos, batch=graphs.batch, r=self.comm_range, loop=False
        )
        graphs.edge_index = edge_index_pose

        z_p = self.pose_gnn_a(graphs.x, graphs.edge_index)
        edge_preds = self.pose_gnn_a_decoder_post(z_p[:, 0])

        graphs.edge_index, edge_mask = dropout_edge(graphs.edge_index, p=0.3)

        # graphs = RelPoseEdgeTransform(dropout_features=0.0)(graphs) # when testing for providing all bev edges

        # edge_preds[:] = 0.0 # When testing for no edge preds
        graphs.edge_attr = edge_preds[edge_mask]

        # graphs.edge_attr = torch.cat([graphs.edge_attr, edge_preds[edge_mask]], dim=1)
        # Edge dropout to be able to deal with variable number of nodes
        edge_index_self, edge_attr_self = add_self_loops(
            graphs.edge_index, graphs.edge_attr, fill_value=0.0
        )
        z = self.bev_gnn(graphs.x, edge_index_self, edge_attr_self)

        z_p_b = self.pose_gnn_b.message(z[edge_index_pose[0]], z_p)
        edge_preds_b = self.pose_gnn_b_decoder_post(z_p_b)[:, 0]

        edge_preds_proc = {
            "pos": edge_preds_b[:, 0:3],
            "pos_var": edge_preds_b[:, 3:6].exp(),
            "rot": roma.symmatrixvec_to_unitquat(
                edge_preds_b[:, 6:16].to(torch.float)
            ).to(edge_preds_b.dtype),
            "rot_var": edge_preds_b[:, 16:17].exp(),
        }

        bev_in = z[:, 0].view(z.shape[0], self.bev_gnn_out_channels, 1, 1)
        dec = self.bev_decoder(bev_in)
        _, _, w, h = dec.shape
        o_w, o_h = 60, 60
        result_crop = torchvision.transforms.functional.crop(
            dec, int(w / 2 - o_w / 2), int(h / 2 - o_h / 2), o_w, o_h
        )
        bev_nodes = result_crop.view(bs, n_nodes, *result_crop.shape[1:])

        return (edge_preds_proc, bev_nodes), (edge_index_pose, graphs.batch)


class BEVGNNModelLoaded(nn.Module):
    def __init__(
        self,
        comm_range: float,
        gnn_in_channels: int,
        gnn_in_seq_len: int,
        pose_gnn_out_channels: int,
        bev_gnn_out_channels: int,
        dec_out_channels: int,
        **kwargs,
    ) -> None:
        # import torch_tensorrt

        super().__init__()
        self.comm_range = comm_range
        self.gnn_in_channels = gnn_in_channels
        self.gnn_in_seq_len = gnn_in_seq_len
        self.gnn_out_channels = pose_gnn_out_channels

        self.param_empty = nn.Parameter(torch.empty(1, 1))

        out_base_dir = Path("./exported_models")
        # model_prefix = "vyed3k6w"
        model_prefix = "oyu1brtpe18"
        self.dtype = torch.half
        type_prefix = str(self.dtype).split(".")[1]
        model_kind = "jit"

        self.enc = torch.jit.load(
            out_base_dir / f"{model_prefix}_{type_prefix}_{model_kind}_enc.ts"
        )
        self.msg = torch.jit.load(
            out_base_dir / f"{model_prefix}_{type_prefix}_{model_kind}_msg.ts"
        )
        self.post = torch.jit.load(out_base_dir / f"{model_prefix}_float32_jit_post.ts")
        self.limits = torch.load(f"export_quant_data_{type_prefix}/enc_limits.pt")

    def map_to_int(self, x):
        x_clamp = torch.max(x, self.limits[:, :, 0])
        x_clamp = torch.min(x_clamp, self.limits[:, :, 1])
        out_max = 255
        out_min = 0
        mapped = (x_clamp - self.limits[:, :, 0]) * (out_max - out_min) / (
            self.limits[:, :, 1] - self.limits[:, :, 0]
        ) + out_min
        return mapped.to(torch.uint8)

    def map_from_int(self, x):
        in_min = 0
        in_max = 255
        mapped = (x - in_min) * (self.limits[:, :, 1] - self.limits[:, :, 0]) / (
            in_max - in_min
        ) + self.limits[:, :, 0]
        return mapped.to(self.dtype)

    def forward(self, input: Tensor, **kwargs):
        batch_size = input["img_raw"].shape[0]
        n_agents = input["img_raw"].shape[1]
        graphs = torch_geometric.data.Batch()
        graphs.batch = torch.repeat_interleave(
            torch.arange(batch_size), n_agents, dim=0
        ).to(input["img_raw"].device)
        graphs.pos = input["pos"][:, :, :2].reshape(batch_size * n_agents, 2)
        graphs.rot = input["rot"][:, :, 1].reshape(batch_size * n_agents, 1)
        graphs.edge_index = torch_geometric.nn.pool.radius_graph(
            graphs.pos, batch=graphs.batch, r=self.comm_range, loop=False
        )
        breakpoint()

        self.limits = self.limits.to(input["img_raw"].device)
        out = []
        img_flat = input["img_raw"].flatten(0, 1).to(self.dtype)
        for i, j in zip(graphs.edge_index[1], graphs.edge_index[0]):
            img_i = img_flat[i].unsqueeze(0)
            img_j = img_flat[j].unsqueeze(0)
            breakpoint()
            x_i = self.enc(img_i.to(self.dtype))
            x_j = self.enc(img_j.to(self.dtype))

            x_i_remapped = x_i  # self.map_from_int(self.map_to_int(x_i))
            x_j_remapped = x_j  # self.map_from_int(self.map_to_int(x_j))

            x_i, x_j = x_i_remapped, x_j_remapped
            aggr = self.msg(x_i.to(self.dtype), x_j.to(self.dtype))
            out.append(self.post(aggr.to(torch.float)))
        edge_preds = {
            "pos": torch.concatenate([o[0] for o in out], dim=0),
            "pos_var": torch.concatenate([o[1] for o in out], dim=0),
            "rot": torch.concatenate([o[2] for o in out], dim=0),
            "rot_var": torch.concatenate([o[3] for o in out], dim=0),
        }
        node_preds = torch.zeros(
            batch_size, n_agents, 1, 60, 60, device=input["img_raw"].device
        )
        return (edge_preds, node_preds), (graphs.edge_index, graphs.batch)


def test():
    out_channels = 1
    bs = 2
    n_nodes = 3

    dev = torch.device("cpu")
    model = BEVGNNModel(2.0, 24, 64, 384, 384, out_channels).to(dev)
    inp = {
        "img_norm": torch.rand(bs, n_nodes, 3, 224, 224, device=dev),
        "pos": torch.rand(bs, n_nodes, 3, device=dev),
        "rot_quat": torch.rand(bs, n_nodes, 4, device=dev),
    }
    (edge_preds, bev_nodes), (edge_index, edge_batch) = model(inp)

    edge_size = bs * (n_nodes**2 - n_nodes)
    assert edge_batch.shape == (bs * n_nodes,)
    assert edge_index.shape == (2, edge_size)
    assert bev_nodes.shape == (bs, n_nodes, out_channels, 60, 60)
    assert all(key in edge_preds for key in ["pos", "pos_var", "rot", "rot_var"])
    assert edge_preds["pos"].shape == (edge_size, 3)
    assert edge_preds["pos_var"].shape == (edge_size, 3)
    assert edge_preds["rot"].shape == (edge_size, 4)
    assert edge_preds["rot_var"].shape == (edge_size, 1)


if __name__ == "__main__":
    test()
