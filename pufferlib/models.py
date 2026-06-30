from dataclasses import dataclass

import numpy as np
from typing import Dict
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

class Policy(nn.Module):
    def __init__(self, encoder, decoder, network):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.network = network

        # sig = inspect.signature(encoder.forward)
        # self.accepts_substep = (
        #     "substep" in sig.parameters
        #     or any(p.kind == inspect.Parameter.VAR_KEYWORD
        #            for p in sig.parameters.values())
        # )

    def initial_state(self, batch_size, device):
        return self.network.initial_state(batch_size, device)

    def forward_eval(self, x, state):
        with nvtx_range("encoder_eval"):
            h = self.encoder(x)
        with nvtx_range("network_eval"):
            h, state = self.network.forward_eval(h, state)
        with nvtx_range("decoder_eval"):
            logits, values = self.decoder(h)

        return logits, values, state

    def forward(self, x):
        B, TT = x.shape[:2]
        with nvtx_range("encoder_train"):
            h = self.encoder(x.reshape(B*TT, *x.shape[2:]))
        with nvtx_range("network_train"):
            h = self.network.forward_train(h.reshape(B, TT, -1) if isinstance(h, torch.Tensor) else h)
        with nvtx_range("decoder_train"):
            logits, values = self.decoder(h.reshape(B*TT, -1) if isinstance(h, torch.Tensor) else h)

        return logits, values.reshape(B, TT) if values is not None else None

class DefaultEncoder(nn.Module):
    def __init__(self, obs_size, hidden_size=128):
        super().__init__()
        self.encoder = nn.Linear(obs_size, hidden_size)

    def forward(self, observations):
        return self.encoder(observations.view(observations.shape[0], -1).float())

class DefaultDecoder(nn.Module):
    def __init__(self, nvec, hidden_size=128):
        super().__init__()
        self.nvec = tuple(nvec)
        self.is_continuous = sum(nvec) == len(nvec)

        if self.is_continuous:
            num_atns = len(nvec)
            self.decoder_mean = nn.Linear(hidden_size, num_atns)
            self.decoder_logstd = nn.Parameter(torch.zeros(1, num_atns))
        else:
            self.decoder = nn.Linear(hidden_size, int(np.sum(nvec)))

        self.value_function = nn.Linear(hidden_size, 1)

    def forward(self, hidden):
        if self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            logits = torch.distributions.Normal(mean, torch.exp(logstd))
        else:
            logits = self.decoder(hidden)
            if len(self.nvec) > 1:
                logits = logits.split(self.nvec, dim=1)

        values = self.value_function(hidden)
        return logits, values

class MLP(nn.Module):
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def initial_state(self, batch_size, device):
        return ()

    def forward_eval(self, h, state):
        return self.net(h), state

    def forward_train(self, h):
        return self.net(h)

class MinGRU(nn.Module):
    # https://arxiv.org/abs/2410.01201v1
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, 3 * hidden_size, bias=False) for _ in range(num_layers)
        ])

    def _g(self, x):
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    def _log_g(self, x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    def _highway(self, x, out, proj):
        g = proj.sigmoid()
        return g * out + (1.0 - g) * x

    def _heinsen_scan(self, log_coeffs, log_values):
        a_star = log_coeffs.cumsum(dim=1)
        return (a_star + (log_values - a_star).logcumsumexp(dim=1)).exp()

    def initial_state(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),)

    def forward_eval(self, h, state):
        state = state[0]
        assert state.shape[1] == h.shape[0]
        h = h.unsqueeze(1)
        state_out = []
        for i in range(self.num_layers):
            hidden, gate, proj = self.layers[i](h).chunk(3, dim=-1)
            out = torch.lerp(state[i:i+1].transpose(0, 1), self._g(hidden), gate.sigmoid())
            h = self._highway(h, out, proj)
            state_out.append(out[:, -1:])
        return h.squeeze(1), (torch.stack(state_out, 0).squeeze(2),)

    def forward_train(self, h):
        T = h.shape[1]
        for i in range(self.num_layers):
            hidden, gate, proj = self.layers[i](h).chunk(3, dim=-1)
            log_coeffs = -F.softplus(gate)
            log_values = -F.softplus(-gate) + self._log_g(hidden)
            out = self._heinsen_scan(log_coeffs, log_values)[:, -T:]
            h = self._highway(h, out, proj)
        return h

class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.cell = nn.ModuleList([torch.nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)])

        for i in range(num_layers):
            cell = self.cell[i]
            w_ih = getattr(self.lstm, f'weight_ih_l{i}')
            w_hh = getattr(self.lstm, f'weight_hh_l{i}')
            b_ih = getattr(self.lstm, f'bias_ih_l{i}')
            b_hh = getattr(self.lstm, f'bias_hh_l{i}')
            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()
            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward_eval(self, h, state):
        assert state[0].shape[1] == state[1].shape[1] == h.shape[0]
        lstm_h, lstm_c = state
        for i in range(self.num_layers):
            h, c = self.cell[i](h, (lstm_h[i], lstm_c[i]))
            lstm_h[i] = h
            lstm_c[i] = c
        return h, (lstm_h, lstm_c)

    def forward_train(self, h):
        # h: [B, T, H]
        h = h.transpose(0, 1)
        h, _ = self.lstm(h)
        return h.transpose(0, 1)

class GRU(nn.Module):
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        self.cell = nn.ModuleList([torch.nn.GRUCell(hidden_size, hidden_size) for _ in range(num_layers)])
        self.norm = torch.nn.RMSNorm(hidden_size)

        for i in range(num_layers):
            cell = self.cell[i]
            w_ih = getattr(self.gru, f'weight_ih_l{i}')
            w_hh = getattr(self.gru, f'weight_hh_l{i}')
            b_ih = getattr(self.gru, f'bias_ih_l{i}')
            b_hh = getattr(self.gru, f'bias_hh_l{i}')
            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()
            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h,)

    def forward_eval(self, h, state):
        assert state[0].shape[1] == h.shape[0]
        state = state[0]
        for i in range(self.num_layers):
            h_in = h
            h = self.cell[i](h, state[i])
            state[i] = h
            h = h + h_in
            h = self.norm(h)
        return h, (state,)

    def forward_train(self, h):
        # h: [B, T, H]
        h = h.transpose(0, 1)
        h_in = h
        h, _ = self.gru(h)
        h = h + h_in
        h = self.norm(h)
        return h.transpose(0, 1)

class NatureEncoder(nn.Module):
    '''NatureCNN encoder (Mnih et al. 2015). Returns [batch, hidden_size].'''
    def __init__(self, env, hidden_size=512, framestack=1, flat_size=64*7*7,
            channels_last=False, downsample=1, **kwargs):
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample
        self.network = nn.Sequential(
            nn.Conv2d(framestack, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flat_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.network(observations.float() / 255.0)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(input_shape[0], out_channels, 3, padding=1)
        self.res_block0 = ResidualBlock(out_channels)
        self.res_block1 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class ImpalaEncoder(nn.Module):
    '''IMPALA ResNet encoder (Espeholt et al. 2018). Returns [batch, hidden_size].'''
    def __init__(self, env, hidden_size=256, cnn_width=16, **kwargs):
        super().__init__()
        h, w, c = env.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [cnn_width, 2*cnn_width, 2*cnn_width]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(shape[0] * shape[1] * shape[2], hidden_size),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)

    def forward(self, observations):
        return self.network(observations.permute(0, 3, 1, 2).float() / 255.0)

# ------------------------------- QUAD MESHING -------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """CleanRL's default layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class _MLP(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            hidden_dim=128,
            num_layers=2,
            std=np.sqrt(2),
            bias_const=0.0,
            activation=nn.SiLU,
            final_activation=False,
    ):
        super().__init__()
        layers = []
        d = in_dim

        for _ in range(num_layers - 1):
            linear = layer_init(nn.Linear(d, hidden_dim), std, bias_const)
            layers += [linear, activation()]
            d = hidden_dim

        final = layer_init(nn.Linear(d, out_dim), std, bias_const)
        layers.append(final)

        if final_activation:
            layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SE2AngleMessagePassing(nn.Module):
    """
    Implements:

        m_ij = 1 / (|N_j| - 1) sum_{k in N_j \\ i}
               phi_m(h_i, h_j, h_k, a_ij, a_jk, cos angle ijk, sin angle ijk)

        h_i' = phi_h(h_i, 1 / |N_i| sum_{j in N_i} m_ji)

    Sparse directed graph representation:
        edge_index[0] = source nodes
        edge_index[1] = target nodes

    For edge i -> j, neighbors k of j are outgoing edges j -> k.
    edge_index must be sorted by source node and edge_ptr must be the matching
    outgoing-edge CSR pointer.
    """

    def __init__(
        self,
        node_dim,
        out_dim,
        msg_dim,
        edge_cont_dim=0,
        edge_cat_cardinalities=None,
        edge_cat_emb_dim=16,
        hidden_dim=128,
        eps=1e-8,
    ):
        super().__init__()

        self.edge_cont_dim = edge_cont_dim
        self.edge_cat_cardinalities = edge_cat_cardinalities or []
        self.edge_cat_emb_dim = edge_cat_emb_dim
        self.eps = eps

        self.edge_cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, edge_cat_emb_dim)
            for cardinality in self.edge_cat_cardinalities
        ])

        edge_cat_total_dim = len(self.edge_cat_cardinalities) * edge_cat_emb_dim
        edge_feat_dim = edge_cont_dim + edge_cat_total_dim

        phi_m_in = (
            3 * node_dim      # h_i, h_j, h_k
            + 2 * edge_feat_dim  # a_ij, a_jk
            + 2              # cos, sin
        )

        self.phi_m = _MLP(phi_m_in, msg_dim, hidden_dim)
        self.phi_h = _MLP(node_dim + msg_dim, out_dim, hidden_dim)

    def encode_edges(self, edge_cont=None, edge_cat=None):
        parts = []

        if self.edge_cont_dim > 0:
            if edge_cont is None:
                raise ValueError("edge_cont must be provided.")
            parts.append(edge_cont)

        if self.edge_cat_cardinalities:
            if edge_cat is None:
                raise ValueError("edge_cat must be provided.")

            # edge_cat: [E, num_categorical_features]
            cat_embs = [
                emb(edge_cat[:, r])
                for r, emb in enumerate(self.edge_cat_embeddings)
            ]
            parts.append(torch.cat(cat_embs, dim=-1))

        if not parts:
            return None

        return torch.cat(parts, dim=-1)

    def forward(self, h, x, edge_index, edge_ptr, edge_cont=None, edge_cat=None):
        """
        Args:
            h:          [N, node_dim]
            x:          [N, 2] coordinates
            edge_index: [2, E], directed edges i -> j
            edge_ptr:   [N + 1] CSR pointers for outgoing edges. Must match edge_index source order.
            edge_cont:  [E, edge_cont_dim], optional continuous edge features
            edge_cat:   [E, num_edge_cat_features], optional categorical features

        Returns:
            h_out:      [N, out_dim]
        """

        device = h.device
        num_nodes = h.size(0)

        src = edge_index[0]  # i
        dst = edge_index[1]  # j
        num_edges = src.numel()

        edge_feat = self.encode_edges(edge_cont, edge_cat)

        if edge_ptr is None:
            raise ValueError("edge_ptr must be provided and must match source-sorted edge_index.")
        if edge_ptr.numel() != num_nodes + 1:
            raise ValueError("edge_ptr must have shape [num_nodes + 1].")
        out_deg = edge_ptr[1:] - edge_ptr[:-1]
        ptr = edge_ptr

        # For every edge i -> j, enumerate outgoing edges j -> k.
        center = dst
        counts = out_deg[center]  # |N_j|

        total_pairs = counts.sum().item()
        if total_pairs == 0:
            agg = torch.zeros(num_nodes, self.phi_m.net[-1].out_features, device=device)
            return self.phi_h(torch.cat([h, agg], dim=-1))

        edge_ij = torch.repeat_interleave(
            torch.arange(num_edges, device=device),
            counts,
        )

        repeated_centers = torch.repeat_interleave(center, counts)

        pair_start = torch.repeat_interleave(
            torch.cumsum(counts, dim=0) - counts,
            counts,
        )
        local_offset = torch.arange(total_pairs, device=device) - pair_start

        edge_jk_sorted_pos = ptr[repeated_centers] + local_offset
        edge_jk = edge_jk_sorted_pos

        # Remove k == i.
        i = src[edge_ij]
        j = dst[edge_ij]
        k = dst[edge_jk]

        mask = k != i

        edge_ij = edge_ij[mask]
        edge_jk = edge_jk[mask]
        i = i[mask]
        j = j[mask]
        k = k[mask]

        if edge_ij.numel() == 0:
            msg = torch.zeros(num_edges, self.phi_m.net[-1].out_features, device=device)
        else:
            xi, xj, xk = x[i], x[j], x[k]

            # angle ijk, centered at j
            v_ji = xi - xj
            v_jk = xk - xj

            n_ji = v_ji.norm(dim=-1).clamp_min(self.eps)
            n_jk = v_jk.norm(dim=-1).clamp_min(self.eps)

            cos = (v_ji * v_jk).sum(dim=-1, keepdim=True) / (n_ji * n_jk).unsqueeze(-1)

            # signed 2D sine; SE(2)-invariant, but not reflection-invariant
            cross = v_ji[:, 0] * v_jk[:, 1] - v_ji[:, 1] * v_jk[:, 0]
            sin = cross.unsqueeze(-1) / (n_ji * n_jk).unsqueeze(-1)

            parts = [
                h[i],
                h[j],
                h[k],
            ]

            if edge_feat is not None:
                parts += [
                    edge_feat[edge_ij],  # a_ij
                    edge_feat[edge_jk],  # a_jk
                ]

            parts += [cos, sin]

            triplet_input = torch.cat(parts, dim=-1)
            triplet_msg = self.phi_m(triplet_input)

            # Sum over k for each directed edge i -> j.
            msg = torch.zeros(
                num_edges,
                triplet_msg.size(-1),
                device=device,
                dtype=triplet_msg.dtype,
            )
            msg.index_add_(0, edge_ij, triplet_msg)

            # Divide by |N_j| - 1.
            denom = (out_deg[dst] - 1).clamp_min(1).to(msg.dtype).unsqueeze(-1)
            msg = msg / denom

        # m_ij is a message from node i to node j, so node j receives it here.
        receiver = dst
        agg = torch.zeros(
            num_nodes,
            msg.size(-1),
            device=device,
            dtype=msg.dtype,
        )
        agg.index_add_(0, receiver, msg)

        # Divide by |N_i|.
        node_denom = out_deg.clamp_min(1).to(agg.dtype).unsqueeze(-1)
        agg = agg / node_denom

        return self.phi_h(torch.cat([h, agg], dim=-1))


@dataclass
class CSRGraph:
    vertices: torch.Tensor       # [N, 2] vertex positions
    batch_offsets: torch.Tensor  # [B + 1] vertex offsets per batch
    edges: torch.Tensor          # [2, E] edge indices
    edge_features: torch.Tensor  # [E, f] edge features
    edge_ptr: torch.Tensor       # [N + 1] CSR pointer for outgoing edges


@dataclass
class CandidateTargets:
    source_idx: torch.Tensor            # [B] local source vertex index per batch item
    target_idx: torch.Tensor            # [Q] global frontier vertex index, or -1 for new candidates
    target_batch_offsets: torch.Tensor  # [B + 1] target offsets per batch
    target_batches: torch.Tensor        # [Q] batch index per target
    target_positions: torch.Tensor      # [Q, 2] existing vertex or new candidate position


# =============================================================================
# Triton kernels
# =============================================================================


@triton.jit
def _gf_init_offsets_kernel(
    vertex_batch_offsets,   # int64*, [B + 1]
    edge_ptr,               # int64*, [N_CAP + 1]
):
    tl.store(vertex_batch_offsets + 0, tl.full((), 0, tl.int64))
    tl.store(edge_ptr + 0, tl.full((), 0, tl.int64))


@triton.jit
def _gf_decode_frontier_sizes_kernel(
    obs,                    # uint8*, [B_full, obs_size]
    valid_batch_idx,        # int64*, [B]
    frontier_size_out,      # int64*, [B]
    OBS_SIZE: tl.constexpr,
    B: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = off < B

    physical_b = tl.load(valid_batch_idx + off, mask=mask, other=0)
    row = physical_b * OBS_SIZE

    # frontier_size at byte offsets 1..2, little-endian u16
    f_lo = tl.load(obs + row + 1, mask=mask, other=0).to(tl.uint32)
    f_hi = tl.load(obs + row + 2, mask=mask, other=0).to(tl.uint32)
    frontier_size = f_lo | (f_hi << 8)

    tl.store(frontier_size_out + off, frontier_size.to(tl.int64), mask=mask)


@triton.jit
def _gf_decode_vertices_kernel(
    obs,                    # uint8*, [B_full, obs_size]
    vertices,               # float32*, [N_CAP, 2]
    vertex_batch_offsets,   # int64*, [B + 1]
    valid_batch_idx,        # int64*, [B]
    OBS_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    block_i = tl.program_id(1)

    i = block_i * BLOCK_N + tl.arange(0, BLOCK_N)

    physical_b = tl.load(valid_batch_idx + b)
    row = physical_b * OBS_SIZE

    # frontier_size at byte offsets 1..2, little-endian u16
    f_lo = tl.load(obs + row + 1).to(tl.uint32)
    f_hi = tl.load(obs + row + 2).to(tl.uint32)
    frontier_size = f_lo | (f_hi << 8)

    valid = i < frontier_size

    vertex_batch_offset = tl.load(vertex_batch_offsets + b)
    global_i = vertex_batch_offset + i

    # Vertices start at byte offset 5, each vertex is two little-endian f32s.
    base = row + 5 + i * 8

    x_u32 = (
        tl.load(obs + base + 0, mask=valid, other=0).to(tl.uint32)
        | (tl.load(obs + base + 1, mask=valid, other=0).to(tl.uint32) << 8)
        | (tl.load(obs + base + 2, mask=valid, other=0).to(tl.uint32) << 16)
        | (tl.load(obs + base + 3, mask=valid, other=0).to(tl.uint32) << 24)
    )

    y_u32 = (
        tl.load(obs + base + 4, mask=valid, other=0).to(tl.uint32)
        | (tl.load(obs + base + 5, mask=valid, other=0).to(tl.uint32) << 8)
        | (tl.load(obs + base + 6, mask=valid, other=0).to(tl.uint32) << 16)
        | (tl.load(obs + base + 7, mask=valid, other=0).to(tl.uint32) << 24)
    )

    tl.store(
        vertices + global_i * 2 + 0,
        x_u32.to(tl.float32, bitcast=True),
        mask=valid,
    )
    tl.store(
        vertices + global_i * 2 + 1,
        y_u32.to(tl.float32, bitcast=True),
        mask=valid,
    )


@triton.jit
def _gf_count_neighbors_kernel(
    obs,                    # uint8*, [B_full, obs_size]
    neighbor_count,         # int64*, [N_CAP]
    vertex_batch_offsets,   # int64*, [B + 1]
    valid_batch_idx,        # int64*, [B]
    OBS_SIZE: tl.constexpr,
    D: tl.constexpr,
    F_CAP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    block_i = tl.program_id(1)

    i = block_i * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    d = tl.arange(0, BLOCK_D)[None, :]

    physical_b = tl.load(valid_batch_idx + b)
    row = physical_b * OBS_SIZE

    f_lo = tl.load(obs + row + 1).to(tl.uint32)
    f_hi = tl.load(obs + row + 2).to(tl.uint32)
    frontier_size = f_lo | (f_hi << 8)

    valid_vertex = i < frontier_size
    valid_i_cap = i < F_CAP
    valid_d = d < D

    vertex_batch_offset = tl.load(vertex_batch_offsets + b)
    global_i = vertex_batch_offset + i

    neighbor_base = row + 5 + frontier_size * 8
    nb = neighbor_base + (i * D + d) * 2

    mask = valid_i_cap & valid_vertex & valid_d

    lo = tl.load(obs + nb + 0, mask=mask, other=255).to(tl.uint32)
    hi = tl.load(obs + nb + 1, mask=mask, other=255).to(tl.uint32)

    local_neighbor = lo | (hi << 8)
    valid_edge = mask & (local_neighbor != 65535)

    count = tl.sum(valid_edge.to(tl.int64), axis=1)

    ii = block_i * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_store = ii < frontier_size

    global_ii = tl.load(vertex_batch_offsets + b) + ii

    tl.store(
        neighbor_count + global_ii,
        count,
        mask=valid_store,
    )


@triton.jit
def _gf_write_sizes_kernel(
    vertex_batch_offsets,   # int64*, [B + 1]
    edge_ptr,               # int64*, [N_CAP + 1]
    sizes_dev,              # int64*, [2], sizes_dev[0] = N, sizes_dev[1] = E
    B: tl.constexpr,
):
    n = tl.load(vertex_batch_offsets + B)
    e = tl.load(edge_ptr + n)

    tl.store(sizes_dev + 0, n)
    tl.store(sizes_dev + 1, e)


@triton.jit
def _gf_scatter_edges_kernel(
    obs,                    # uint8*, [B_full, obs_size]
    edges_flat,             # int64*, capacity [2 * E_CAP]
    edge_features_flat,     # bool*, capacity [2 * E_CAP]
    edge_ptr,               # int64*, [N_CAP + 1]
    vertex_batch_offsets,   # int64*, [B + 1]
    sizes_dev,              # int64*, [2], sizes_dev[1] = E actual
    valid_batch_idx,        # int64*, [B]
    OBS_SIZE: tl.constexpr,
    D: tl.constexpr,
    F_CAP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    block_i = tl.program_id(1)

    ii = block_i * BLOCK_N + tl.arange(0, BLOCK_N)
    dd = tl.arange(0, BLOCK_D)

    # Shape [BLOCK_N, 1]
    i = ii[:, None]

    # Shape [1, BLOCK_D]
    d = dd[None, :]

    physical_b = tl.load(valid_batch_idx + b)
    row = physical_b * OBS_SIZE

    f_lo = tl.load(obs + row + 1).to(tl.uint32)
    f_hi = tl.load(obs + row + 2).to(tl.uint32)
    frontier_size = f_lo | (f_hi << 8)

    valid_i_cap = i < F_CAP
    valid_vertex = i < frontier_size
    valid_d = d < D

    vertex_batch_offset = tl.load(vertex_batch_offsets + b)

    # Shape [BLOCK_N]
    global_src_vec = vertex_batch_offset + ii

    neighbor_base = row + 5 + frontier_size * 8
    face_base = neighbor_base + frontier_size * D * 2

    nb = neighbor_base + (i * D + d) * 2

    mask = valid_i_cap & valid_vertex & valid_d

    lo = tl.load(obs + nb + 0, mask=mask, other=255).to(tl.uint32)
    hi = tl.load(obs + nb + 1, mask=mask, other=255).to(tl.uint32)

    local_neighbor = lo | (hi << 8)
    valid_edge = mask & (local_neighbor != 65535)

    face = tl.load(
        obs + face_base + i * D + d,
        mask=mask,
        other=0,
    ).to(tl.uint32)

    # Per-row rank within each vertex's neighbor list.
    # Layout is [BLOCK_N, BLOCK_D], so the D dimension is axis=1.
    rank = tl.cumsum(valid_edge.to(tl.int64), axis=1) - 1

    # Shape [BLOCK_N, 1]
    global_src = global_src_vec[:, None]

    # edge_ptr is indexed by compact global vertex id.
    out_base = tl.load(edge_ptr + global_src_vec, mask=ii < frontier_size, other=0)
    out = out_base[:, None] + rank

    global_dst = vertex_batch_offset + local_neighbor.to(tl.int64)

    E_actual = tl.load(sizes_dev + 1)

    incident_ccw = ((face >> 1) & 1) != 0
    incident_cw = (face & 1) != 0

    tl.store(edges_flat + out, global_src, mask=valid_edge)
    tl.store(edges_flat + E_actual + out, global_dst, mask=valid_edge)

    tl.store(edge_features_flat + out * 2 + 0, incident_ccw, mask=valid_edge)
    tl.store(edge_features_flat + out * 2 + 1, incident_cw, mask=valid_edge)


@triton.jit
def _cand_init_offsets_kernel(
    target_batch_offsets,  # int64*, [B + 1]
):
    tl.store(target_batch_offsets + 0, tl.full((), 0, tl.int64))


@triton.jit
def _cand_decode_counts_kernel(
    obs,                    # uint8*, [B_full, obs_size]
    valid_batch_idx,        # int64*, [B]
    vertex_batch_offsets,   # int64*, [B + 1]
    source_idx,             # int64*, [B]
    candidate_count,        # int64*, [B]
    valid_node_count,       # int64*, [B]
    target_count,           # int64*, [B]
    node_validity,          # bool*, [N_CAP]
    OBS_SIZE: tl.constexpr,
    D: tl.constexpr,
    F_CAP: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.arange(0, BLOCK_F)

    physical_b = tl.load(valid_batch_idx + b)
    row = physical_b * OBS_SIZE

    f_lo = tl.load(obs + row + 1).to(tl.uint32)
    f_hi = tl.load(obs + row + 2).to(tl.uint32)
    frontier_size = f_lo | (f_hi << 8)

    neighbor_start = row + 5 + frontier_size * 8
    source_idx_pos = neighbor_start + frontier_size * D * 3
    validity_start = source_idx_pos + 2
    candidates_start = validity_start + frontier_size

    src_lo = tl.load(obs + source_idx_pos + 0).to(tl.uint32)
    src_hi = tl.load(obs + source_idx_pos + 1).to(tl.uint32)
    src = src_lo | (src_hi << 8)

    cand_lo = tl.load(obs + candidates_start + 0).to(tl.uint32)
    cand_hi = tl.load(obs + candidates_start + 1).to(tl.uint32)
    cand_count = cand_lo | (cand_hi << 8)

    valid_i = i < frontier_size
    validity = tl.load(obs + validity_start + i, mask=valid_i, other=0) != 0
    valid_count = tl.sum(validity.to(tl.int64), axis=0)

    vertex_batch_offset = tl.load(vertex_batch_offsets + b)
    tl.store(node_validity + vertex_batch_offset + i, validity, mask=valid_i)

    tl.store(source_idx + b, src.to(tl.int64))
    tl.store(candidate_count + b, cand_count.to(tl.int64))
    tl.store(valid_node_count + b, valid_count)
    tl.store(target_count + b, valid_count + cand_count.to(tl.int64))


@triton.jit
def _cand_write_target_size_kernel(
    target_batch_offsets,   # int64*, [B + 1]
    sizes_dev,              # int64*, [3], sizes_dev[2] = Q
    B: tl.constexpr,
):
    q = tl.load(target_batch_offsets + B)
    tl.store(sizes_dev + 2, q)


@triton.jit
def _cand_scatter_existing_targets_kernel(
    vertices,               # float32*, [N_CAP, 2]
    vertex_batch_offsets,   # int64*, [B + 1]
    target_batch_offsets,   # int64*, [B + 1]
    valid_node_count,       # int64*, [B]
    node_validity,          # bool*, [N_CAP]
    target_idx,             # int64*, [Q_CAP]
    target_batches,         # int64*, [Q_CAP]
    target_positions,       # float32*, [Q_CAP, 2]
    F_CAP: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.arange(0, BLOCK_F)

    vertex_begin = tl.load(vertex_batch_offsets + b)
    vertex_end = tl.load(vertex_batch_offsets + b + 1)
    frontier_size = vertex_end - vertex_begin

    valid_i = i < frontier_size
    global_i = vertex_begin + i
    validity = tl.load(node_validity + global_i, mask=valid_i, other=0) != 0
    rank = tl.cumsum(validity.to(tl.int64), axis=0) - 1

    target_begin = tl.load(target_batch_offsets + b)
    out = target_begin + rank
    mask = valid_i & validity

    x = tl.load(vertices + global_i * 2 + 0, mask=mask, other=0.0)
    y = tl.load(vertices + global_i * 2 + 1, mask=mask, other=0.0)

    tl.store(target_idx + out, global_i, mask=mask)
    tl.store(target_batches + out, b, mask=mask)
    tl.store(target_positions + out * 2 + 0, x, mask=mask)
    tl.store(target_positions + out * 2 + 1, y, mask=mask)


@triton.jit
def _cand_scatter_new_targets_kernel(
    obs,                    # uint8*, [B_full, obs_size]
    valid_batch_idx,        # int64*, [B]
    target_batch_offsets,   # int64*, [B + 1]
    valid_node_count,       # int64*, [B]
    candidate_count,        # int64*, [B]
    target_idx,             # int64*, [Q_CAP]
    target_batches,         # int64*, [Q_CAP]
    target_positions,       # float32*, [Q_CAP, 2]
    OBS_SIZE: tl.constexpr,
    D: tl.constexpr,
    C_CAP: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    b = tl.program_id(0)
    block_j = tl.program_id(1)
    j = block_j * BLOCK_C + tl.arange(0, BLOCK_C)

    physical_b = tl.load(valid_batch_idx + b)
    row = physical_b * OBS_SIZE

    f_lo = tl.load(obs + row + 1).to(tl.uint32)
    f_hi = tl.load(obs + row + 2).to(tl.uint32)
    frontier_size = f_lo | (f_hi << 8)

    neighbor_start = row + 5 + frontier_size * 8
    source_idx_pos = neighbor_start + frontier_size * D * 3
    validity_start = source_idx_pos + 2
    candidates_start = validity_start + frontier_size

    cand_count = tl.load(candidate_count + b)
    valid = (j < cand_count) & (j < C_CAP)

    base = candidates_start + 2 + j * 8
    x_u32 = (
        tl.load(obs + base + 0, mask=valid, other=0).to(tl.uint32)
        | (tl.load(obs + base + 1, mask=valid, other=0).to(tl.uint32) << 8)
        | (tl.load(obs + base + 2, mask=valid, other=0).to(tl.uint32) << 16)
        | (tl.load(obs + base + 3, mask=valid, other=0).to(tl.uint32) << 24)
    )
    y_u32 = (
        tl.load(obs + base + 4, mask=valid, other=0).to(tl.uint32)
        | (tl.load(obs + base + 5, mask=valid, other=0).to(tl.uint32) << 8)
        | (tl.load(obs + base + 6, mask=valid, other=0).to(tl.uint32) << 16)
        | (tl.load(obs + base + 7, mask=valid, other=0).to(tl.uint32) << 24)
    )

    out = tl.load(target_batch_offsets + b) + tl.load(valid_node_count + b) + j

    tl.store(target_idx + out, tl.full((BLOCK_C,), -1, tl.int64), mask=valid)
    tl.store(target_batches + out, b, mask=valid)
    tl.store(target_positions + out * 2 + 0, x_u32.to(tl.float32, bitcast=True), mask=valid)
    tl.store(target_positions + out * 2 + 1, y_u32.to(tl.float32, bitcast=True), mask=valid)


# =============================================================================
# CUDA-graph-backed deserializer
# =============================================================================


def _pow2_at_least_1(x: int) -> int:
    x = max(int(x), 1)
    return 1 << (x - 1).bit_length()


def _assert_same_cuda_buffer(name, x, static):
    if x.data_ptr() != static.data_ptr():
        raise RuntimeError(
            f"{name} data_ptr changed under copy_obs=False. "
            f"capture ptr={static.data_ptr()}, call ptr={x.data_ptr()}. "
            f"Use a persistent CUDA tensor and update it in-place, or set copy_obs=True."
        )
    if x.shape != static.shape:
        raise RuntimeError(f"{name} shape changed: {x.shape} vs {static.shape}")
    if x.dtype != static.dtype:
        raise RuntimeError(f"{name} dtype changed: {x.dtype} vs {static.dtype}")
    if x.device != static.device:
        raise RuntimeError(f"{name} device changed: {x.device} vs {static.device}")


class CUDAGraphObservationDeserializer:
    """
    CUDA-graph-backed observation deserializer.

    Assumes:
      - obs shape is fixed.
      - valid_batch_idx length is fixed.
      - D is fixed.
      - F_CAP is an upper bound for frontier_size.
      - C_CAP is an upper bound for new candidate count when candidates are decoded.
      - Output buffers are reused across calls.

    exact_output=True:
      Returns exact compact CSR views, but pays one final synchronization
      to read [N, E] to CPU for Python slicing.

    exact_output=False:
      Avoids the final sync, but returns capacity-sized buffers. This is the
      fastest path only if downstream code can consume capacity buffers plus
      device-side sizes.
    """

    def __init__(
        self,
        obs_example: torch.Tensor,
        valid_batch_idx_example: torch.Tensor,
        D: int,
        F_CAP: int,
        C_CAP: int = 0,
        decode_candidates: bool = False,
        exact_output: bool = True,
        copy_obs: bool = True,
        use_cuda_graph: bool = True,
        warmup_iters: int = 3,
    ):
        assert obs_example.dtype == torch.uint8
        assert obs_example.is_cuda
        assert valid_batch_idx_example.dtype == torch.long
        assert valid_batch_idx_example.is_cuda

        self.device = obs_example.device
        self.B_full, self.obs_size = obs_example.shape
        self.B = int(valid_batch_idx_example.numel())

        self.D = D
        self.F_CAP = F_CAP
        self.C_CAP = int(C_CAP)
        self.decode_candidates = bool(decode_candidates)

        self.BLOCK_B = 128
        self.BLOCK_D = _pow2_at_least_1(self.D)
        self.BLOCK_F = _pow2_at_least_1(self.F_CAP)
        self.BLOCK_C = 128
        self.num_warps_d = 1 if self.BLOCK_D <= 64 else 4

        self.N_CAP = max(self.B * self.F_CAP, 1)
        self.E_CAP = max(self.N_CAP * self.D, 1)
        self.Q_CAP = max(self.B * (self.F_CAP + self.C_CAP), 1)

        self.exact_output = bool(exact_output)
        self.copy_obs = bool(copy_obs)
        self.use_cuda_graph = bool(use_cuda_graph)

        with torch.cuda.device(self.device):
            if self.copy_obs:
                self.obs_static = torch.empty_like(obs_example)
                self.valid_batch_idx_static = torch.empty_like(valid_batch_idx_example)
            else:
                # Fastest mode, but caller must reuse the same tensor storage.
                self.obs_static = obs_example
                self.valid_batch_idx_static = valid_batch_idx_example

            self.frontier_size = torch.empty(
                (self.B,),
                dtype=torch.long,
                device=self.device,
            )

            self.vertex_batch_offsets = torch.empty(
                (self.B + 1,),
                dtype=torch.long,
                device=self.device,
            )

            self.vertices = torch.empty(
                (self.N_CAP, 2),
                dtype=torch.float32,
                device=self.device,
            )

            self.neighbor_count = torch.empty(
                (self.N_CAP,),
                dtype=torch.long,
                device=self.device,
            )

            self.edge_ptr = torch.empty(
                (self.N_CAP + 1,),
                dtype=torch.long,
                device=self.device,
            )

            # Flat capacity buffers so exact compact views can be contiguous:
            #   edges_flat[:2 * E].view(2, E)
            #   edge_features_flat[:2 * E].view(E, 2)
            self.edges_flat = torch.empty(
                (2 * self.E_CAP,),
                dtype=torch.long,
                device=self.device,
            )

            self.edge_features_flat = torch.empty(
                (2 * self.E_CAP,),
                dtype=torch.bool,
                device=self.device,
            )

            self.sizes_dev = torch.empty(
                (3,),
                dtype=torch.long,
                device=self.device,
            )

            # Pinned host staging for exact Python views.
            self.sizes_host = torch.empty(
                (3,),
                dtype=torch.long,
                pin_memory=True,
            )

            if self.decode_candidates:
                self.source_idx = torch.empty(
                    (self.B,),
                    dtype=torch.long,
                    device=self.device,
                )
                self.candidate_count = torch.empty(
                    (self.B,),
                    dtype=torch.long,
                    device=self.device,
                )
                self.valid_node_count = torch.empty(
                    (self.B,),
                    dtype=torch.long,
                    device=self.device,
                )
                self.target_count = torch.empty(
                    (self.B,),
                    dtype=torch.long,
                    device=self.device,
                )
                self.node_validity = torch.empty(
                    (self.N_CAP,),
                    dtype=torch.bool,
                    device=self.device,
                )
                self.target_batch_offsets = torch.empty(
                    (self.B + 1,),
                    dtype=torch.long,
                    device=self.device,
                )
                self.target_idx = torch.empty(
                    (self.Q_CAP,),
                    dtype=torch.long,
                    device=self.device,
                )
                self.target_batches = torch.empty(
                    (self.Q_CAP,),
                    dtype=torch.long,
                    device=self.device,
                )
                self.target_positions = torch.empty(
                    (self.Q_CAP, 2),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.target_batch_offsets_tail = self.target_batch_offsets[1:]
            else:
                self.source_idx = None
                self.candidate_count = None
                self.valid_node_count = None
                self.target_count = None
                self.node_validity = None
                self.target_batch_offsets = None
                self.target_idx = None
                self.target_batches = None
                self.target_positions = None
                self.target_batch_offsets_tail = None

            self.vertex_batch_offsets_tail = self.vertex_batch_offsets[1:]
            self.edge_ptr_tail = self.edge_ptr[1:]

            if self.copy_obs:
                self.obs_static.copy_(obs_example, non_blocking=True)
                self.valid_batch_idx_static.copy_(valid_batch_idx_example, non_blocking=True)

            if self.use_cuda_graph:
                self.graph = torch.cuda.CUDAGraph()

                # Warm up on a side stream, as recommended for PyTorch CUDA Graphs.
                # This also forces Triton compilation before capture.
                cur_stream = torch.cuda.current_stream(self.device)
                warmup_stream = torch.cuda.Stream(device=self.device)
                warmup_stream.wait_stream(cur_stream)

                with torch.cuda.stream(warmup_stream):
                    for _ in range(int(warmup_iters)):
                        self._run_static_pipeline()

                cur_stream.wait_stream(warmup_stream)
                cur_stream.synchronize()

                with torch.cuda.graph(self.graph):
                    self._run_static_pipeline()
            else:
                self.graph = None

    def _run_static_pipeline(self):
        _gf_init_offsets_kernel[(1,)](
            self.vertex_batch_offsets,
            self.edge_ptr,
            num_warps=1,
        )

        _gf_decode_frontier_sizes_kernel[
            (triton.cdiv(self.B, self.BLOCK_B),)
        ](
            self.obs_static,
            self.valid_batch_idx_static,
            self.frontier_size,
            OBS_SIZE=self.obs_size,
            B=self.B,
            BLOCK_B=self.BLOCK_B,
            num_warps=4,
        )

        # vertex_batch_offsets[0] was set to 0 above.
        torch.cumsum(
            self.frontier_size,
            dim=0,
            out=self.vertex_batch_offsets_tail,
        )

        # Needed for correctness when scanning capacity buffers.
        # Without this, invalid/padded entries may contain stale counts.
        self.neighbor_count.zero_()

        BLOCK_N = 128

        _gf_decode_vertices_kernel[
            (self.B, triton.cdiv(self.F_CAP, BLOCK_N))
        ](
            self.obs_static,
            self.vertices,
            self.vertex_batch_offsets,
            self.valid_batch_idx_static,
            OBS_SIZE=self.obs_size,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )

        # Try 8, 16, 32. For D=24, I would start with 16.
        BLOCK_COUNT_N = 16

        _gf_count_neighbors_kernel[
            (self.B, triton.cdiv(self.F_CAP, BLOCK_COUNT_N))
        ](
            self.obs_static,
            self.neighbor_count,
            self.vertex_batch_offsets,
            self.valid_batch_idx_static,
            OBS_SIZE=self.obs_size,
            D=self.D,
            F_CAP=self.F_CAP,
            BLOCK_N=BLOCK_COUNT_N,
            BLOCK_D=self.BLOCK_D,
            num_warps=1,
        )

        # edge_ptr[0] was set to 0 above.
        # We scan the full capacity to keep the captured shape static.
        torch.cumsum(
            self.neighbor_count,
            dim=0,
            out=self.edge_ptr_tail,
        )

        _gf_write_sizes_kernel[(1,)](
            self.vertex_batch_offsets,
            self.edge_ptr,
            self.sizes_dev,
            B=self.B,
            num_warps=1,
        )

        BLOCK_SCATTER_N = BLOCK_COUNT_N

        # Always launch scatter. If E == 0, masks suppress all stores.
        _gf_scatter_edges_kernel[
            (self.B, triton.cdiv(self.F_CAP, BLOCK_SCATTER_N))
        ](
            self.obs_static,
            self.edges_flat,
            self.edge_features_flat,
            self.edge_ptr,
            self.vertex_batch_offsets,
            self.sizes_dev,
            self.valid_batch_idx_static,
            OBS_SIZE=self.obs_size,
            D=self.D,
            F_CAP=self.F_CAP,
            BLOCK_N=BLOCK_SCATTER_N,
            BLOCK_D=self.BLOCK_D,
            num_warps=1,
        )

        if self.decode_candidates:
            _cand_init_offsets_kernel[(1,)](
                self.target_batch_offsets,
                num_warps=1,
            )

            _cand_decode_counts_kernel[(self.B,)](
                self.obs_static,
                self.valid_batch_idx_static,
                self.vertex_batch_offsets,
                self.source_idx,
                self.candidate_count,
                self.valid_node_count,
                self.target_count,
                self.node_validity,
                OBS_SIZE=self.obs_size,
                D=self.D,
                F_CAP=self.F_CAP,
                BLOCK_F=self.BLOCK_F,
                num_warps=4,
            )

            torch.cumsum(
                self.target_count,
                dim=0,
                out=self.target_batch_offsets_tail,
            )

            _cand_write_target_size_kernel[(1,)](
                self.target_batch_offsets,
                self.sizes_dev,
                B=self.B,
                num_warps=1,
            )

            _cand_scatter_existing_targets_kernel[(self.B,)](
                self.vertices,
                self.vertex_batch_offsets,
                self.target_batch_offsets,
                self.valid_node_count,
                self.node_validity,
                self.target_idx,
                self.target_batches,
                self.target_positions,
                F_CAP=self.F_CAP,
                BLOCK_F=self.BLOCK_F,
                num_warps=4,
            )

            _cand_scatter_new_targets_kernel[
                (self.B, triton.cdiv(max(self.C_CAP, 1), self.BLOCK_C))
            ](
                self.obs_static,
                self.valid_batch_idx_static,
                self.target_batch_offsets,
                self.valid_node_count,
                self.candidate_count,
                self.target_idx,
                self.target_batches,
                self.target_positions,
                OBS_SIZE=self.obs_size,
                D=self.D,
                C_CAP=max(self.C_CAP, 1),
                BLOCK_C=self.BLOCK_C,
                num_warps=4,
            )

    def _candidate_targets(self, Q: int | None = None) -> CandidateTargets:
        if Q is None:
            return CandidateTargets(
                source_idx=self.source_idx,
                target_idx=self.target_idx,
                target_batch_offsets=self.target_batch_offsets,
                target_batches=self.target_batches,
                target_positions=self.target_positions,
            )

        return CandidateTargets(
            source_idx=self.source_idx,
            target_idx=self.target_idx[:Q],
            target_batch_offsets=self.target_batch_offsets,
            target_batches=self.target_batches[:Q],
            target_positions=self.target_positions[:Q],
        )

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor, valid_batch_idx: torch.Tensor):
        assert obs.dtype == torch.uint8
        assert obs.is_cuda
        assert valid_batch_idx.dtype == torch.long
        assert valid_batch_idx.is_cuda
        assert obs.shape == (self.B_full, self.obs_size)
        assert valid_batch_idx.numel() == self.B

        self.valid_batch_idx_static.copy_(valid_batch_idx, non_blocking=True)

        if self.copy_obs:
            self.obs_static.copy_(obs, non_blocking=True)
        else:
            # Fastest mode: no input staging copy.
            # Requires the same underlying storage as the tensors used at capture.
            _assert_same_cuda_buffer("obs", obs, self.obs_static)

        if self.use_cuda_graph:
            self.graph.replay()
        else:
            self._run_static_pipeline()

        if not self.exact_output:
            # Fastest return path. No CPU sync.
            # Downstream must know that these are capacity-backed.
            graph = CSRGraph(
                vertices=self.vertices,
                batch_offsets=self.vertex_batch_offsets,
                edges=self.edges_flat.view(2, self.E_CAP),
                edge_features=self.edge_features_flat.view(self.E_CAP, 2),
                edge_ptr=self.edge_ptr,
            )
            if self.decode_candidates:
                return graph, self._candidate_targets()
            return graph

        # Exact drop-in mode:
        # Copy [N, E] to pinned host memory once, then synchronize once.
        self.sizes_host.copy_(self.sizes_dev, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()

        N = int(self.sizes_host[0].item())
        E = int(self.sizes_host[1].item())
        Q = int(self.sizes_host[2].item()) if self.decode_candidates else 0

        graph = CSRGraph(
            vertices=self.vertices[:N],
            batch_offsets=self.vertex_batch_offsets,
            edges=self.edges_flat[: 2 * E].view(2, E),
            edge_features=self.edge_features_flat[: 2 * E].view(E, 2),
            edge_ptr=self.edge_ptr[: N + 1],
        )

        if self.decode_candidates:
            return graph, self._candidate_targets(Q)
        return graph


_DESERIALIZE_OBSERVATION_CACHE = {}


@torch.no_grad()
def deserialize_observation(
    obs: torch.Tensor,
    valid_batch_idx: torch.Tensor,
    D: int,
    F_CAP: int,
    C_CAP: int = 0,
    deserialize_candidates: bool = False,
    exact_output: bool = True,
    copy_obs: bool = True,
    use_cuda_graph: bool = True,
) -> CSRGraph | tuple[CSRGraph, CandidateTargets]:
    """
    Deserializes frontier graph data, optionally with packed candidate targets.

    Args:
        obs: uint8 CUDA tensor [B, obs_size]
        valid_batch_idx: long CUDA tensor [B]

    Returns:
        CSRGraph, or (CSRGraph, CandidateTargets) when deserialize_candidates=True.

    Important:
      - Returned tensors alias internal reusable buffers and are overwritten
        on the next call. Use double buffering if consumers overlap calls.
    """

    assert obs.dtype == torch.uint8
    assert obs.is_cuda
    assert valid_batch_idx.dtype == torch.long
    assert valid_batch_idx.is_cuda

    key = (
        obs.device.type,
        obs.device.index,
        tuple(obs.shape),
        int(valid_batch_idx.numel()),
        int(D),
        int(F_CAP),
        int(C_CAP),
        bool(deserialize_candidates),
        bool(exact_output),
        bool(copy_obs),
        bool(use_cuda_graph),
    )

    deser = _DESERIALIZE_OBSERVATION_CACHE.get(key)
    if deser is None:
        deser = CUDAGraphObservationDeserializer(
            obs,
            valid_batch_idx,
            D=D,
            F_CAP=F_CAP,
            C_CAP=C_CAP,
            decode_candidates=deserialize_candidates,
            exact_output=exact_output,
            copy_obs=copy_obs,
            use_cuda_graph=use_cuda_graph,
        )
        _DESERIALIZE_OBSERVATION_CACHE[key] = deser

    return deser(obs, valid_batch_idx)


# ============================================================================
# QuadMeshing Boundary Policy Components
# ============================================================================

import math
from typing import Optional

class FourierEncoder2D(nn.Module):
    def __init__(
        self,
        num_bands: int,
        max_freq: Optional[float] = None,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        if num_bands < 0:
            raise ValueError("num_bands can't be negative.")
        if max_freq is not None and max_freq <= 0:
            raise ValueError("max_freq must be positive.")
        self.num_bands = num_bands
        self.include_input = include_input
        if max_freq is None:
            freq_bands = 2.0 ** torch.arange(num_bands, dtype=torch.float32)
        else:
            freq_bands = torch.logspace(
                0.0, math.log2(max_freq), steps=num_bands, base=2.0
            ).to(dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def out_dim(self) -> int:
        base = 4 * self.num_bands
        return base + 2 if self.include_input else base

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != 2:
            raise ValueError("Expected coords with last dimension size 2.")
        coords = coords.to(dtype=torch.float32)
        scaled = coords.unsqueeze(-1) * self.freq_bands
        scaled = 2.0 * math.pi * scaled
        sin = torch.sin(scaled)
        cos = torch.cos(scaled)
        feat = torch.cat([sin, cos], dim=-1)
        feat = feat.flatten(-2)
        if self.include_input:
            feat = torch.cat([coords, feat], dim=-1)
        return feat


class PerceiverCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query, key_value, key_padding_mask=None):
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        out, _ = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + self.dropout(out)
        return query + self.mlp(self.norm_mlp(query))


class PerceiverSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm_attn(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(out)
        return x + self.mlp(self.norm_mlp(x))


@dataclass
class FrontierState:
    graph: CSRGraph
    node_features: torch.Tensor       # [N, node_dim]
    edge_features: torch.Tensor       # [E, edge_dim]
    context_features: torch.Tensor    # [B, context_dim] or empty


class FrontierInitStage(nn.Module):
    def __init__(
        self,
        node_hidden_size,
        edge_hidden_size,
        pos_bands,
        n_neighbors=3,
        node_position=True,
        node_ring=False,
        edge_midpoint=True,
        edge_length=True,
        edge_direction=True,
        edge_flags=True,
        eps=1e-8,
    ):
        super().__init__()
        self.node_hidden_size = node_hidden_size
        self.edge_hidden_size = edge_hidden_size
        self.n_neighbors = n_neighbors
        self.node_position = node_position
        self.node_ring = node_ring
        self.edge_midpoint = edge_midpoint
        self.edge_length = edge_length
        self.edge_direction = edge_direction
        self.edge_flags = edge_flags
        self.eps = eps
        self.pos_encoder = FourierEncoder2D(num_bands=pos_bands, include_input=True)

        node_in_dim = 0
        if node_position:
            node_in_dim += self.pos_encoder.out_dim
        if node_ring:
            node_in_dim += 2 + 2 * n_neighbors * 3

        edge_in_dim = 0
        if edge_midpoint:
            edge_in_dim += self.pos_encoder.out_dim
        if edge_length:
            edge_in_dim += 1
        if edge_direction:
            edge_in_dim += 2
        if edge_flags:
            edge_in_dim += 2

        self.node_proj = layer_init(nn.Linear(node_in_dim, node_hidden_size)) if node_in_dim > 0 else None
        self.edge_proj = layer_init(nn.Linear(edge_in_dim, edge_hidden_size)) if edge_in_dim > 0 else None
        self.node_initial = nn.Parameter(torch.zeros(node_hidden_size))
        self.edge_initial = nn.Parameter(torch.zeros(edge_hidden_size))

    @staticmethod
    def _canonical_direction(delta, length, eps):
        direction = delta / length.clamp_min(eps)
        flip = (direction[:, 1] < 0) | ((direction[:, 1] == 0) & (direction[:, 0] < 0))
        return torch.where(flip.unsqueeze(-1), -direction, direction)

    def _node_features(self, graph):
        parts = []
        if self.node_position:
            parts.append(self.pos_encoder(graph.vertices))
        if self.node_ring:
            with nvtx_range("compute_ring_source_features"):
                parts.append(compute_ring_source_features(graph, self.n_neighbors))
        if not parts:
            return self.node_initial.unsqueeze(0).expand(graph.vertices.size(0), -1)
        return self.node_proj(torch.cat(parts, dim=-1))

    def _edge_features(self, graph):
        src = graph.edges[0]
        dst = graph.edges[1]
        parts = []
        p0 = graph.vertices[src]
        p1 = graph.vertices[dst]
        delta = p1 - p0
        length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)

        if self.edge_midpoint:
            parts.append(self.pos_encoder(0.5 * (p0 + p1)))
        if self.edge_length:
            parts.append(length)
        if self.edge_direction:
            parts.append(self._canonical_direction(delta, length, self.eps))
        if self.edge_flags:
            parts.append(graph.edge_features.to(dtype=graph.vertices.dtype))

        if not parts:
            return self.edge_initial.unsqueeze(0).expand(graph.edges.size(1), -1)
        return self.edge_proj(torch.cat(parts, dim=-1))

    def forward(self, state):
        graph = state.graph
        return FrontierState(
            graph=graph,
            node_features=self._node_features(graph),
            edge_features=self._edge_features(graph),
            context_features=graph.vertices.new_empty(0, 0),
        )


class SE2FrontierStage(nn.Module):
    def __init__(self, node_hidden_size, edge_hidden_size, num_layers=2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("frontier_se2_layers must be at least 1.")
        self.layers = nn.ModuleList([
            SE2AngleMessagePassing(
                node_dim=node_hidden_size,
                out_dim=node_hidden_size,
                msg_dim=node_hidden_size,
                edge_cont_dim=edge_hidden_size,
                hidden_dim=node_hidden_size,
            )
            for _ in range(num_layers)
        ])

    def forward(self, state):
        h = state.node_features
        for layer in self.layers:
            with nvtx_range("se2_frontier_message_passing"):
                h = layer(
                    h,
                    state.graph.vertices,
                    state.graph.edges,
                    state.graph.edge_ptr,
                    edge_cont=state.edge_features,
                )
        return FrontierState(state.graph, h, state.edge_features, state.context_features)


class FrontierPerceiverStage(nn.Module):
    def __init__(
        self,
        node_hidden_size,
        edge_hidden_size,
        context_hidden_size,
        d_model=None,
        num_latents=32,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.0,
        decode_node=True,
        decode_edge=False,
        update_context=True,
    ):
        super().__init__()
        d_model = context_hidden_size if d_model is None else d_model
        if num_latents < 1:
            raise ValueError("frontier_perceiver_num_latents must be at least 1.")
        if num_layers < 1:
            raise ValueError("frontier_perceiver_layers must be at least 1.")
        if d_model % num_heads != 0:
            raise ValueError("frontier_perceiver_d_model must be divisible by frontier_perceiver_heads.")

        self.node_hidden_size = node_hidden_size
        self.edge_hidden_size = edge_hidden_size
        self.context_hidden_size = context_hidden_size
        self.d_model = d_model
        self.decode_node = decode_node
        self.decode_edge = decode_edge
        self.update_context = update_context

        self.edge_input_proj = layer_init(nn.Linear(edge_hidden_size, d_model))
        self.node_query_proj = layer_init(nn.Linear(node_hidden_size, d_model))
        self.edge_query_proj = layer_init(nn.Linear(edge_hidden_size, d_model))
        self.node_out_proj = layer_init(nn.Linear(d_model, node_hidden_size))
        self.edge_out_proj = layer_init(nn.Linear(d_model, edge_hidden_size))
        self.context_proj = layer_init(nn.Linear(d_model, context_hidden_size))

        self.latents = nn.Parameter(torch.empty(num_latents, d_model))
        self.null_edge = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.latents, std=0.02)

        mlp_dim = int(d_model * mlp_ratio)
        self.input_cross_attn = PerceiverCrossAttention(d_model, num_heads, mlp_dim, dropout)
        self.blocks = nn.ModuleList([
            PerceiverSelfAttentionBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.query_cross_attn = PerceiverCrossAttention(d_model, num_heads, mlp_dim, dropout)

    @staticmethod
    def _batch_index_from_offsets(indices, offsets):
        return torch.searchsorted(offsets[1:], indices, right=True)

    @staticmethod
    def _pad_by_batch(values, batch_idx, counts, fill_value=0.0, add_null=False, null_value=None):
        B = counts.numel()
        width = int(counts.max().item()) if counts.numel() > 0 else 0
        if add_null:
            width += 1
        width = max(width, 1)

        out = values.new_full((B, width, values.size(-1)), fill_value)
        mask = torch.ones(B, width, dtype=torch.bool, device=values.device)
        if add_null:
            if null_value is None:
                out[:, 0] = 0.0
            else:
                out[:, 0] = null_value
            mask[:, 0] = False

        if values.numel() == 0:
            return out, mask

        offsets = torch.empty(B + 1, dtype=torch.long, device=values.device)
        offsets[0] = 0
        offsets[1:] = torch.cumsum(counts, dim=0)
        pos = torch.arange(values.size(0), device=values.device) - offsets[batch_idx]
        if add_null:
            pos = pos + 1
        out[batch_idx, pos] = values
        mask[batch_idx, pos] = False
        return out, mask

    def _undirected_edge_tokens(self, state):
        graph = state.graph
        src = graph.edges[0]
        dst = graph.edges[1]
        keep = src < dst
        edge_src = src[keep]
        if edge_src.numel() == 0:
            return state.edge_features.new_empty(0, self.d_model), edge_src
        return self.edge_input_proj(state.edge_features[keep]), edge_src

    def _decode_packed(self, queries, query_batch, counts, latents, out_proj):
        padded_queries, _ = self._pad_by_batch(queries, query_batch, counts)
        decoded = self.query_cross_attn(padded_queries, latents)
        offsets = torch.empty(counts.numel() + 1, dtype=torch.long, device=queries.device)
        offsets[0] = 0
        offsets[1:] = torch.cumsum(counts, dim=0)
        pos = torch.arange(queries.size(0), device=queries.device) - offsets[query_batch]
        return out_proj(decoded[query_batch, pos])

    def forward(self, state):
        graph = state.graph
        B = graph.batch_offsets.numel() - 1
        if B == 0:
            return state

        edge_tokens, edge_src = self._undirected_edge_tokens(state)
        vertex_counts = graph.batch_offsets[1:] - graph.batch_offsets[:-1]

        if edge_src.numel() == 0:
            edge_counts = torch.zeros(B, dtype=torch.long, device=graph.vertices.device)
            edge_batch = torch.empty(0, dtype=torch.long, device=graph.vertices.device)
        else:
            edge_batch = self._batch_index_from_offsets(edge_src, graph.batch_offsets)
            edge_counts = torch.bincount(edge_batch, minlength=B)

        padded_edges, edge_mask = self._pad_by_batch(
            edge_tokens,
            edge_batch,
            edge_counts,
            add_null=True,
            null_value=self.null_edge,
        )

        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        latents = self.input_cross_attn(latents, padded_edges, key_padding_mask=edge_mask)
        for block in self.blocks:
            latents = block(latents)

        node_features = state.node_features
        edge_features = state.edge_features
        context_features = state.context_features
        if self.update_context:
            context_features = self.context_proj(latents.mean(dim=1))

        node_batch = torch.repeat_interleave(
            torch.arange(B, device=graph.vertices.device), vertex_counts
        )
        if self.decode_node and state.node_features.numel() > 0:
            node_queries = self.node_query_proj(state.node_features)
            node_features = self._decode_packed(
                node_queries, node_batch, vertex_counts, latents, self.node_out_proj
            )

        if self.decode_edge and state.edge_features.numel() > 0:
            edge_batch = self._batch_index_from_offsets(graph.edges[0], graph.batch_offsets)
            edge_counts = torch.bincount(edge_batch, minlength=B)
            edge_queries = self.edge_query_proj(state.edge_features)
            edge_features = self._decode_packed(
                edge_queries, edge_batch, edge_counts, latents, self.edge_out_proj
            )

        return FrontierState(graph, node_features, edge_features, context_features)


@triton.jit
def _ring_source_features_kernel(
    vertices,       # float32*, [N, 2]
    edges,          # int64*, [2, E]
    edge_features,  # bool*, [E, 2]
    edge_ptr,       # int64*, [N + 1]
    out,            # float32*, [N, 2 + 2 * K * 3]
    N,
    E,
    K: tl.constexpr,
    OUT_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    vx = tl.load(vertices + offs * 2 + 0, mask=mask, other=0.0)
    vy = tl.load(vertices + offs * 2 + 1, mask=mask, other=0.0)

    e0 = tl.load(edge_ptr + offs, mask=mask, other=0)
    e1 = e0 + 1
    e0_ccw = tl.load(edge_features + e0 * 2 + 1, mask=mask, other=0)

    ccw_edge = tl.where(e0_ccw, e0, e1)
    cw_edge = tl.where(e0_ccw, e1, e0)

    next_v = tl.load(edges + E + ccw_edge, mask=mask, other=0)
    prev_v = tl.load(edges + E + cw_edge, mask=mask, other=0)

    next_x = tl.load(vertices + next_v * 2 + 0, mask=mask, other=0.0)
    next_y = tl.load(vertices + next_v * 2 + 1, mask=mask, other=0.0)
    prev_x = tl.load(vertices + prev_v * 2 + 0, mask=mask, other=0.0)
    prev_y = tl.load(vertices + prev_v * 2 + 1, mask=mask, other=0.0)

    dnx = next_x - vx
    dny = next_y - vy
    dpx = prev_x - vx
    dpy = prev_y - vy
    inv_next = tl.rsqrt(tl.maximum(dnx * dnx + dny * dny, EPS))
    inv_prev = tl.rsqrt(tl.maximum(dpx * dpx + dpy * dpy, EPS))
    dnx = dnx * inv_next
    dny = dny * inv_next
    dpx = dpx * inv_prev
    dpy = dpy * inv_prev

    cos_val = dnx * dpx + dny * dpy
    sin_val = dnx * dpy - dny * dpx

    tl.store(out + offs * OUT_DIM + 0, cos_val, mask=mask)
    tl.store(out + offs * OUT_DIM + 1, sin_val, mask=mask)

    cur_cw_v = offs.to(tl.int64)
    cur_ccw_v = offs.to(tl.int64)

    for hop in range(1, K + 1):
        cur_cw_e0 = tl.load(edge_ptr + cur_cw_v, mask=mask, other=0)
        cur_cw_e1 = cur_cw_e0 + 1
        cur_cw_e0_ccw = tl.load(edge_features + cur_cw_e0 * 2 + 1, mask=mask, other=0)
        cur_cw_edge = tl.where(cur_cw_e0_ccw, cur_cw_e1, cur_cw_e0)
        next_cw_v = tl.load(edges + E + cur_cw_edge, mask=mask, other=0)

        cur_ccw_e0 = tl.load(edge_ptr + cur_ccw_v, mask=mask, other=0)
        cur_ccw_e1 = cur_ccw_e0 + 1
        cur_ccw_e0_ccw = tl.load(edge_features + cur_ccw_e0 * 2 + 1, mask=mask, other=0)
        cur_ccw_edge = tl.where(cur_ccw_e0_ccw, cur_ccw_e0, cur_ccw_e1)
        next_ccw_v = tl.load(edges + E + cur_ccw_edge, mask=mask, other=0)

        for side in range(0, 2):
            feat_v = tl.where(side == 0, next_cw_v, next_ccw_v)
            feat_edge = tl.where(side == 0, cur_cw_edge, cur_ccw_edge)
            col = tl.where(side == 0, K - hop, K + hop - 1)
            base = 2 + col * 3

            fx = tl.load(vertices + feat_v * 2 + 0, mask=mask, other=0.0)
            fy = tl.load(vertices + feat_v * 2 + 1, mask=mask, other=0.0)

            fe0 = tl.load(edge_ptr + feat_v, mask=mask, other=0)
            fe1 = fe0 + 1
            fe0_ccw = tl.load(edge_features + fe0 * 2 + 1, mask=mask, other=0)
            f_ccw_edge = tl.where(fe0_ccw, fe0, fe1)
            f_cw_edge = tl.where(fe0_ccw, fe1, fe0)

            f_next_v = tl.load(edges + E + f_ccw_edge, mask=mask, other=0)
            f_prev_v = tl.load(edges + E + f_cw_edge, mask=mask, other=0)

            f_next_x = tl.load(vertices + f_next_v * 2 + 0, mask=mask, other=0.0)
            f_next_y = tl.load(vertices + f_next_v * 2 + 1, mask=mask, other=0.0)
            f_prev_x = tl.load(vertices + f_prev_v * 2 + 0, mask=mask, other=0.0)
            f_prev_y = tl.load(vertices + f_prev_v * 2 + 1, mask=mask, other=0.0)

            fdnx = f_next_x - fx
            fdny = f_next_y - fy
            fdpx = f_prev_x - fx
            fdpy = f_prev_y - fy
            f_inv_next = tl.rsqrt(tl.maximum(fdnx * fdnx + fdny * fdny, EPS))
            f_inv_prev = tl.rsqrt(tl.maximum(fdpx * fdpx + fdpy * fdpy, EPS))
            fdnx = fdnx * f_inv_next
            fdny = fdny * f_inv_next
            fdpx = fdpx * f_inv_prev
            fdpy = fdpy * f_inv_prev

            f_cos = fdnx * fdpx + fdny * fdpy
            f_sin = fdnx * fdpy - fdny * fdpx

            edge_src = tl.load(edges + feat_edge, mask=mask, other=0)
            edge_dst = tl.load(edges + E + feat_edge, mask=mask, other=0)
            sx = tl.load(vertices + edge_src * 2 + 0, mask=mask, other=0.0)
            sy = tl.load(vertices + edge_src * 2 + 1, mask=mask, other=0.0)
            dx = tl.load(vertices + edge_dst * 2 + 0, mask=mask, other=0.0)
            dy = tl.load(vertices + edge_dst * 2 + 1, mask=mask, other=0.0)
            elen = tl.sqrt(tl.maximum((dx - sx) * (dx - sx) + (dy - sy) * (dy - sy), EPS))

            tl.store(out + offs * OUT_DIM + base + 0, f_cos, mask=mask)
            tl.store(out + offs * OUT_DIM + base + 1, f_sin, mask=mask)
            tl.store(out + offs * OUT_DIM + base + 2, elen, mask=mask)

        cur_cw_v = next_cw_v
        cur_ccw_v = next_ccw_v


@triton.jit
def _ring_target_local_positions_kernel(
    vertices,           # float32*, [N, 2]
    edges,              # int64*, [2, E]
    edge_features,      # bool*, [E, 2]
    edge_ptr,           # int64*, [N + 1]
    target_positions,   # float32*, [Q, 2]
    global_source_idx,  # int64*, [B]
    target_batches,     # int64*, [Q]
    out,                # float32*, [Q, 2]
    Q,
    E,
    BLOCK_Q: tl.constexpr,
    EPS: tl.constexpr,
):
    q = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask = q < Q

    b = tl.load(target_batches + q, mask=mask, other=0)
    src = tl.load(global_source_idx + b, mask=mask, other=0)

    sx = tl.load(vertices + src * 2 + 0, mask=mask, other=0.0)
    sy = tl.load(vertices + src * 2 + 1, mask=mask, other=0.0)

    e0 = tl.load(edge_ptr + src, mask=mask, other=0)
    e1 = e0 + 1
    e0_ccw = tl.load(edge_features + e0 * 2 + 1, mask=mask, other=0)
    ccw_edge = tl.where(e0_ccw, e0, e1)
    cw_edge = tl.where(e0_ccw, e1, e0)

    next_v = tl.load(edges + E + ccw_edge, mask=mask, other=0)
    prev_v = tl.load(edges + E + cw_edge, mask=mask, other=0)

    nx = tl.load(vertices + next_v * 2 + 0, mask=mask, other=0.0)
    ny = tl.load(vertices + next_v * 2 + 1, mask=mask, other=0.0)
    px = tl.load(vertices + prev_v * 2 + 0, mask=mask, other=0.0)
    py = tl.load(vertices + prev_v * 2 + 1, mask=mask, other=0.0)

    t_in_x = sx - px
    t_in_y = sy - py
    t_out_x = nx - sx
    t_out_y = ny - sy
    inv_in = tl.rsqrt(tl.maximum(t_in_x * t_in_x + t_in_y * t_in_y, EPS))
    inv_out = tl.rsqrt(tl.maximum(t_out_x * t_out_x + t_out_y * t_out_y, EPS))
    t_in_x = t_in_x * inv_in
    t_in_y = t_in_y * inv_in
    t_out_x = t_out_x * inv_out
    t_out_y = t_out_y * inv_out

    x_axis_x = -t_in_y - t_out_y
    x_axis_y = t_in_x + t_out_x
    inv_axis = tl.rsqrt(tl.maximum(x_axis_x * x_axis_x + x_axis_y * x_axis_y, EPS))
    x_axis_x = x_axis_x * inv_axis
    x_axis_y = x_axis_y * inv_axis

    y_axis_x = -x_axis_y
    y_axis_y = x_axis_x

    tx = tl.load(target_positions + q * 2 + 0, mask=mask, other=0.0)
    ty = tl.load(target_positions + q * 2 + 1, mask=mask, other=0.0)
    cx = tx - sx
    cy = ty - sy

    tl.store(out + q * 2 + 0, cx * x_axis_x + cy * x_axis_y, mask=mask)
    tl.store(out + q * 2 + 1, cx * y_axis_x + cy * y_axis_y, mask=mask)


@torch.no_grad()
def compute_ring_source_features(graph: CSRGraph, n_neighbors: int, eps: float = 1e-8):
    N = graph.vertices.shape[0]
    out_dim = 2 + 2 * n_neighbors * 3
    out = torch.empty((N, out_dim), device=graph.vertices.device, dtype=torch.float32)
    if N == 0:
        return out

    E = graph.edges.shape[1]
    block_n = 128
    _ring_source_features_kernel[(triton.cdiv(N, block_n),)](
        graph.vertices,
        graph.edges,
        graph.edge_features,
        graph.edge_ptr,
        out,
        N=N,
        E=E,
        K=n_neighbors,
        OUT_DIM=out_dim,
        BLOCK_N=block_n,
        EPS=eps,
        num_warps=4,
    )
    return out


@torch.no_grad()
def compute_ring_target_local_positions(
    graph: CSRGraph,
    target_positions: torch.Tensor,
    global_source_idx: torch.Tensor,
    target_batches: torch.Tensor,
    eps: float = 1e-8,
):
    Q = target_positions.shape[0]
    out = torch.empty_like(target_positions)
    if Q == 0:
        return out

    E = graph.edges.shape[1]
    block_q = 128
    _ring_target_local_positions_kernel[(triton.cdiv(Q, block_q),)](
        graph.vertices,
        graph.edges,
        graph.edge_features,
        graph.edge_ptr,
        target_positions,
        global_source_idx,
        target_batches,
        out,
        Q=Q,
        E=E,
        BLOCK_Q=block_q,
        EPS=eps,
        num_warps=4,
    )
    return out


@triton.jit
def _csr_point_edge_distance_kernel(
    vertices,              # float32 [N, 2]
    edges,                 # int64   [2, E]
    edge_ptr,              # int64   [N + 1]
    vertex_batch_offsets,  # int64   [B + 1]
    query_points,          # float32 [Q, 2]
    query_batch_offsets,   # int64   [B + 1]
    out,                   # float32 [Q]
    E_TOTAL,
    BLOCK_Q: tl.constexpr,
    BLOCK_E: tl.constexpr,
    EPS: tl.constexpr,
):
    b = tl.program_id(0)
    qb = tl.program_id(1)

    q_begin = tl.load(query_batch_offsets + b)
    q_end = tl.load(query_batch_offsets + b + 1)

    v_begin = tl.load(vertex_batch_offsets + b)
    v_end = tl.load(vertex_batch_offsets + b + 1)

    e_begin = tl.load(edge_ptr + v_begin)
    e_end = tl.load(edge_ptr + v_end)

    q_offsets = qb * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_idx = q_begin + q_offsets
    valid_q = q_idx < q_end

    px = tl.load(query_points + q_idx * 2 + 0, mask=valid_q, other=0.0)
    py = tl.load(query_points + q_idx * 2 + 1, mask=valid_q, other=0.0)

    best = tl.full((BLOCK_Q,), float("inf"), tl.float32)

    e_offsets = tl.arange(0, BLOCK_E)
    e_base = e_begin

    while e_base < e_end:
        e_idx = e_base + e_offsets
        valid_e = e_idx < e_end

        src = tl.load(edges + e_idx, mask=valid_e, other=0)
        dst = tl.load(edges + E_TOTAL + e_idx, mask=valid_e, other=0)

        ax = tl.load(vertices + src * 2 + 0, mask=valid_e, other=0.0)
        ay = tl.load(vertices + src * 2 + 1, mask=valid_e, other=0.0)

        bx = tl.load(vertices + dst * 2 + 0, mask=valid_e, other=0.0)
        by = tl.load(vertices + dst * 2 + 1, mask=valid_e, other=0.0)

        vx = bx - ax
        vy = by - ay

        len2 = vx * vx + vy * vy
        len2_safe = tl.maximum(len2, EPS)

        # Shapes:
        #   query dimension: [BLOCK_Q, 1]
        #   edge dimension:  [1, BLOCK_E]
        apx = px[:, None] - ax[None, :]
        apy = py[:, None] - ay[None, :]

        t = (apx * vx[None, :] + apy * vy[None, :]) / len2_safe[None, :]
        t = tl.minimum(tl.maximum(t, 0.0), 1.0)

        cx = ax[None, :] + t * vx[None, :]
        cy = ay[None, :] + t * vy[None, :]

        dx = px[:, None] - cx
        dy = py[:, None] - cy

        d2 = dx * dx + dy * dy

        valid = valid_q[:, None] & valid_e[None, :]
        d2 = tl.where(valid, d2, float("inf"))

        block_best = tl.min(d2, axis=1)
        best = tl.minimum(best, block_best)

        e_base += BLOCK_E

    tl.store(out + q_idx, tl.sqrt(best), mask=valid_q)


@torch.no_grad()
def distance_to_graph_edges(
    graph,
    query_points: torch.Tensor,          # [Q, 2], float32 CUDA
    query_batch_offsets: torch.Tensor,   # [B + 1], int64 CUDA
    *,
    block_q: int = 16,
    block_e: int = 64,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute shortest distance from each query point to graph edges in the same
    batch.

    Returns:
        distances: [Q]
    """

    vertices = graph.vertices
    edges = graph.edges
    edge_ptr = graph.edge_ptr
    vertex_batch_offsets = graph.batch_offsets

    assert query_points.is_cuda
    assert vertices.is_cuda
    assert edges.is_cuda
    assert edge_ptr.is_cuda
    assert vertex_batch_offsets.is_cuda
    assert query_batch_offsets.is_cuda

    assert query_points.dtype == torch.float32
    assert vertices.dtype == torch.float32
    assert edges.dtype == torch.long
    assert edge_ptr.dtype == torch.long
    assert vertex_batch_offsets.dtype == torch.long
    assert query_batch_offsets.dtype == torch.long

    assert query_points.ndim == 2 and query_points.shape[1] == 2
    assert vertices.ndim == 2 and vertices.shape[1] == 2
    assert edges.ndim == 2 and edges.shape[0] == 2
    assert vertex_batch_offsets.ndim == 1
    assert query_batch_offsets.ndim == 1
    assert vertex_batch_offsets.numel() == query_batch_offsets.numel()

    # Important: make these preconditions explicit to avoid hidden PyTorch copies.
    assert query_points.is_contiguous()
    assert vertices.is_contiguous()
    assert edges.is_contiguous()
    assert edge_ptr.is_contiguous()
    assert vertex_batch_offsets.is_contiguous()
    assert query_batch_offsets.is_contiguous()

    Q = query_points.shape[0]
    B = query_batch_offsets.numel() - 1
    E = edges.shape[1]

    out = torch.empty((Q,), dtype=torch.float32, device=query_points.device)

    if Q == 0:
        return out

    if B == 0:
        return out

    # Need a Python launch-grid size. This is one small sync.
    query_counts = query_batch_offsets[1:] - query_batch_offsets[:-1]
    max_q_per_batch = int(query_counts.max().item())

    if max_q_per_batch == 0:
        return out

    grid = (
        B,
        triton.cdiv(max_q_per_batch, block_q),
    )

    _csr_point_edge_distance_kernel[grid](
        vertices,
        edges,
        edge_ptr,
        vertex_batch_offsets,
        query_points,
        query_batch_offsets,
        out,
        E_TOTAL=E,
        BLOCK_Q=block_q,
        BLOCK_E=block_e,
        EPS=eps,
        num_warps=4,
    )

    return out


@dataclass
class QuadMeshEncoding:
    substep: torch.Tensor                    # [B] substep per agent
    h_source0: torch.Tensor                  # [total_nodes0, frontier_node_hidden_size], source vertex hidden states (substep 0)
    h_source1: torch.Tensor                  # [total_nodes1, frontier_node_hidden_size], source vertex hidden states (substep 1)
    h_source0_context: torch.Tensor          # [B0, frontier_context_hidden_size], per-agent source context for substep 0 critic
    h_source0_batch_offset: torch.Tensor     # [B0 + 1] h_source offsets per agent (substep 0)
    h_source1_batch_offset: torch.Tensor     # [B1 + 1] h_source offsets per agent (substep 1)
    source_idx: torch.Tensor                 # [B1] source index (substep 1)
    h_target_source: torch.Tensor            # [total_targets, target_hidden_size], source vertex hidden states per target (substep 1)
    h_target: torch.Tensor                   # [total_targets, target_hidden_size], target hidden states (substep 1)
    h_target_batch_offset: torch.Tensor      # [B1 + 1] h_target offsets per agent


class QuadMeshingBoundaryEncoder(nn.Module):
    """
    Boundary-aware encoder for quad_meshing in boundary mode.

    Args:
        frontier_node_hidden_size: Dimension of per-frontier-vertex embeddings
        frontier_edge_hidden_size: Dimension of per-frontier-edge embeddings
        frontier_context_hidden_size: Dimension of per-observation frontier context embeddings
        target_hidden_size: Dimension of hidden target embeddings
        pos_bands: Number of Fourier frequency bands
        frontier_pipeline: Ordered list of frontier stages, e.g. ["init", "perceiver", "se2"]
    """

    def __init__(
            self,
            obs_size,
            max_frontier=1024,
            max_degree=24,
            max_candidates=1024,
            frontier_node_hidden_size=128,
            frontier_edge_hidden_size=128,
            frontier_context_hidden_size=128,
            target_hidden_size=128,
            pos_bands=6,
            n_neighbors=3,
            frontier_pipeline=("init",),
            frontier_init_node_position=True,
            frontier_init_node_ring=False,
            frontier_init_edge_midpoint=True,
            frontier_init_edge_length=True,
            frontier_init_edge_direction=True,
            frontier_init_edge_flags=True,
            frontier_se2_layers=2,
            frontier_perceiver_num_latents=32,
            frontier_perceiver_layers=2,
            frontier_perceiver_heads=4,
            frontier_perceiver_mlp_ratio=4,
            frontier_perceiver_dropout=0.0,
            frontier_perceiver_d_model=None,
            frontier_perceiver_decode_node=True,
            frontier_perceiver_decode_edge=False,
            frontier_perceiver_update_context=True,
            **kwargs
    ):
        super().__init__()

        self.max_frontier = max_frontier
        self.max_degree = max_degree
        self.max_candidates = max_candidates

        self.frontier_node_hidden_size = frontier_node_hidden_size
        self.frontier_edge_hidden_size = frontier_edge_hidden_size
        self.frontier_context_hidden_size = frontier_context_hidden_size
        self.target_hidden_size = target_hidden_size
        self.n_neighbors = n_neighbors
        self.pos_encoder = FourierEncoder2D(num_bands=pos_bands, include_input=True)

        if isinstance(frontier_pipeline, str):
            frontier_pipeline = [stage.strip() for stage in frontier_pipeline.split(",") if stage.strip()]
        self.frontier_pipeline_names = tuple(frontier_pipeline)
        if not self.frontier_pipeline_names:
            raise ValueError("frontier_pipeline must contain at least one stage.")
        if self.frontier_pipeline_names[0] != "init":
            raise ValueError("frontier_pipeline must start with 'init'.")

        stages = []
        for stage in self.frontier_pipeline_names:
            if stage == "init":
                stages.append(FrontierInitStage(
                    node_hidden_size=frontier_node_hidden_size,
                    edge_hidden_size=frontier_edge_hidden_size,
                    pos_bands=pos_bands,
                    n_neighbors=n_neighbors,
                    node_position=frontier_init_node_position,
                    node_ring=frontier_init_node_ring,
                    edge_midpoint=frontier_init_edge_midpoint,
                    edge_length=frontier_init_edge_length,
                    edge_direction=frontier_init_edge_direction,
                    edge_flags=frontier_init_edge_flags,
                ))
            elif stage == "perceiver":
                stages.append(FrontierPerceiverStage(
                    node_hidden_size=frontier_node_hidden_size,
                    edge_hidden_size=frontier_edge_hidden_size,
                    context_hidden_size=frontier_context_hidden_size,
                    d_model=frontier_perceiver_d_model,
                    num_latents=frontier_perceiver_num_latents,
                    num_layers=frontier_perceiver_layers,
                    num_heads=frontier_perceiver_heads,
                    mlp_ratio=frontier_perceiver_mlp_ratio,
                    dropout=frontier_perceiver_dropout,
                    decode_node=frontier_perceiver_decode_node,
                    decode_edge=frontier_perceiver_decode_edge,
                    update_context=frontier_perceiver_update_context,
                ))
            elif stage == "se2":
                stages.append(SE2FrontierStage(
                    node_hidden_size=frontier_node_hidden_size,
                    edge_hidden_size=frontier_edge_hidden_size,
                    num_layers=frontier_se2_layers,
                ))
            else:
                raise ValueError(f"Unknown frontier stage: {stage!r}.")
        self.frontier_pipeline = nn.ModuleList(stages)

        self.target_encoder = layer_init(nn.Linear(
            in_features=self.pos_encoder.out_dim + 2,  # (4*bands+2) + dist(1) + is_boundary(1)
            out_features=target_hidden_size,
        ))

    def _encode_sources(self, graph: CSRGraph):
        state = FrontierState(
            graph=graph,
            node_features=graph.vertices.new_empty(0, self.frontier_node_hidden_size),
            edge_features=graph.vertices.new_empty(0, self.frontier_edge_hidden_size),
            context_features=graph.vertices.new_empty(0, self.frontier_context_hidden_size),
        )
        for name, stage in zip(self.frontier_pipeline_names, self.frontier_pipeline):
            with nvtx_range(f"frontier_stage_{name}"):
                state = stage(state)
        return state.node_features, state.context_features


    def forward(self, obs) -> QuadMeshEncoding:
        """
        Args:
            obs: uint8 tensor [B, obs_size]

        Returns:
            QuadMeshEncoding with initial hidden states
        """
        assert obs.dtype == torch.uint8
        device = obs.device
        B, _ = obs.shape

        substep = obs[:, 0]
        # assert torch.all(substep <= 1).item() # Ensure no invalid substeps

        B1 = int(torch.count_nonzero(substep).item())
        B0 = B - B1

        substep0_idx = torch.nonzero(substep == 0, as_tuple=False)
        substep1_idx = torch.nonzero(substep == 1, as_tuple=False)

        # SUBSTEP 0
        if B0 > 0:
            with nvtx_range("deserialize_observation (substep 0)"):
                graph = deserialize_observation(
                    obs,
                    substep0_idx,
                    D=self.max_degree,
                    F_CAP=self.max_frontier,
                    deserialize_candidates=False,
                    exact_output=True,
                    copy_obs=False,
                    use_cuda_graph=True,
                )

            h_source0, h_source0_context = self._encode_sources(graph)
            h_source0_batch_offset = graph.batch_offsets
        else:
            h_source0 = torch.empty(0, self.frontier_node_hidden_size, device=device)
            h_source0_context = torch.empty(0, self.frontier_context_hidden_size, device=device)
            h_source0_batch_offset = torch.zeros(1, device=device)


        # SUBSTEP 1
        if B1 > 0:
            with nvtx_range("deserialize_observation (substep 1)"):
                graph, targets = deserialize_observation(
                    obs,
                    substep1_idx,
                    D=self.max_degree,
                    F_CAP=self.max_frontier,
                    C_CAP=self.max_candidates,
                    deserialize_candidates=True,
                    exact_output=True,
                    copy_obs=False,
                    use_cuda_graph=True,
                )

            h_source1, _ = self._encode_sources(graph)
            h_source1_batch_offset = graph.batch_offsets

            source_idx = targets.source_idx
            global_source_idx = h_source1_batch_offset[:-1] + source_idx

            boundary_mask = targets.target_idx >= 0

            with nvtx_range("compute_ring_target_local_positions"):
                positions_local = compute_ring_target_local_positions(
                    graph,
                    targets.target_positions,
                    global_source_idx,
                    targets.target_batches,
                )

            with nvtx_range("pos_encoder"):
                encoded_pos = self.pos_encoder(positions_local) # (total_targets, 2 + 4*bands)

            with nvtx_range("distances"):
                distances = distance_to_graph_edges(
                    graph,
                    targets.target_positions,
                    targets.target_batch_offsets,
                    block_q=16,
                    block_e=64,
                )

            target_features = torch.cat(
                [
                    encoded_pos,
                    distances.unsqueeze(-1),
                    boundary_mask.to(encoded_pos.dtype).unsqueeze(-1),
                ],
                dim=-1,
            )

            with nvtx_range("source_encoder"):
                h_target_source = h_source1[global_source_idx][targets.target_batches]
            with nvtx_range("target_encoder"):
                h_target = self.target_encoder(target_features)
            h_target_batch_offset = targets.target_batch_offsets
        else:
            h_source1 = torch.empty(0, self.frontier_node_hidden_size, device=device)
            h_source1_batch_offset = torch.zeros(1, device=device)
            source_idx = torch.empty(0, device=device)
            h_target_source = torch.empty(0, self.frontier_node_hidden_size, device=device)
            h_target = torch.empty(0, self.target_encoder.out_features, device=device)
            h_target_batch_offset = torch.zeros(1, device=device)

        return QuadMeshEncoding(
            substep=substep,
            h_source0=h_source0,
            h_source1=h_source1,
            h_source0_context=h_source0_context,
            h_source0_batch_offset=h_source0_batch_offset,
            h_source1_batch_offset=h_source1_batch_offset,
            source_idx=source_idx,
            h_target_source=h_target_source,
            h_target=h_target,
            h_target_batch_offset=h_target_batch_offset,
        )


class QuadMeshingBoundaryNetwork(nn.Module):
    """
    Network for boundary-aware quad_meshing policy.

    Processes the encoded representations from QuadMeshingBoundaryEncoder to
    produce per-target scores and a per-agent value estimate.

    Args:
        frontier_node_hidden_size: Dimension of per-frontier-vertex embeddings
        frontier_context_hidden_size: Dimension of per-observation frontier context embeddings
        target_hidden_size: Dimension of hidden target embeddings
        num_layers: Number of MLP layers
    """

    def __init__(
        self,
        frontier_node_hidden_size,
        target_hidden_size,
        frontier_context_hidden_size=None,
        num_layers=2,
        **kwargs,
    ):
        super().__init__()
        if frontier_context_hidden_size is None:
            frontier_context_hidden_size = frontier_node_hidden_size
        self.frontier_node_hidden_size = frontier_node_hidden_size
        self.frontier_context_hidden_size = frontier_context_hidden_size
        self.target_hidden_size = target_hidden_size

        self.source_mlp = _MLP(
            frontier_node_hidden_size,
            frontier_node_hidden_size,
            frontier_node_hidden_size,
            num_layers,
            final_activation=True,
        )
        self.source_pool_context_mlp = _MLP(
            frontier_node_hidden_size,
            frontier_context_hidden_size,
            frontier_context_hidden_size,
            num_layers,
            final_activation=True,
        )
        self.source_context_mlp = _MLP(
            frontier_context_hidden_size,
            frontier_context_hidden_size,
            frontier_context_hidden_size,
            num_layers,
            final_activation=True,
        )

        pair_hidden_size = frontier_node_hidden_size + target_hidden_size
        self.target_mlp = _MLP(pair_hidden_size, pair_hidden_size, pair_hidden_size, num_layers, final_activation=True)


    def forward(self, encoded):
        return self.forward_train(encoded)

    @staticmethod
    def _mean_pool_by_offsets(h, offsets):
        counts = offsets[1:] - offsets[:-1]
        pooled = torch.zeros(counts.numel(), h.size(-1), device=h.device, dtype=h.dtype)
        if h.numel() == 0:
            return pooled

        batch = torch.repeat_interleave(torch.arange(counts.numel(), device=h.device), counts)
        pooled.index_add_(0, batch, h)
        return pooled / counts.clamp_min(1).to(h.dtype).unsqueeze(-1)


    def forward_train(self, encoded: QuadMeshEncoding) -> QuadMeshEncoding:
        """
        Args:
            encoded: QuadMeshEncoding with source and target hidden states

        Returns:
            QuadMeshEncoding with updated hidden states
        """
        h_source0 = encoded.h_source0
        h_source1 = encoded.h_source1
        h_source0_context = encoded.h_source0_context
        h_target_source = encoded.h_target_source
        h_target = encoded.h_target

        encoded.h_source0 = self.source_mlp(h_source0)
        if h_source0_context.numel() == 0:
            pooled = self._mean_pool_by_offsets(
                encoded.h_source0,
                encoded.h_source0_batch_offset,
            )
            encoded.h_source0_context = self.source_pool_context_mlp(pooled)
        else:
            encoded.h_source0_context = self.source_context_mlp(h_source0_context)
        encoded.h_source1 = self.source_mlp(h_source1)
        encoded.h_target_source = self.source_mlp(h_target_source)
        encoded.h_target = self.target_mlp(torch.cat([h_target_source, h_target], dim=-1))

        return encoded

    def forward_eval(self, encoded: QuadMeshEncoding, state):
        return self.forward_train(encoded), state


    def initial_state(self, batch_size, device):
        return ()


class QuadMeshingBoundaryDecoder(nn.Module):
    """
    Decoder for boundary-aware quad_meshing policy.

    Args:
        nvec: Number of action dimensions (always [2] for source/target)
        frontier_node_hidden_size: Dimension of per-frontier-vertex embeddings
        frontier_context_hidden_size: Dimension of per-observation frontier context embeddings
        target_hidden_size: Dimension of hidden target embeddings
    """
    def __init__(
        self,
        act_sizes,
        frontier_node_hidden_size,
        target_hidden_size,
        frontier_context_hidden_size=None,
        **kwargs,
    ):
        super().__init__()
        assert len(act_sizes) == 1, "QuadMeshingBoundaryDecoder expects nvec=[1]"
        if frontier_context_hidden_size is None:
            frontier_context_hidden_size = frontier_node_hidden_size

        self.frontier_node_hidden_size = frontier_node_hidden_size
        self.frontier_context_hidden_size = frontier_context_hidden_size
        self.target_hidden_size = target_hidden_size

        self.source_head = layer_init(nn.Linear(frontier_node_hidden_size, 1), 0.01)

        pair_hidden_size = frontier_node_hidden_size + target_hidden_size
        self.target_head = layer_init(nn.Linear(pair_hidden_size, 1), 0.01)

        self.source_value_head = layer_init(nn.Linear(frontier_context_hidden_size, 1), 1)
        self.target_value_head = layer_init(nn.Linear(frontier_node_hidden_size, 1), 1)


    def forward(self, encoded: QuadMeshEncoding):
        """
        Args:
            encoded: QuadMeshEncoding with source and target hidden states

        Returns:
            tuple of (logits, values)
            - logits: [B, L]
            - values: [B, 1] value estimates
        """
        substep = encoded.substep
        h_source0 = encoded.h_source0
        h_source1 = encoded.h_source1
        h_source0_context = encoded.h_source0_context
        h_source0_batch_offset = encoded.h_source0_batch_offset
        h_source1_batch_offset = encoded.h_source1_batch_offset
        source_idx = encoded.source_idx
        h_target = encoded.h_target
        h_target_batch_offset = encoded.h_target_batch_offset

        device = h_source0_batch_offset.device
        sources_per_batch = h_source0_batch_offset[1:] - h_source0_batch_offset[:-1]
        targets_per_batch = h_target_batch_offset[1:] - h_target_batch_offset[:-1]
        B  = substep.numel()
        B0 = sources_per_batch.numel()
        B1 = targets_per_batch.numel()
        L = int(max(
            sources_per_batch.max().item() if B0 > 0 else 1,
            targets_per_batch.max().item() if B1 > 0 else 1,
        ))

        logits = torch.full((B, L), -torch.inf, device=device)
        values = torch.zeros(B, 1, device=device)

        # SUBSTEP 0
        if B0 > 0:
            packed_logits = self.source_head(h_source0).squeeze(1)

            target_rows = (substep == 0).nonzero(as_tuple=True)[0] # [B0]
            target_rows_2d = target_rows[:, None].expand(B0, L)    # [B0, L]
            cols = torch.arange(L, device=device).expand(B0, L)    # [B0, L]
            mask = cols < sources_per_batch[:, None]               # [B0, L]

            idx = h_source0_batch_offset[:-1, None] + cols
            logits[target_rows_2d[mask], cols[mask]] = packed_logits[idx[mask]]

            values[substep == 0] = self.source_value_head(h_source0_context)

        # SUBSTEP 1
        if B1 > 0:
            packed_logits = self.target_head(h_target).squeeze(1)

            target_rows = (substep == 1).nonzero(as_tuple=True)[0] # [B1]
            target_rows_2d = target_rows[:, None].expand(B1, L)    # [B1, L]
            cols = torch.arange(L, device=device).expand(B1, L)    # [B1, L]
            mask = cols < targets_per_batch[:, None]               # [B1, L]

            idx = h_target_batch_offset[:-1, None] + cols
            logits[target_rows_2d[mask], cols[mask]] = packed_logits[idx[mask]]

            # Fix no valid options
            no_valid = targets_per_batch == 0                        # [B1]
            logits[target_rows[no_valid], 0] = 1.0

            h_source_per_batch = h_source1[h_source1_batch_offset[:-1] + source_idx] # [B, frontier_node_hidden_size]
            values[substep == 1] = self.target_value_head(h_source_per_batch)

        return logits, values
