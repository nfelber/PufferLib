import numpy as np
from typing import Dict
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

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

        sig = inspect.signature(encoder.forward)
        self.accepts_substep = (
            "substep" in sig.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD
                   for p in sig.parameters.values())
        )

    def initial_state(self, batch_size, device):
        return self.network.initial_state(batch_size, device)

    def forward_eval(self, x, state, substep):
        if self.accepts_substep:
            with nvtx_range("encoder_eval"):
                h = self.encoder(x, substep)
            with nvtx_range("network_eval"):
                h, state = self.network.forward_eval(h, state, substep)
            with nvtx_range("decoder_eval"):
                logits, values = self.decoder(h, substep)
        else:
            with nvtx_range("encoder_eval"):
                h = self.encoder(x)
            with nvtx_range("network_eval"):
                h, state = self.network.forward_eval(h, state)
            with nvtx_range("decoder_eval"):
                logits, values = self.decoder(h)

        return logits, values, state

    def forward(self, x, substep):
        B, TT = x.shape[:2]
        if self.accepts_substep:
            with nvtx_range("encoder_train"):
                h = self.encoder(x.reshape(B*TT, *x.shape[2:]), substep)
            with nvtx_range("network_train"):
                h = self.network.forward_train(h if isinstance(h, dict) else h.reshape(B, TT, -1), substep)
            with nvtx_range("decoder_train"):
                logits, values = self.decoder(h if isinstance(h, dict) else h.reshape(B*TT, -1), substep)
        else:
            with nvtx_range("encoder_train"):
                h = self.encoder(x.reshape(B*TT, *x.shape[2:]))
            with nvtx_range("network_train"):
                h = self.network.forward_train(h if isinstance(h, dict) else h.reshape(B, TT, -1))
            with nvtx_range("decoder_train"):
                logits, values = self.decoder(h if isinstance(h, dict) else h.reshape(B*TT, -1))

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

    @staticmethod
    def _make_csr_from_sources(src, num_nodes):
        """
        Sort directed edges by source node and construct CSR pointer.
        """
        perm = torch.argsort(src)
        src_sorted = src[perm]

        deg = torch.bincount(src, minlength=num_nodes)
        ptr = torch.zeros(num_nodes + 1, device=src.device, dtype=torch.long)
        ptr[1:] = torch.cumsum(deg, dim=0)

        return perm, src_sorted, deg, ptr

    def forward(self, h, x, edge_index, edge_ptr=None, edge_cont=None, edge_cat=None):
        """
        Args:
            h:          [N, node_dim]
            x:          [N, 2] coordinates
            edge_index: [2, E], directed edges i -> j
            edge_ptr:   [N + 1] CSR pointers for outgoing edges (optional; if None, computed from edge_index)
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

        # Sort edges by source so that outgoing neighbors of each node are contiguous.
        if edge_ptr is not None:
            # edge_ptr pre-computed: extract out-degrees from CSR pointers
            out_deg = edge_ptr[1:] - edge_ptr[:-1]
            perm = torch.arange(num_edges, device=device)
            ptr = edge_ptr
        else:
            perm, _, out_deg, ptr = self._make_csr_from_sources(src, num_nodes)

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
        edge_jk = perm[edge_jk_sorted_pos]

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

        # h_i receives m_ji, i.e. messages whose destination is i.
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


UINT16_MAX = 65535


@torch.no_grad()
def _u16_le(buf: torch.Tensor, byte_offset: int) -> torch.Tensor:
    """
    Decode little-endian uint16 from obs[:, byte_offset:byte_offset+2].

    Returns:
        long tensor [B]
    """
    lo = buf[:, byte_offset].to(torch.long)
    hi = buf[:, byte_offset + 1].to(torch.long)
    return lo | (hi << 8)


@torch.no_grad()
def _u16_le_offsets(buf: torch.Tensor, byte_offsets: torch.Tensor) -> torch.Tensor:
    """
    Decode little-endian uint16 from obs[:, byte_offset:byte_offset+2].

    Returns:
        long tensor [B]
    """
    rows = torch.arange(buf.shape[0], device=buf.device)
    lo = buf[rows, byte_offsets].to(torch.long)
    hi = buf[rows, byte_offsets + 1].to(torch.long)
    return lo | (hi << 8)
    

@torch.no_grad()
@torch.compile(fullgraph=True)
def _extract_slices(buf: torch.Tensor, start: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """
    Args:
        buf: [B, M]
        start: [B]
        length: [B]

    Returns:
        contiguous buffer of all slices [length.sum()]
    """
    idx = torch.arange(buf.size(1), device=buf.device)
    mask = (idx[None, :] >= start[:, None]) & (idx[None, :] < (start + length)[:, None])
    return buf[mask]


@torch.no_grad()
def deserialize_substep0_obs(obs):
    """
    Args:
        obs: uint8 tensor [num_agents, obs_size]

    Returns:
        dict with:
            x:                [total_nodes, 2] float32
            edge_index:       [2, total_edges] long
            edge_length:      [total_edges, 1] float32, edge length
            edge_incidence:   [total_edges, 2] bool, [incident_ccw, incident_cw]
            batch:            [total_nodes] long, graph id per node
            ptr:              [num_agents + 1] long, node offsets per graph
            edge_ptr:         [total_nodes + 1] long, CSR pointer for outgoing edges
    """

    # ------------------------------------------------------------------------
    # Layout:
    #   u16 frontier_size
    #   u16 max_degree
    #   frontier_size * (float x, float y)
    #   frontier_size * max_degree * (u16 neighbor index, u8 face_incidence)
    # ------------------------------------------------------------------------

    assert obs.dtype == torch.uint8
    device = obs.device
    B, obs_size = obs.shape

    frontier_size = _u16_le(obs, 0)      # [B]
    max_degree = _u16_le(obs, 2)         # [B]

    max_F = int(frontier_size.max().item())
    max_D = int(max_degree.max().item())

    node_offsets = torch.zeros(B + 1, device=device, dtype=torch.long)
    node_offsets[1:] = torch.cumsum(frontier_size, dim=0)

    total_nodes = int(node_offsets[-1].item())

    local_nodes = torch.arange(max_F, device=device)
    valid_nodes = local_nodes[None, :] < frontier_size[:, None]

    byte_idx = torch.arange(max_F * 8, device=device)[None, :]
    vertex_offsets = torch.full((B, 1), 4, device=device) + byte_idx

    vertex_bytes = obs.gather(1, vertex_offsets)
    vertex_xy = (
        vertex_bytes.contiguous()
        .view(torch.float32)
        .view(B, max_F, 2)
    )

    x = vertex_xy.reshape(B * max_F, 2)[valid_nodes.flatten()]

    batch = torch.repeat_interleave(
        torch.arange(B, device=device),
        frontier_size,
    )

    # ------------------------------------------------------------
    # Vectorized decode of directed neighbors.
    # ------------------------------------------------------------

    node_offsets = torch.empty(B + 1, device=device, dtype=torch.long)
    node_offsets[0] = 0
    node_offsets[1:] = torch.cumsum(frontier_size, dim=0)

    graph_grid = torch.arange(B, device=device)[:, None, None].expand(B, max_F, max_D)
    src_grid = torch.arange(max_F, device=device)[None, :, None].expand(B, max_F, max_D)
    slot_grid = torch.arange(max_D, device=device)[None, None, :].expand(B, max_F, max_D)

    valid_slots = (
        (src_grid < frontier_size[:, None, None])
        & (slot_grid < max_degree[:, None, None])
    )

    neigh_base = 4 + 8 * frontier_size

    off = (
        neigh_base[:, None, None]
        + 3 * (src_grid * max_degree[:, None, None] + slot_grid)
    )

    lo = obs[graph_grid, off].long()
    hi = obs[graph_grid, off + 1].long()
    face_all = obs[graph_grid, off + 2].long()

    nidx = lo | (hi << 8)
    valid_edges = valid_slots & (nidx != UINT16_MAX)

    graph_id = graph_grid[valid_edges]
    src_local = src_grid[valid_edges]
    dst_local = nidx[valid_edges]
    face = face_all[valid_edges]

    src = node_offsets[graph_id] + src_local
    dst = node_offsets[graph_id] + dst_local

    edge_index = torch.stack([src, dst], dim=0)

    edge_length = (x[dst] - x[src]).norm(dim=-1, keepdim=True)

    edge_incidence = torch.stack(
        [
            (face >> 1) & 1,  # incident_ccw
            face & 1,         # incident_cw
        ],
        dim=-1,
    ).bool()

    out_deg = valid_edges.sum(dim=2)[valid_nodes]
    edge_ptr = torch.empty(total_nodes + 1, device=device, dtype=torch.long)
    edge_ptr[0] = 0
    edge_ptr[1:] = torch.cumsum(out_deg, dim=0)

    return dict(
        x=x,
        edge_index=edge_index,
        edge_length=edge_length,
        edge_incidence=edge_incidence,
        batch=batch,
        ptr=node_offsets,
        edge_ptr=edge_ptr,
    )


# @torch.no_grad()
# def decode_substep0_obs_static(
#     obs: torch.Tensor,
#     max_F_cap: int,
#     max_D_cap: int,
# ) -> Dict[str, torch.Tensor]:
#     """
#     Shape-stable decoding helper.
#
#     This stage is intended for torch.compile. It returns fixed-capacity buffers
#     and masks, but does not compact variable-length outputs.
#
#     Args:
#         obs:
#             uint8 tensor [B, obs_size]
#         max_F_cap:
#             static maximum frontier size to decode
#         max_D_cap:
#             static maximum degree to decode
#
#     Returns:
#         dict with fixed-shape tensors:
#             x_padded:        [B, max_F_cap, 2] float32
#             node_mask:       [B, max_F_cap] bool
#             dst_local:       [B, max_F_cap, max_D_cap] long
#             face:            [B, max_F_cap, max_D_cap] long
#             edge_mask:       [B, max_F_cap, max_D_cap] bool
#             frontier_size:   [B] long
#             max_degree:      [B] long
#             node_offsets:    [B + 1] long
#             out_deg_padded:  [B, max_F_cap] long
#     """
#
#     assert obs.dtype == torch.uint8
#
#     device = obs.device
#     B, _ = obs.shape
#
#     frontier_size = _u16_le(obs, 0)      # [B]
#     max_degree = _u16_le(obs, 2)         # [B]
#
#     # ---------------------------------------------------------------------
#     # Decode vertices into fixed-capacity [B, max_F_cap, 2].
#     #
#     # Layout:
#     #   u16 frontier_size
#     #   u16 max_degree
#     #   frontier_size * (float x, float y)
#     #
#     # We decode max_F_cap vertices per graph. Entries beyond frontier_size are
#     # garbage/padding and must be ignored using node_mask.
#     # ---------------------------------------------------------------------
#
#     local_nodes = torch.arange(max_F_cap, device=device)
#
#     node_mask = local_nodes[None, :] < frontier_size[:, None]  # [B, F]
#
#     vertex_byte_offsets = 4 + torch.arange(
#         max_F_cap * 8,
#         device=device,
#         dtype=torch.long,
#     )  # [F * 8]
#
#     vertex_bytes = obs.gather(
#         dim=1,
#         index=vertex_byte_offsets[None, :].expand(B, max_F_cap * 8),
#     )
#
#     # Reinterpret groups of 4 uint8 bytes as float32.
#     x_padded = (
#         vertex_bytes.contiguous()
#         .view(torch.float32)
#         .view(B, max_F_cap, 2)
#     )
#
#     # ---------------------------------------------------------------------
#     # Decode directed neighbor slots into fixed-capacity tensors.
#     #
#     # Layout after vertices:
#     #   frontier_size * max_degree * (u16 neighbor index, u8 face_incidence)
#     #
#     # We decode [B, max_F_cap, max_D_cap]. Invalid slots are ignored using
#     # edge_mask.
#     # ---------------------------------------------------------------------
#
#     src_grid = torch.arange(
#         max_F_cap,
#         device=device,
#         dtype=torch.long,
#     )[None, :, None]  # [1, F, 1]
#
#     slot_grid = torch.arange(
#         max_D_cap,
#         device=device,
#         dtype=torch.long,
#     )[None, None, :]  # [1, 1, D]
#
#     src_grid = src_grid.expand(B, max_F_cap, max_D_cap)
#     slot_grid = slot_grid.expand(B, max_F_cap, max_D_cap)
#
#     valid_slots = (
#         (src_grid < frontier_size[:, None, None])
#         & (slot_grid < max_degree[:, None, None])
#     )
#
#     neigh_base = 4 + 8 * frontier_size  # [B]
#
#     # Per-row byte offsets for each neighbor slot.
#     off = (
#         neigh_base[:, None, None]
#         + 3 * (src_grid * max_degree[:, None, None] + slot_grid)
#     )  # [B, F, D]
#
#     off_flat = off.reshape(B, max_F_cap * max_D_cap)
#
#     lo = obs.gather(1, off_flat).reshape(B, max_F_cap, max_D_cap).to(torch.long)
#     hi = obs.gather(1, off_flat + 1).reshape(B, max_F_cap, max_D_cap).to(torch.long)
#     face = obs.gather(1, off_flat + 2).reshape(B, max_F_cap, max_D_cap).to(torch.long)
#
#     dst_local = lo | (hi << 8)
#
#     # Include dst_local < frontier_size as a defensive validity check.
#     # If your serialized data guarantees this, it should not change semantics.
#     edge_mask = (
#         valid_slots
#         & (dst_local != UINT16_MAX)
#         & (dst_local < frontier_size[:, None, None])
#     )
#
#     node_offsets = torch.empty(B + 1, device=device, dtype=torch.long)
#     node_offsets[0] = 0
#     node_offsets[1:] = torch.cumsum(frontier_size, dim=0)
#
#     out_deg_padded = edge_mask.sum(dim=2).to(torch.long)  # [B, F]
#
#     return {
#         "x_padded": x_padded,
#         "node_mask": node_mask,
#         "dst_local": dst_local,
#         "face": face,
#         "edge_mask": edge_mask,
#         "frontier_size": frontier_size,
#         "max_degree": max_degree,
#         "node_offsets": node_offsets,
#         "out_deg_padded": out_deg_padded,
#     }
#
#
# @torch.no_grad()
# def compact_decoded_substep0_obs(
#     decoded: Dict[str, torch.Tensor],
# ) -> Dict[str, torch.Tensor]:
#     """
#     Late compaction stage.
#
#     This takes the fixed-capacity decoded representation and produces the same
#     compact outputs as your original function.
#
#     This stage is intentionally allowed to be dynamic.
#     """
#
#     x_padded = decoded["x_padded"]                  # [B, F, 2]
#     node_mask = decoded["node_mask"]                # [B, F]
#     dst_local_padded = decoded["dst_local"]         # [B, F, D]
#     face_padded = decoded["face"]                   # [B, F, D]
#     edge_mask = decoded["edge_mask"]                # [B, F, D]
#     frontier_size = decoded["frontier_size"]        # [B]
#     node_offsets = decoded["node_offsets"]          # [B + 1]
#     out_deg_padded = decoded["out_deg_padded"]      # [B, F]
#
#     device = x_padded.device
#     B, F, _ = x_padded.shape
#     D = dst_local_padded.shape[2]
#
#     # ---------------------------------------------------------------------
#     # Compact nodes.
#     # ---------------------------------------------------------------------
#
#     x = x_padded.reshape(B * F, 2)[node_mask.flatten()]
#
#     batch = torch.repeat_interleave(
#         torch.arange(B, device=device, dtype=torch.long),
#         frontier_size,
#     )
#
#     # ---------------------------------------------------------------------
#     # Compact edges.
#     # ---------------------------------------------------------------------
#
#     graph_grid = torch.arange(
#         B,
#         device=device,
#         dtype=torch.long,
#     )[:, None, None].expand(B, F, D)
#
#     src_grid = torch.arange(
#         F,
#         device=device,
#         dtype=torch.long,
#     )[None, :, None].expand(B, F, D)
#
#     graph_id = graph_grid[edge_mask]
#     src_local = src_grid[edge_mask]
#     dst_local = dst_local_padded[edge_mask]
#     face = face_padded[edge_mask]
#
#     src = node_offsets[graph_id] + src_local
#     dst = node_offsets[graph_id] + dst_local
#
#     edge_index = torch.stack([src, dst], dim=0)
#
#     edge_length = (x[dst] - x[src]).norm(dim=-1, keepdim=True)
#
#     edge_incidence = torch.stack(
#         [
#             ((face >> 1) & 1),
#             (face & 1),
#         ],
#         dim=-1,
#     ).bool()
#
#     # ---------------------------------------------------------------------
#     # CSR pointer over compact nodes.
#     # ---------------------------------------------------------------------
#
#     out_deg = out_deg_padded[node_mask]
#
#     total_nodes = x.shape[0]
#
#     edge_ptr = torch.empty(total_nodes + 1, device=device, dtype=torch.long)
#     edge_ptr[0] = 0
#     edge_ptr[1:] = torch.cumsum(out_deg, dim=0)
#
#     return {
#         "x": x,
#         "edge_index": edge_index,
#         "edge_length": edge_length,
#         "edge_incidence": edge_incidence,
#         "batch": batch,
#         "ptr": node_offsets,
#         "edge_ptr": edge_ptr,
#     }
#
#
# def make_compiled_substep0_decoder(
#     max_F_cap: int,
#     max_D_cap: int,
#     *,
#     fullgraph: bool = False,
# ):
#     """
#     Build a compiled decoder specialized to fixed capacities.
#
#     Use fullgraph=False first. After it works, try fullgraph=True to force
#     PyTorch to tell you whether anything still causes a graph break.
#     """
#
#     @torch.no_grad()
#     def _decode(obs: torch.Tensor) -> Dict[str, torch.Tensor]:
#         return decode_substep0_obs_static(
#             obs,
#             max_F_cap=max_F_cap,
#             max_D_cap=max_D_cap,
#         )
#
#     return torch.compile(
#         _decode,
#         mode="reduce-overhead",
#         fullgraph=fullgraph,
#     )
#
#
# @torch.no_grad()
# def deserialize_substep0_obs_hybrid(
#     obs: torch.Tensor,
#     compiled_decode,
# ) -> Dict[str, torch.Tensor]:
#     """
#     Drop-in-ish replacement for the original deserialize_substep0_obs.
#
#     The decode stage is compiled and shape-stable.
#     The compaction stage is dynamic and runs once at the end.
#     """
#
#     decoded = compiled_decode(obs)
#     return compact_decoded_substep0_obs(decoded)


@torch.no_grad()
def deserialize_substep1_obs(obs):
    """
    Deserialize the validity mask and new vertex candidates for substep 1 observations.

    Args:
        obs: uint8 tensor [num_agents, obs_size]

    Returns:
        dict with:
            source_idx:    [num_agents] int64, source vertex index in frontier
            validity:      [total_nodes] bool
            candidates:    [total_candidates, 2] float32
            counts:        [num_agents] long, candidate count per agent
    """

    assert obs.dtype == torch.uint8
    device = obs.device
    B, obs_size = obs.shape

    frontier_size = _u16_le(obs, 0)
    max_degree = _u16_le(obs, 2)

    max_F = int(frontier_size.max().item())

    neighbor_start = 4 + 8 * frontier_size
    source_idx_pos = neighbor_start + frontier_size * max_degree * 3
    validity_start = source_idx_pos + 2
    source_idx = _u16_le_offsets(obs, source_idx_pos)

    f_idx = torch.arange(max_F, device=device)[None, :]
    valid_nodes = f_idx < frontier_size[:, None]

    validity_offsets = validity_start[:, None] + f_idx
    validity_bytes = obs.gather(1, validity_offsets.clamp_max(obs_size - 1))
    validity_mask = validity_bytes.bool()[valid_nodes]

    candidates_start = validity_start + frontier_size

    valid_count = _u16_le_offsets(obs, candidates_start)
    max_V = int(valid_count.max().item())

    if max_V > 0:
        byte_idx = torch.arange(max_V * 8, device=device)[None, :]
        candidate_offsets = candidates_start[:, None] + 2 + byte_idx

        candidate_bytes = obs.gather(1, candidate_offsets)
        candidates = (
            candidate_bytes.contiguous()
            .view(torch.float32)
            .view(B, max_V, 2)
        )

        valid_cands = torch.arange(max_V, device=device)[None, :] < valid_count[:, None]
        candidates_flat = candidates[valid_cands]
    else:
        candidates_flat = torch.empty(0, 2, dtype=torch.float32, device=device)

    return dict(
        source_idx=source_idx,
        validity=validity_mask,
        candidates=candidates_flat,
        counts=valid_count,
    )


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


@torch.no_grad()
def _compute_interior_angles(
    x,
    edge_index,
    edge_ptr,
    edge_ccw,
    eps=1e-8,
):
    """
    Orientation-aware interior angle computation for CCW boundary topology.

    Args:
        x: [N, 2] vertex coordinates
        edge_index: [2, E] directed edges
        edge_ptr: [N + 1] CSR pointer for outgoing edges
        edge_ccw: [E] bool, True iff directed edge follows CCW boundary order

    Returns:
        cos_sin: [N, 2], where cos_sin[v] = (cos(theta_v), sin(theta_v))
                 for the CCW-boundary interior angle at vertex v.
    """
    N = x.shape[0]
    cos_sin = x.new_zeros((N, 2))

    if N == 0 or edge_index.numel() == 0:
        return cos_sin

    deg = edge_ptr[1:] - edge_ptr[:-1]
    # if (deg.min().item() < 2 or deg.max().item() > 2):
    #     print(f"deg min: {deg.min()}, deg max: {deg.max()}")
    #     import time
    #     time.sleep(300)
    assert torch.all(deg == 2).item()

    # Assumes the first two outgoing edges are the two ring neighbors.
    idx_a = edge_ptr[:-1] # [N]
    idx_b = idx_a + 1

    ccw_a = edge_ccw[idx_a]
    ccw_b = edge_ccw[idx_b]

    # We need exactly one CCW and one non-CCW edge among the two neighbors.
    assert torch.all(ccw_a ^ ccw_b).item()

    # Edge marked CCW gives the "next" vertex along the boundary.
    idx_next = torch.where(ccw_a, idx_a, idx_b)

    # Other edge gives the "prev" vertex.
    idx_prev = torch.where(ccw_a, idx_b, idx_a)

    n_next = edge_index[1, idx_next]
    n_prev = edge_index[1, idx_prev]

    d_next = x[n_next] - x
    d_prev = x[n_prev] - x

    d_next = F.normalize(d_next, p=2, dim=1, eps=eps)
    d_prev = F.normalize(d_prev, p=2, dim=1, eps=eps)

    cos_val = (d_next * d_prev).sum(dim=1)

    # 2D cross product z-component: d_next x d_prev
    sin_val = (
        d_next[:, 0] * d_prev[:, 1] -
        d_next[:, 1] * d_prev[:, 0]
    )

    cos_sin[:, 0] = cos_val
    cos_sin[:, 1] = sin_val

    return cos_sin

@torch.no_grad()
def _compute_interior_angles_compilable(
    x,
    edge_index,
    edge_ptr,
    edge_ccw,
    eps=1e-8,
):
    idx_a = edge_ptr[:-1]
    idx_b = idx_a + 1

    ccw_a = edge_ccw[idx_a]
    ccw_b = edge_ccw[idx_b]

    idx_next = torch.where(ccw_a, idx_a, idx_b)
    idx_prev = torch.where(ccw_a, idx_b, idx_a)

    n_next = edge_index[1, idx_next]
    n_prev = edge_index[1, idx_prev]

    d_next = F.normalize(x[n_next] - x, p=2, dim=1, eps=eps)
    d_prev = F.normalize(x[n_prev] - x, p=2, dim=1, eps=eps)

    cos_val = (d_next * d_prev).sum(dim=1)
    sin_val = d_next[:, 0] * d_prev[:, 1] - d_next[:, 1] * d_prev[:, 0]

    return torch.stack((cos_val, sin_val), dim=1)


@torch.no_grad()
def _boundary_k_neighbors(
    x,
    edge_index,
    edge_ccw,
    k=3,
):
    """
    Return k-hop clockwise and counter-clockwise boundary neighbors for each vertex.

    Args:
        x: [V, 2] vertex coordinates
        edge_index: [2, E] directed edges. edge_index[0] is source, edge_index[1] is target.
        edge_ccw: [E] bool. True iff edge follows CCW boundary direction.
        k: number of hops in each direction.

    Returns:
        neighbors: [V, 2 * k] long tensor.
        For k = 3, the columns are:
            [cw_3, cw_2, cw_1, ccw_1, ccw_2, ccw_3]
        That is, it starts with the 3-hop CW neighbor and ends with the 3-hop CCW neighbor.

        edges: [V, 2 * k] long tensor.
        For k = 3, the columns are:
            [cw_2 -> cw_3, cw_1 -> cw_2, x -> cw_1, x -> ccw_1, ccw1 -> ccw_2, ccw2 -> ccw_3]
    """
    device = edge_index.device
    V = x.shape[0]

    if V == 0 or k == 0:
        empty = torch.empty((V, 2 * k), device=device, dtype=torch.long)
        return empty, empty

    src = edge_index[0]
    dst = edge_index[1]
    E = edge_index.shape[1]

    edge_ids = torch.arange(E, device=device, dtype=torch.long)

    next_cw_edge = torch.empty((V,), device=device, dtype=torch.long)
    next_ccw_edge = torch.empty((V,), device=device, dtype=torch.long)

    # edge_ccw=True means src -> dst is the CCW successor edge.
    next_ccw_edge[src[edge_ccw]] = edge_ids[edge_ccw]

    # edge_ccw=False means src -> dst is the CW successor edge.
    next_cw_edge[src[~edge_ccw]] = edge_ids[~edge_ccw]

    vertices = torch.arange(V, device=device, dtype=torch.long)

    def follow_edges(successor_edge):
        """
        Returns:
            hop_vertices: [V, k], columns [hop_1, hop_2, ..., hop_k]
            hop_edges:    [V, k], where hop_edges[:, i] is the edge used
                          to reach hop_vertices[:, i].
        """
        hop_vertices = []
        hop_edges = []

        cur = vertices

        for _ in range(k):
            edge = successor_edge[cur]
            cur = dst[edge]

            hop_edges.append(edge)
            hop_vertices.append(cur)

        return (
            torch.stack(hop_vertices, dim=1),
            torch.stack(hop_edges, dim=1),
        )

    cw_vertices, cw_edges = follow_edges(next_cw_edge)
    ccw_vertices, ccw_edges = follow_edges(next_ccw_edge)

    # Desired order:
    # [cw_k, ..., cw_1, ccw_1, ..., ccw_k]
    neighbors = torch.cat(
        [
            torch.flip(cw_vertices, dims=[1]),
            ccw_vertices,
        ],
        dim=1,
    )

    # Desired edge order:
    # [cw_{k-1} -> cw_k, ..., x -> cw_1, x -> ccw_1, ..., ccw_{k-1} -> ccw_k]
    edges = torch.cat(
        [
            torch.flip(cw_edges, dims=[1]),
            ccw_edges,
        ],
        dim=1,
    )

    return neighbors, edges


@torch.no_grad()
def _compute_boundary_local_frames_2d(
    x,
    edge_index,
    edge_ccw,
    eps=1e-8,
):
    """
    Compute one 2D local frame per boundary vertex.

    The local x-axis points along the interior angle bisector, towards the
    interior, assuming the boundary orientation is CCW.

    Args:
        x: [V, 2] vertex coordinates.
        edge_index: [2, E] directed edges.
            edge_index[0] = source vertex
            edge_index[1] = target vertex
        edge_ccw: [E] bool.
            True iff the directed edge follows the CCW boundary direction.
            False iff the directed edge follows the CW boundary direction.
        eps: numerical epsilon.

    Returns:
        basis: [V, 2, 2]
            basis[v, :, 0] = local x-axis in global coordinates
            basis[v, :, 1] = local y-axis in global coordinates

    Convention:
        local x-axis = inward angle bisector.
        local y-axis = +90 degree rotation of local x-axis.
    """
    V = x.shape[0]
    device = x.device

    basis = x.new_zeros((V, 2, 2))

    if V == 0:
        return basis

    src = edge_index[0]
    dst = edge_index[1]

    # next_ccw[v] = next vertex when walking CCW along the boundary
    # next_cw[v]  = next vertex when walking CW along the boundary
    next_ccw = torch.full((V,), -1, device=device, dtype=torch.long)
    next_cw = torch.full((V,), -1, device=device, dtype=torch.long)
    next_ccw[src[edge_ccw]] = dst[edge_ccw]
    next_cw[src[~edge_ccw]] = dst[~edge_ccw]

    assert torch.all((next_ccw >= 0) & (next_cw >= 0)).item() # Each vertex has 2 neighbors

    x_next = x[next_ccw]
    x_prev = x[next_cw]

    # Tangent of incoming and outgoing CCW edges
    t_in = F.normalize(x - x_prev, p=2, dim=1, eps=eps)
    t_out = F.normalize(x_next - x, p=2, dim=1, eps=eps)

    # For a CCW boundary, the interior lies to the left of each boundary tangent.
    #
    # left([x, y]) = [-y, x]
    #
    # Averaging the inward normals gives an interior bisector that works for both
    # convex and reflex vertices.
    n_in = torch.stack([-t_in[:, 1], t_in[:, 0]], dim=1)
    n_out = torch.stack([-t_out[:, 1], t_out[:, 0]], dim=1)

    x_axis = F.normalize(n_in + n_out, p=2, dim=1, eps=eps)

    # Local y-axis: +90 degree rotation of local x-axis.
    y_axis = torch.stack([-x_axis[:, 1], x_axis[:, 0]], dim=1)

    basis[:, :, 0] = x_axis
    basis[:, :, 1] = y_axis

    return basis


def _global_to_local(points_global, origins, basis):
    """
    Transform global 2D points into per-vertex local frames.

    Args:
        points_global: [V, ..., 2]
        origins: [V, 2]
        basis: [V, 2, 2]
            basis[v, :, 0] = local x-axis in global coords
            basis[v, :, 1] = local y-axis in global coords

    Returns:
        points_local with same leading shape as points_global.
    """
    # points_global shape: [V, ..., 2]
    extra_dims = points_global.ndim - 2

    o = origins.reshape(origins.shape[0], *([1] * extra_dims), 2)
    R = basis.reshape(basis.shape[0], *([1] * extra_dims), 2, 2)

    centered = points_global - o

    # Since basis columns are local axes in global coordinates:
    # local = centered @ basis
    return torch.matmul(centered.unsqueeze(-2), R).squeeze(-2)


def _local_to_global(points_local, origins, basis):
    """
    Transform per-vertex local 2D points into global coordinates.

    Args:
        points_local: [V, ..., 2]
        origins: [V, 2]
        basis: [V, 2, 2]

    Returns:
        points_global with same leading shape as points_local.
    """
    # points_local shape: [V, ..., 2]
    extra_dims = points_local.ndim - 2

    o = origins.reshape(origins.shape[0], *([1] * extra_dims), 2)
    R_T = basis.transpose(-1, -2).reshape(
        basis.shape[0], *([1] * extra_dims), 2, 2
    )

    # global = local @ basis.T + origin
    return torch.matmul(points_local.unsqueeze(-2), R_T).squeeze(-2) + o


def _pack_candidate_indices(
    node_validity: torch.Tensor,         # [N] bool
    new_candidate_counts: torch.Tensor,  # [B]
    node_ptr: torch.Tensor,              # [B + 1]
):
    device = node_validity.device
    B = new_candidate_counts.numel()
    N = node_ptr[1:] - node_ptr[:-1]  # [B]

    node_batch = torch.repeat_interleave(
        torch.arange(B, device=device),
        N,
    )  # [N]

    valid_counts = torch.zeros(B, device=device, dtype=torch.long)
    valid_counts.scatter_add_(0, node_batch, node_validity.long())

    out_counts = valid_counts + new_candidate_counts

    out_ptr = torch.empty(B + 1, device=device, dtype=torch.long)
    out_ptr[0] = 0
    out_ptr[1:] = out_counts.cumsum(dim=0)

    total_out = int(out_ptr[-1].item())
    idx = torch.full((total_out,), -1, device=device, dtype=torch.long)

    # Global indices of valid nodes
    node_idx = torch.arange(node_validity.numel(), device=device)
    valid_node_idx = node_idx[node_validity]
    valid_node_batch = node_batch[node_validity]

    # Rank of each valid node within its batch
    valid_cumsum = node_validity.long().cumsum(dim=0) - 1

    valid_ptr = torch.empty(B + 1, device=device, dtype=torch.long)
    valid_ptr[0] = 0
    valid_ptr[1:] = valid_counts.cumsum(dim=0)

    valid_rank_in_batch = valid_cumsum[node_validity] - valid_ptr[valid_node_batch]

    valid_out_idx = out_ptr[valid_node_batch] + valid_rank_in_batch

    idx[valid_out_idx] = valid_node_idx

    return idx, out_ptr


class QuadMeshingBoundaryEncoder(nn.Module):
    """
    Boundary-aware encoder for quad_meshing in boundary mode.

    Args:
        source_hidden_size: Dimension of hidden source embeddings
        target_hidden_size: Dimension of hidden target embeddings
        pos_bands: Number of Fourier frequency bands
        n_neighbors: Number of boundary neighbors to embed in each direction
    """

    def __init__(self, obs_size, source_hidden_size=128, target_hidden_size=128, pos_bands=6, n_neighbors=3, **kwargs):
        super().__init__()
        self.source_hidden_size = source_hidden_size
        self.target_hidden_size = target_hidden_size
        self.n_neighbors = n_neighbors
        self.pos_encoder = FourierEncoder2D(num_bands=pos_bands, include_input=True)

        # Use same embedding for source selection and source observation
        self.source_encoder = layer_init(nn.Linear(
            in_features=2 + 2 * n_neighbors * 3,  # cos_sin(2) + 2 * n_neighbors * (cos_sin(2) + edge_length(1))
            out_features=source_hidden_size,
        ))

        self.target_encoder = layer_init(nn.Linear(
            in_features=self.pos_encoder.out_dim + 1,  # (4*bands+2) + is_boundary(1)
            out_features=target_hidden_size,
        ))

        # self.compiled_decode = lambda obs: decode_substep0_obs_static(obs, 1024, 8)
        # self.compiled_decode = make_compiled_substep0_decoder(
        #     1024,
        #     8,
        #     fullgraph=False,
        # )

        # self.compiled_cia = torch.compile(
        #     _compute_interior_angles_compilable,
        #     mode="reduce-overhead",
        #     fullgraph=False,
        # )



    def forward(self, obs, substep):
        """
        Args:
            obs: uint8 tensor [B, obs_size]

        Returns:
            dict with:
                source_idx:      source index (substep 1)
                h_source:        [total_nodes, source_hidden_size], source vertex hidden states
                h_target_source: [total_targets, target_hidden_size], source vertex hidden states per target (substep 1)
                h_target:        [total_targets, target_hidden_size], target hidden states (substep 1)
                h_source_ptr:    [B + 1] h_source offsets per agent
                h_target_ptr:    [B + 1] h_target offsets per agent
        """
        assert obs.dtype == torch.uint8
        device = obs.device
        B, _ = obs.shape

        with nvtx_range("deserialize_substep0_obs"):
            obs0 = deserialize_substep0_obs(obs)

        V = obs0['x'].shape[0]
        K = 2 * self.n_neighbors

        edge_ccw = obs0['edge_incidence'][:, 1]
        with nvtx_range("compute_interior_angles"):
            angles = _compute_interior_angles(obs0['x'], obs0['edge_index'], obs0['edge_ptr'], edge_ccw) # [V, 2]
        with nvtx_range("boundary_k_neighbors"):
            neighbors, neigh_edges = _boundary_k_neighbors(obs0['x'], obs0['edge_index'], edge_ccw, self.n_neighbors) # [V, K]

        source_idx = None

        # [V, 2 + K * (2 + 1)]
        source_features = torch.cat(
            [
                angles,
                torch.cat(
                    [
                        angles[neighbors],                 # [V, K, 2]
                        obs0['edge_length'][neigh_edges],  # [V, K, 1]
                    ],
                    dim=-1,
                ).reshape(V, K * 3),
            ],
            dim=-1,
        )

        with nvtx_range("source_encoder"):
            h_source = self.source_encoder(source_features)
        h_source_ptr = obs0['ptr']
        h_target_source = None
        h_target = None
        h_target_ptr = None

        if substep == 1:
            with nvtx_range("deserialize_substep1_obs"):
                obs1 = deserialize_substep1_obs(obs)
            source_idx = obs1['source_idx']
            global_source_idx = h_source_ptr[:-1] + source_idx

            with nvtx_range("pack_candidate_indices"):
                idx, idx_ptr = _pack_candidate_indices(obs1['validity'], obs1['counts'], obs0['ptr'])
            targets_per_batch = idx_ptr[1:] - idx_ptr[:-1]
            target_batches = torch.repeat_interleave(torch.arange(B, device=device), targets_per_batch)

            src_angles = angles[global_source_idx]              # [B, 2]
            src_neighbors = neighbors[global_source_idx]        # [B, K]
            src_neigh_edges = neigh_edges[global_source_idx]    # [B, K]

            src_neighbor_angles = angles[src_neighbors]                   # [B, K, 2]
            src_neighbor_lengths = obs0['edge_length'][src_neigh_edges]   # [B, K, 1]

            # [total_targets, 2 + K * (2 + 1)]
            src_features = torch.cat(
                [
                    src_angles,
                    torch.cat(
                        [src_neighbor_angles, src_neighbor_lengths],
                        dim=-1,
                    ).reshape(B, -1),
                ],
                dim=-1,
            )[target_batches]

            boundary_mask = idx >= 0

            positions = torch.empty(idx.numel(), 2, device=device, dtype=torch.float32) # (total_targets, 2)
            positions[boundary_mask] = obs0['x'][idx[boundary_mask]]
            positions[~boundary_mask] = obs1['candidates']

            with nvtx_range("compute_boundary_local_frames_2d"):
                basis = _compute_boundary_local_frames_2d(obs0['x'], obs0['edge_index'], edge_ccw) # [V, 2, 2]
            src_nodes = obs0['x'][global_source_idx] # [B, 2]
            src_basis = basis[global_source_idx]     # [B, 2, 2]
            with nvtx_range("global_to_local"):
                positions_local = _global_to_local(positions, src_nodes[target_batches], src_basis[target_batches])

            with nvtx_range("pos_encoder"):
                encoded_pos = self.pos_encoder(positions_local) # (total_targets, 2 + 4*bands)

            # (4*bands+2) + is_boundary(1)
            target_features = torch.cat(
                [
                    encoded_pos,
                    boundary_mask.to(encoded_pos.dtype).unsqueeze(-1),
                ],
                dim=-1,
            )

            # for i in range(obs0['x'].shape[0]):
            #     print(f"{i}: ({angles[i][0]:.4f}, {angles[i][1]:.4f})")
            # for i in range(obs0['x'].shape[0]):
            #     print(f"{i}: ({basis[i][0][0]:.4f}, {basis[i][1][0]:.4f}), ({basis[i][0][1]:.4f}, {basis[i][1][1]:.4f})")

            # for i in range(obs1['validity'].shape[0]):
            #     print(f"{i}: {obs1['validity'][i]}")
            #
            # print(f"source idx: {source_idx.item()}, p: ({obs0['x'][source_idx][0][0]}, {obs0['x'][source_idx][0][1]})")
            # for i in range(positions.shape[0]):
            #     print(f"p: ({positions[i][0]:.4f}, {positions[i][1]:.4f}), lp: ({target_features[i][0]:.4f}, {target_features[i][1]:.4f}), is_bdry: {target_features[i][2]}, idx: {idx[i]}")

            # print(src_features)
            # import sys
            # sys.exit()

            with nvtx_range("source_encoder"):
                h_target_source = self.source_encoder(src_features)
            with nvtx_range("target_encoder"):
                h_target = self.target_encoder(target_features)
            h_target_ptr = idx_ptr
        elif substep > 1:
            # Substep must be in {0, 1}
            raise ValueError(f"Unknown substep: {substep}")

        return dict(
            source_idx=source_idx,
            h_source=h_source,
            h_target_source=h_target_source,
            h_target=h_target,
            h_source_ptr=h_source_ptr,
            h_target_ptr=h_target_ptr,
        )


class QuadMeshingBoundaryNetwork(nn.Module):
    """
    Network for boundary-aware quad_meshing policy.

    Processes the encoded representations from QuadMeshingBoundaryEncoder to
    produce per-target scores and a per-agent value estimate.

    Args:
        source_hidden_size: Dimension of hidden source embeddings
        target_hidden_size: Dimension of hidden target embeddings
        num_layers: Number of MLP layers
    """

    def __init__(self, source_hidden_size, target_hidden_size, num_layers=2, **kwargs):
        super().__init__()
        self.source_hidden_size = source_hidden_size
        self.target_hidden_size = target_hidden_size

        self.source_mlp = _MLP(source_hidden_size, source_hidden_size, source_hidden_size, num_layers, final_activation=True)

        pair_hidden_size = source_hidden_size + target_hidden_size
        self.target_mlp = _MLP(pair_hidden_size, pair_hidden_size, pair_hidden_size, num_layers, final_activation=True)


    def forward(self, encoded, substep):
        return self.forward_train(encoded, substep)


    def forward_train(self, encoded, substep):
        """
        Args:
            encoded:
                h_source:        [total_nodes, source_hidden_size], source vertex hidden states
                h_target_source: [total_targets, target_hidden_size], source vertex hidden states per target (substep 1)
                h_target:        [total_targets, target_hidden_size], target hidden states (substep 1)
                h_source_ptr:    [B + 1] h_source offsets per agent
                h_target_ptr:    [B + 1] h_target offsets per agent

        Returns:
            dict with:
                source_idx:      source index (substep 1)
                h_source:        [total_nodes, source_hidden_size], source vertex hidden states
                h_target_source: [total_targets, target_hidden_size], source vertex hidden states per target (substep 1)
                h_target:        [total_targets, source_hidden_size + target_hidden_size], target hidden states (substep 1)
                h_source_ptr:    [B + 1] h_source offsets per agent
                h_target_ptr:    [B + 1] h_target offsets per agent
        """
        h_source = encoded['h_source']
        h_target_source = encoded['h_target_source']
        h_target = encoded['h_target']

        h_source = self.source_mlp(h_source)
        encoded['h_source'] = h_source

        if substep == 1:
            h_target_source = self.source_mlp(h_target_source)
            h_target = self.target_mlp(torch.cat([h_target_source, h_target], dim=-1))
            encoded['h_target_source'] = h_target_source
            encoded['h_target'] = h_target
        elif substep > 1:
            # Substep must be in {0, 1}
            raise ValueError(f"Unknown substep: {substep}")

        return encoded

    def forward_eval(self, encoded, state, substep):
        return self.forward_train(encoded, substep), state


    def initial_state(self, batch_size, device):
        return ()


class QuadMeshingBoundaryDecoder(nn.Module):
    """
    Decoder for boundary-aware quad_meshing policy.

    Args:
        nvec: Number of action dimensions (always [2] for source/target)
        source_hidden_size: Dimension of hidden source embeddings
        target_hidden_size: Dimension of hidden target embeddings
    """
    def __init__(self, act_sizes, source_hidden_size, target_hidden_size, **kwargs):
        super().__init__()
        assert len(act_sizes) == 2, "QuadMeshingBoundaryDecoder expects nvec=[1, 1]"

        self.source_hidden_size = source_hidden_size
        self.target_hidden_size = target_hidden_size

        self.source_head = layer_init(nn.Linear(source_hidden_size, 1), 0.01)

        pair_hidden_size = source_hidden_size + target_hidden_size
        self.target_head = layer_init(nn.Linear(pair_hidden_size, 1), 0.01)

        self.value_head = layer_init(nn.Linear(source_hidden_size, 1), 1)


    def forward(self, encoded, substep):
        """
        Args:
            encoded:
                source_idx:      source index (substep 1)
                h_source:        [total_nodes, source_hidden_size], source vertex hidden states
                h_target_source: [total_targets, target_hidden_size], source vertex hidden states per target (substep 1)
                h_target:        [total_targets, source_hidden_size + target_hidden_size], target hidden states (substep 1)
                h_source_ptr:    [B + 1] h_source offsets per agent
                h_target_ptr:    [B + 1] h_target offsets per agent

        Returns:
            tuple of (logits, values)
            - logits: ([B, S], [B, 1]) (substep 0)
                      ([B, 1], [B, T]) (substep 1)
            - values: [B] value estimates
        """
        source_idx = encoded['source_idx']
        h_source = encoded['h_source']
        h_target = encoded['h_target']
        h_source_ptr = encoded['h_source_ptr']
        h_target_ptr = encoded['h_target_ptr']

        device = h_source.device

        values = None

        if substep == 0:
            packed_logits = self.source_head(h_source).squeeze(1)

            sources_per_batch = h_source_ptr[1:] - h_source_ptr[:-1]
            B = sources_per_batch.numel()
            S = sources_per_batch.max().item()

            source_logits = torch.full((B, S), -torch.inf, device=device)

            cols = torch.arange(S, device=device).expand(B, S)
            mask = cols < sources_per_batch[:, None]

            idx = h_source_ptr[:-1, None] + cols
            source_logits[mask] = packed_logits[idx[mask]]
            target_logits = torch.ones(B, 1, device=device, dtype=torch.float32)
        elif substep == 1:
            packed_logits = self.target_head(h_target).squeeze(1)

            targets_per_batch = h_target_ptr[1:] - h_target_ptr[:-1]
            B = targets_per_batch.numel()
            T = max(1, targets_per_batch.max().item())
            S = source_idx.max().item() + 1

            target_logits = torch.full((B, T), -torch.inf, device=device)

            cols = torch.arange(T, device=device).expand(B, T)
            mask = cols < targets_per_batch[:, None]

            idx = h_target_ptr[:-1, None] + cols
            target_logits[mask] = packed_logits[idx[mask]]
            target_logits[targets_per_batch == 0, 0] = 1.0 # Fix no valid options
            source_logits = torch.full((B, S), -torch.inf, device=device)
            rows = torch.arange(B, device=device)
            source_logits[rows, source_idx] = 1.0

            h_source_per_batch = h_source[h_source_ptr[:-1] + source_idx] # [B, source_hidden_size]
            # h_source_per_batch = h_source[source_idx] # [B, source_hidden_size]
            h_source_per_batch = torch.zeros_like(h_source_per_batch, device=device) # [B, source_hidden_size]
            values = self.value_head(h_source_per_batch) # [B, 1]
        else:
            raise ValueError(f"Unknown substep: {substep[0]}")

        return (source_logits, target_logits), values

