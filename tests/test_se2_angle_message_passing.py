import math
from contextlib import nullcontext

import pytest
import torch

import pufferlib.models as models
from pufferlib.models import (
    CSRGraph,
    CandidateTargets,
    QuadMeshEncoding,
    QuadMesh3DGraph,
    QuadMesh3DTargets,
    QuadMeshingDecoder,
    QuadMeshingEncoder,
    QuadMeshingNetwork,
    SE2AngleMessagePassing,
)


def make_directed_ring(num_nodes=8, device="cpu"):
    theta = torch.linspace(0, 2 * math.pi, num_nodes + 1, device=device)[:-1]
    x = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    src = torch.repeat_interleave(torch.arange(num_nodes, device=device), 2)
    dst = []
    edge_cont = []
    for i in range(num_nodes):
        neighbors = ((i + 1) % num_nodes, (i - 1) % num_nodes)
        for rank, j in enumerate(neighbors):
            dst.append(j)
            length = torch.linalg.vector_norm(x[j] - x[i])
            is_ccw = 1.0 if rank == 0 else 0.0
            is_cw = 1.0 if rank == 1 else 0.0
            edge_cont.append(torch.stack([length, x.new_tensor(is_ccw), x.new_tensor(is_cw)]))

    edge_index = torch.stack([src, torch.tensor(dst, device=device)])
    edge_ptr = torch.arange(0, 2 * num_nodes + 1, 2, device=device)
    edge_cont = torch.stack(edge_cont)
    return x, edge_index, edge_ptr, edge_cont


def make_csr_graph(vertices, batch_offsets, edges, edge_features, edge_ptr):
    return CSRGraph(
        vertices=vertices,
        batch_offsets=batch_offsets,
        edges=edges,
        edge_features=edge_features,
        edge_ptr=edge_ptr,
        target_edge_length=torch.ones(batch_offsets.numel() - 1, device=vertices.device, dtype=vertices.dtype),
    )


def make_quad_mesh_3d_graph():
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [1.0, 1.0, 0.2],
            [0.0, 1.0, 0.1],
        ],
        dtype=torch.float32,
    )
    normals = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 1.0],
            [0.1, 0.0, 1.0],
            [0.0, -0.1, 1.0],
        ],
        dtype=torch.float32,
    )
    normals = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
    edges = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [1, 3, 0, 2, 1, 3, 2, 0],
        ],
        dtype=torch.long,
    )
    edge_ptr = torch.tensor([0, 2, 4, 6, 8], dtype=torch.long)
    return QuadMesh3DGraph(
        vertices=vertices,
        normals=normals,
        batch_offsets=torch.tensor([0, vertices.size(0)], dtype=torch.long),
        edges=edges,
        edge_features=torch.empty(edges.size(1), 0, dtype=torch.bool),
        edge_ptr=edge_ptr,
        target_edge_length=torch.ones(1, dtype=torch.float32),
        suggested_vertex_idx=torch.tensor([0], dtype=torch.long),
    )


def make_quad_mesh_3d_targets():
    target_normals = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.2, 1.0],
        ],
        dtype=torch.float32,
    )
    target_normals = target_normals / torch.linalg.vector_norm(target_normals, dim=-1, keepdim=True)
    return QuadMesh3DTargets(
        source_idx=torch.tensor([0], dtype=torch.long),
        target_batch_offsets=torch.tensor([0, 2], dtype=torch.long),
        target_batches=torch.tensor([0, 0], dtype=torch.long),
        target_positions=torch.tensor([[0.7, 0.2, 0.05], [1.0, 1.0, 0.2]], dtype=torch.float32),
        target_normals=target_normals,
        path_lengths=torch.tensor([0.8, 1.5], dtype=torch.float32),
        target_kind=torch.tensor([0, 1], dtype=torch.long),
        target_frontier_parity=torch.tensor([True, False]),
    )


def make_layer(node_dim=5, out_dim=7, msg_dim=11):
    torch.manual_seed(0)
    return SE2AngleMessagePassing(
        node_dim=node_dim,
        out_dim=out_dim,
        msg_dim=msg_dim,
        edge_cont_dim=3,
        hidden_dim=16,
    )


def test_forward_shape_and_gradients():
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=8)
    h = torch.randn(x.size(0), 5, requires_grad=True)
    x = x.requires_grad_(True)
    edge_cont = edge_cont.requires_grad_(True)
    layer = make_layer()

    out = layer(h, x, edge_index, edge_ptr, edge_cont=edge_cont)

    assert out.shape == (x.size(0), 7)
    loss = out.square().mean()
    loss.backward()
    assert torch.isfinite(h.grad).all()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(edge_cont.grad).all()


def test_forward_is_translation_and_rotation_invariant():
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=10)
    h = torch.randn(x.size(0), 5)
    layer = make_layer()

    angle = 0.73
    rot = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=x.dtype,
    )
    x_transformed = x @ rot.T + torch.tensor([3.0, -2.0])

    out = layer(h, x, edge_index, edge_ptr, edge_cont=edge_cont)
    out_transformed = layer(h, x_transformed, edge_index, edge_ptr, edge_cont=edge_cont)

    torch.testing.assert_close(out_transformed, out, rtol=1e-5, atol=1e-6)


def test_degree_one_graph_has_valid_output():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_ptr = torch.tensor([0, 1, 2])
    edge_cont = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    h = torch.randn(2, 5)
    layer = make_layer()

    out = layer(h, x, edge_index, edge_ptr, edge_cont=edge_cont)

    assert out.shape == (2, 7)
    assert torch.isfinite(out).all()


def test_edge_ptr_is_required():
    x, edge_index, _, edge_cont = make_directed_ring(num_nodes=4)
    h = torch.randn(x.size(0), 5)
    layer = make_layer()

    with pytest.raises(ValueError, match="edge_ptr must be provided"):
        layer(h, x, edge_index, None, edge_cont=edge_cont)


def test_precomputed_triplet_topology_matches_default_forward():
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=8)
    h = torch.randn(x.size(0), 5)
    layer = make_layer()
    topology = models.compute_se2_triplet_topology(edge_index, edge_ptr, x.size(0))

    out_default = layer(h, x, edge_index, edge_ptr, edge_cont=edge_cont)
    out_precomputed = layer(h, x, edge_index, edge_ptr, edge_cont=edge_cont, topology=topology)

    torch.testing.assert_close(out_precomputed, out_default)


@pytest.mark.parametrize("chiral", [False, True])
def test_painn_forward_is_translation_rotation_equivariant(chiral):
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=10)
    h = torch.randn(x.size(0), 5)
    vectors = torch.randn(x.size(0), 4, 2)
    layer = models.SE2PaiNNMessagePassing2D(
        node_dim=5,
        out_dim=7,
        edge_cont_dim=3,
        vector_channels=4,
        hidden_dim=16,
        chiral=chiral,
    )

    angle = 0.73
    rot = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=x.dtype,
    )
    x_transformed = x @ rot.T + torch.tensor([3.0, -2.0])
    vectors_transformed = vectors @ rot.T

    h_out, v_out = layer(h, vectors, x, edge_index, edge_ptr, edge_cont=edge_cont)
    h_rot, v_rot = layer(
        h,
        vectors_transformed,
        x_transformed,
        edge_index,
        edge_ptr,
        edge_cont=edge_cont,
    )

    torch.testing.assert_close(h_rot, h_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(v_rot, v_out @ rot.T, rtol=1e-5, atol=1e-6)


def test_painn_forward_shape_and_gradients():
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=8)
    h = torch.randn(x.size(0), 5, requires_grad=True)
    vectors = torch.randn(x.size(0), 4, 2, requires_grad=True)
    x = x.requires_grad_(True)
    edge_cont = edge_cont.requires_grad_(True)
    layer = models.SE2PaiNNMessagePassing2D(
        node_dim=5,
        out_dim=7,
        edge_cont_dim=3,
        vector_channels=4,
        hidden_dim=16,
    )

    h_out, v_out = layer(h, vectors, x, edge_index, edge_ptr, edge_cont=edge_cont)

    assert h_out.shape == (x.size(0), 7)
    assert v_out.shape == (x.size(0), 4, 2)
    loss = h_out.square().mean() + v_out.square().mean()
    loss.backward()
    assert torch.isfinite(h.grad).all()
    assert torch.isfinite(vectors.grad).all()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(edge_cont.grad).all()


def test_quad_meshing_encoder_init_se2_pipeline(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=6)
    graph = make_csr_graph(
        vertices=x,
        batch_offsets=torch.tensor([0, x.size(0)]),
        edges=edge_index,
        edge_features=edge_cont[:, 1:].bool(),
        edge_ptr=edge_ptr,
    )
    encoder = QuadMeshingEncoder(
        obs_size=1,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=8,
        frontier_pipeline=["init", "se2"],
        frontier_se2_layers=1,
    )

    state = encoder._encode_frontier(graph)
    out = state.node_features
    context = state.context_features

    assert out.shape == (x.size(0), 8)
    assert context.numel() == 0
    assert torch.isfinite(out).all()


def test_frontier_se2_reuses_triplet_topology(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    calls = 0
    original_compute = models.compute_se2_triplet_topology

    def counted_compute(edge_index, edge_ptr, num_nodes):
        nonlocal calls
        calls += 1
        return original_compute(edge_index, edge_ptr, num_nodes)

    monkeypatch.setattr(models, "compute_se2_triplet_topology", counted_compute)
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=6)
    graph = make_csr_graph(
        vertices=x,
        batch_offsets=torch.tensor([0, x.size(0)]),
        edges=edge_index,
        edge_features=edge_cont[:, 1:].bool(),
        edge_ptr=edge_ptr,
    )
    encoder = QuadMeshingEncoder(
        obs_size=1,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=8,
        frontier_pipeline=["init", "se2"],
        frontier_se2_layers=2,
    )

    encoder._encode_frontier(graph)

    assert calls == 1


def test_quad_meshing_encoder_init_painn_pipeline(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=6)
    graph = make_csr_graph(
        vertices=x,
        batch_offsets=torch.tensor([0, x.size(0)]),
        edges=edge_index,
        edge_features=edge_cont[:, 1:].bool(),
        edge_ptr=edge_ptr,
    )
    encoder = QuadMeshingEncoder(
        obs_size=1,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=8,
        frontier_pipeline=["init", "painn"],
        frontier_painn_layers=1,
        frontier_painn_vector_channels=4,
    )

    state = encoder._encode_frontier(graph)

    assert state.node_features.shape == (x.size(0), 8)
    assert state.node_vectors.shape == (x.size(0), 4, 2)
    assert torch.isfinite(state.node_features).all()
    assert torch.isfinite(state.node_vectors).all()


def test_quad_meshing_3d_init_stage_seeds_normals(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    graph = make_quad_mesh_3d_graph()
    encoder = QuadMeshingEncoder(
        obs_size=1,
        spatial_dim=3,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=8,
        frontier_pipeline=["init"],
        frontier_init_node_normal=True,
        frontier_init_edge_face_incidence=False,
        frontier_painn_vector_channels=4,
        target_pipeline=["init"],
        target_init_distance=False,
        target_init_relative_distance=False,
        target_init_normal=True,
    )

    state = encoder._encode_frontier(graph)

    assert state.node_features.shape == (graph.vertices.size(0), 8)
    assert state.node_vectors.shape == (graph.vertices.size(0), 4, 3)
    torch.testing.assert_close(state.node_vectors[:, 0, :], graph.normals)
    torch.testing.assert_close(state.node_vectors[:, 1:, :], torch.zeros_like(state.node_vectors[:, 1:, :]))


def test_quad_meshing_3d_painn_global_pipeline(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    graph = make_quad_mesh_3d_graph()
    targets = make_quad_mesh_3d_targets()
    encoder = QuadMeshingEncoder(
        obs_size=1,
        spatial_dim=3,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=8,
        frontier_pipeline=["init", "painn"],
        frontier_init_node_normal=True,
        frontier_init_edge_face_incidence=False,
        frontier_painn_layers=1,
        frontier_painn_vector_channels=4,
        frontier_painn_chiral=False,
        target_pipeline=["init", "painn", "source_global_perceiver"],
        target_init_distance=False,
        target_init_relative_distance=False,
        target_init_path_length=True,
        target_init_relative_path_length=True,
        target_init_normal=True,
        target_painn_length_bands=2,
        target_global_perceiver_d_model=8,
        target_global_perceiver_num_latents=4,
        target_global_perceiver_layers=1,
        target_global_perceiver_heads=2,
    )

    frontier_state = encoder._encode_frontier(graph)
    h_target = encoder._encode_targets(targets, frontier_state)

    assert frontier_state.node_features.shape == (graph.vertices.size(0), 8)
    assert frontier_state.node_vectors.shape == (graph.vertices.size(0), 4, 3)
    assert h_target.shape == (2, 8)
    assert torch.isfinite(frontier_state.node_features).all()
    assert torch.isfinite(frontier_state.node_vectors).all()
    assert torch.isfinite(h_target).all()


def test_quad_meshing_3d_rejects_2d_only_features():
    with pytest.raises(ValueError, match="frontier_init_node_ring"):
        QuadMeshingEncoder(
            obs_size=1,
            spatial_dim=3,
            frontier_pipeline=["init"],
            frontier_init_node_ring=True,
            frontier_init_edge_face_incidence=False,
            target_init_distance=False,
            target_init_relative_distance=False,
        )

    with pytest.raises(ValueError, match="target_init_distance"):
        QuadMeshingEncoder(
            obs_size=1,
            spatial_dim=3,
            frontier_pipeline=["init"],
            frontier_init_edge_face_incidence=False,
            target_init_distance=True,
            target_init_relative_distance=False,
        )

    with pytest.raises(ValueError, match="frontier_painn_chiral"):
        QuadMeshingEncoder(
            obs_size=1,
            spatial_dim=3,
            frontier_pipeline=["init", "painn"],
            frontier_init_edge_face_incidence=False,
            frontier_painn_chiral=True,
            target_init_distance=False,
            target_init_relative_distance=False,
        )


def test_quad_meshing_encoder_init_only_pipeline(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=6)
    graph = make_csr_graph(
        vertices=x,
        batch_offsets=torch.tensor([0, x.size(0)]),
        edges=edge_index,
        edge_features=edge_cont[:, 1:].bool(),
        edge_ptr=edge_ptr,
    )
    encoder = QuadMeshingEncoder(
        obs_size=1,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=8,
        frontier_pipeline=["init"],
    )

    state = encoder._encode_frontier(graph)
    out = state.node_features
    context = state.context_features

    assert out.shape == (x.size(0), 8)
    assert context.numel() == 0
    assert torch.isfinite(out).all()


def test_quad_meshing_encoder_init_perceiver_pipeline(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=6)
    graph = make_csr_graph(
        vertices=x,
        batch_offsets=torch.tensor([0, x.size(0)]),
        edges=edge_index,
        edge_features=edge_cont[:, 1:].bool(),
        edge_ptr=edge_ptr,
    )
    encoder = QuadMeshingEncoder(
        obs_size=1,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=8,
        frontier_pipeline=["init", "perceiver"],
        frontier_perceiver_num_latents=4,
        frontier_perceiver_layers=1,
        frontier_perceiver_heads=2,
    )

    state = encoder._encode_frontier(graph)
    out = state.node_features
    context = state.context_features

    assert out.shape == (x.size(0), 8)
    assert context.shape == (1, 6)
    assert torch.isfinite(out).all()
    assert torch.isfinite(context).all()


def test_quad_meshing_encoder_perceiver_context_only(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=6)
    graph = make_csr_graph(
        vertices=x,
        batch_offsets=torch.tensor([0, x.size(0)]),
        edges=edge_index,
        edge_features=edge_cont[:, 1:].bool(),
        edge_ptr=edge_ptr,
    )
    encoder = QuadMeshingEncoder(
        obs_size=1,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=8,
        frontier_pipeline=["init", "perceiver"],
        frontier_perceiver_num_latents=4,
        frontier_perceiver_layers=1,
        frontier_perceiver_heads=2,
        frontier_perceiver_decode_node=False,
        frontier_perceiver_decode_edge=False,
        frontier_perceiver_update_context=True,
    )

    init_state = models.FrontierState(
        graph=graph,
        node_features=x.new_empty(0, 8),
        edge_features=x.new_empty(0, 5),
        context_features=x.new_empty(0, 6),
    )
    init_state = encoder.frontier_pipeline[0](init_state)
    state = encoder._encode_frontier(graph)
    out = state.node_features
    context = state.context_features

    torch.testing.assert_close(out, init_state.node_features)
    assert context.shape == (1, 6)
    assert torch.isfinite(context).all()


def test_quad_meshing_encoder_composes_perceiver_and_se2(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=6)
    graph = make_csr_graph(
        vertices=x,
        batch_offsets=torch.tensor([0, x.size(0)]),
        edges=edge_index,
        edge_features=edge_cont[:, 1:].bool(),
        edge_ptr=edge_ptr,
    )
    encoder = QuadMeshingEncoder(
        obs_size=1,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=8,
        frontier_pipeline=["init", "perceiver", "se2"],
        frontier_perceiver_num_latents=4,
        frontier_perceiver_layers=1,
        frontier_perceiver_heads=2,
        frontier_se2_layers=1,
    )

    state = encoder._encode_frontier(graph)
    out = state.node_features
    context = state.context_features

    assert out.shape == (x.size(0), 8)
    assert context.shape == (1, 6)
    assert torch.isfinite(out).all()
    assert torch.isfinite(context).all()


def test_frontier_perceiver_can_decode_edge_features(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=6)
    graph = make_csr_graph(
        vertices=x,
        batch_offsets=torch.tensor([0, x.size(0)]),
        edges=edge_index,
        edge_features=edge_cont[:, 1:].bool(),
        edge_ptr=edge_ptr,
    )
    encoder = QuadMeshingEncoder(
        obs_size=1,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=8,
        frontier_pipeline=["init", "perceiver"],
        frontier_perceiver_num_latents=4,
        frontier_perceiver_layers=1,
        frontier_perceiver_heads=2,
        frontier_perceiver_decode_edge=True,
    )

    state = models.FrontierState(
        graph=graph,
        node_features=x.new_empty(0, 8),
        edge_features=x.new_empty(0, 5),
        context_features=x.new_empty(0, 6),
    )
    for stage in encoder.frontier_pipeline:
        state = stage(state)

    assert state.edge_features.shape == (edge_index.size(1), 5)
    assert torch.isfinite(state.edge_features).all()


def test_quad_meshing_target_se2_pipeline_without_ring_frame(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=6)
    graph = make_csr_graph(
        vertices=x,
        batch_offsets=torch.tensor([0, x.size(0)]),
        edges=edge_index,
        edge_features=edge_cont[:, 1:].bool(),
        edge_ptr=edge_ptr,
    )
    targets = CandidateTargets(
        source_idx=torch.tensor([0]),
        target_idx=torch.tensor([-1, 2]),
        target_batch_offsets=torch.tensor([0, 2]),
        target_batches=torch.tensor([0, 0]),
        target_positions=torch.tensor([[0.8, 0.1], [0.2, 0.7]], dtype=x.dtype),
        target_frontier_parity=torch.tensor([True, False]),
    )
    encoder = QuadMeshingEncoder(
        obs_size=1,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=7,
        frontier_pipeline=["init", "se2"],
        frontier_se2_layers=1,
        target_pipeline=["init", "se2"],
        target_init_ring_frame_pos=False,
        target_init_distance=False,
        target_init_relative_distance=False,
        target_init_boundary_flag=True,
    )

    frontier_state = encoder._encode_frontier(graph)
    h_target = encoder._encode_targets(targets, frontier_state)

    assert h_target.shape == (2, 7)
    assert torch.isfinite(h_target).all()


def test_quad_meshing_target_painn_pipeline(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=6)
    graph = make_csr_graph(
        vertices=x,
        batch_offsets=torch.tensor([0, x.size(0)]),
        edges=edge_index,
        edge_features=edge_cont[:, 1:].bool(),
        edge_ptr=edge_ptr,
    )
    targets = CandidateTargets(
        source_idx=torch.tensor([0]),
        target_idx=torch.tensor([-1, 2]),
        target_batch_offsets=torch.tensor([0, 2]),
        target_batches=torch.tensor([0, 0]),
        target_positions=torch.tensor([[0.8, 0.1], [0.2, 0.7]], dtype=x.dtype),
        target_frontier_parity=torch.tensor([True, False]),
    )
    encoder = QuadMeshingEncoder(
        obs_size=1,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=7,
        frontier_pipeline=["init", "painn"],
        frontier_painn_layers=1,
        frontier_painn_vector_channels=4,
        target_pipeline=["init", "painn"],
        target_init_ring_frame_pos=False,
        target_init_distance=False,
        target_init_relative_distance=False,
        target_init_boundary_flag=True,
    )

    frontier_state = encoder._encode_frontier(graph)
    h_target = encoder._encode_targets(targets, frontier_state)

    assert h_target.shape == (2, 7)
    assert torch.isfinite(h_target).all()


def test_quad_meshing_target_source_global_perceiver_pipeline(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=6)
    graph = make_csr_graph(
        vertices=x,
        batch_offsets=torch.tensor([0, x.size(0)]),
        edges=edge_index,
        edge_features=edge_cont[:, 1:].bool(),
        edge_ptr=edge_ptr,
    )
    targets = CandidateTargets(
        source_idx=torch.tensor([0]),
        target_idx=torch.tensor([-1, 2]),
        target_batch_offsets=torch.tensor([0, 2]),
        target_batches=torch.tensor([0, 0]),
        target_positions=torch.tensor([[0.8, 0.1], [0.2, 0.7]], dtype=x.dtype),
        target_frontier_parity=torch.tensor([True, False]),
    )
    encoder = QuadMeshingEncoder(
        obs_size=1,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=8,
        frontier_pipeline=["init", "painn"],
        frontier_painn_layers=1,
        frontier_painn_vector_channels=4,
        target_pipeline=["init", "painn", "source_global_perceiver"],
        target_init_ring_frame_pos=False,
        target_init_distance=False,
        target_init_relative_distance=False,
        target_init_boundary_flag=True,
        target_global_perceiver_d_model=8,
        target_global_perceiver_num_latents=4,
        target_global_perceiver_layers=1,
        target_global_perceiver_heads=2,
    )

    frontier_state = encoder._encode_frontier(graph)
    h_target = encoder._encode_targets(targets, frontier_state)

    assert h_target.shape == (2, 8)
    assert torch.isfinite(h_target).all()

    angle = 0.37
    rot = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=x.dtype,
    )
    shift = torch.tensor([2.0, -1.0], dtype=x.dtype)
    graph_rot = make_csr_graph(
        vertices=x @ rot.T + shift,
        batch_offsets=graph.batch_offsets,
        edges=edge_index,
        edge_features=graph.edge_features,
        edge_ptr=edge_ptr,
    )
    targets_rot = CandidateTargets(
        source_idx=targets.source_idx,
        target_idx=targets.target_idx,
        target_batch_offsets=targets.target_batch_offsets,
        target_batches=targets.target_batches,
        target_positions=targets.target_positions @ rot.T + shift,
        target_frontier_parity=targets.target_frontier_parity,
    )

    h_target_rot = encoder._encode_targets(targets_rot, encoder._encode_frontier(graph_rot))

    torch.testing.assert_close(h_target_rot, h_target, rtol=1e-5, atol=5e-6)


def test_target_source_global_perceiver_zero_scale_is_identity(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=6)
    graph = make_csr_graph(
        vertices=x,
        batch_offsets=torch.tensor([0, x.size(0)]),
        edges=edge_index,
        edge_features=edge_cont[:, 1:].bool(),
        edge_ptr=edge_ptr,
    )
    targets = CandidateTargets(
        source_idx=torch.tensor([0]),
        target_idx=torch.tensor([-1, 2]),
        target_batch_offsets=torch.tensor([0, 2]),
        target_batches=torch.tensor([0, 0]),
        target_positions=torch.tensor([[0.8, 0.1], [0.2, 0.7]], dtype=x.dtype),
        target_frontier_parity=torch.tensor([True, False]),
    )
    encoder = QuadMeshingEncoder(
        obs_size=1,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=8,
        frontier_pipeline=["init", "painn"],
        frontier_painn_layers=1,
        frontier_painn_vector_channels=4,
        target_pipeline=["init", "painn", "source_global_perceiver"],
        target_init_ring_frame_pos=False,
        target_init_distance=False,
        target_init_relative_distance=False,
        target_init_boundary_flag=True,
        target_global_perceiver_d_model=8,
        target_global_perceiver_num_latents=4,
        target_global_perceiver_layers=1,
        target_global_perceiver_heads=2,
        target_global_perceiver_residual_scale_init=0.0,
    )

    frontier_state = encoder._encode_frontier(graph)
    state = models.TargetState(
        targets=targets,
        target_features=graph.vertices.new_empty(0, 8),
    )
    state = encoder.target_pipeline[0](state, frontier_state)
    state = encoder.target_pipeline[1](state, frontier_state)
    before_global = state.target_features.clone()
    state = encoder.target_pipeline[2](state, frontier_state)

    assert encoder.target_pipeline[2].global_residual_scale.requires_grad
    torch.testing.assert_close(state.target_features, before_global)


def test_quad_meshing_target_source_condition_stage(monkeypatch):
    monkeypatch.setattr(models, "nvtx_range", lambda _name: nullcontext())
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=6)
    graph = make_csr_graph(
        vertices=x,
        batch_offsets=torch.tensor([0, x.size(0)]),
        edges=edge_index,
        edge_features=edge_cont[:, 1:].bool(),
        edge_ptr=edge_ptr,
    )
    targets = CandidateTargets(
        source_idx=torch.tensor([0]),
        target_idx=torch.tensor([-1]),
        target_batch_offsets=torch.tensor([0, 1]),
        target_batches=torch.tensor([0]),
        target_positions=torch.tensor([[0.8, 0.1]], dtype=x.dtype),
        target_frontier_parity=torch.tensor([True]),
    )
    encoder = QuadMeshingEncoder(
        obs_size=1,
        frontier_node_hidden_size=8,
        frontier_edge_hidden_size=5,
        frontier_context_hidden_size=6,
        target_hidden_size=7,
        frontier_pipeline=["init"],
        target_pipeline=["init", "source_condition"],
        target_init_ring_frame_pos=False,
        target_init_distance=False,
        target_init_relative_distance=False,
        target_init_boundary_flag=True,
    )

    frontier_state = encoder._encode_frontier(graph)
    h_target = encoder._encode_targets(targets, frontier_state)

    assert h_target.shape == (1, 7)
    assert torch.isfinite(h_target).all()


def test_target_init_stage_accepts_frontier_parity_feature():
    x, edge_index, edge_ptr, edge_cont = make_directed_ring(num_nodes=4)
    graph = make_csr_graph(
        vertices=x,
        batch_offsets=torch.tensor([0, x.size(0)]),
        edges=edge_index,
        edge_features=edge_cont[:, 1:].bool(),
        edge_ptr=edge_ptr,
    )
    frontier_state = models.FrontierState(
        graph=graph,
        node_features=torch.empty(4, 3),
        edge_features=torch.empty(edge_index.size(1), 2),
        context_features=torch.empty(1, 3),
        node_vectors=None,
    )
    targets = CandidateTargets(
        source_idx=torch.tensor([0]),
        target_idx=torch.tensor([-1, 2]),
        target_batch_offsets=torch.tensor([0, 2]),
        target_batches=torch.tensor([0, 0]),
        target_positions=torch.tensor([[0.8, 0.1], [0.2, 0.7]], dtype=x.dtype),
        target_frontier_parity=torch.tensor([True, False]),
    )
    state = models.TargetState(
        targets=targets,
        target_features=torch.empty(0, 5),
    )
    stage = models.TargetInitStage(
        target_hidden_size=5,
        pos_bands=0,
        include_ring_frame_pos=False,
        include_distance=False,
        include_target_length=False,
        include_target_log_length=False,
        include_relative_distance=False,
        include_boundary_flag=False,
        include_frontier_parity=True,
    )

    out = stage(state, frontier_state)

    assert out.target_features.shape == (2, 5)
    assert torch.isfinite(out.target_features).all()


def test_source_edge_perceiver_canonical_direction_tie_breaker():
    delta = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, -3.0], [0.0, 3.0]])
    length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)

    direction = models.FrontierInitStage._canonical_direction(delta, length, 1e-8)

    expected = torch.tensor([[2.0, -0.0], [2.0, 0.0], [-0.0, 3.0], [0.0, 3.0]])
    expected = expected / torch.linalg.vector_norm(expected, dim=-1, keepdim=True)
    torch.testing.assert_close(direction, expected)


def test_quad_meshing_substep0_context_falls_back_to_pooled_sources():
    frontier_node_hidden_size = 4
    frontier_context_hidden_size = 5
    target_hidden_size = 3
    h_source0 = torch.arange(20, dtype=torch.float32).view(5, frontier_node_hidden_size)
    offsets = torch.tensor([0, 2, 5])
    encoded = QuadMeshEncoding(
        graph0=None,
        graph1=None,
        substep=torch.tensor([0, 0], dtype=torch.uint8),
        h_source0=h_source0,
        h_source1=torch.empty(0, frontier_node_hidden_size),
        h_source0_context=torch.empty(0, frontier_context_hidden_size),
        h_source0_batch_offset=offsets,
        h_source1_batch_offset=torch.zeros(1, dtype=torch.long),
        source_idx=torch.empty(0, dtype=torch.long),
        h_target=torch.empty(0, target_hidden_size),
        h_target_batch_offset=torch.zeros(1, dtype=torch.long),
    )
    network = QuadMeshingNetwork(
        frontier_node_hidden_size,
        target_hidden_size,
        frontier_context_hidden_size=frontier_context_hidden_size,
    )
    network.source_pool_context_mlp = torch.nn.Linear(frontier_node_hidden_size, frontier_context_hidden_size, bias=False)
    decoder = QuadMeshingDecoder(
        [1],
        frontier_node_hidden_size,
        target_hidden_size,
        frontier_context_hidden_size=frontier_context_hidden_size,
    )
    with torch.no_grad():
        network.source_pool_context_mlp.weight.zero_()
        network.source_pool_context_mlp.weight[:frontier_node_hidden_size].copy_(torch.eye(frontier_node_hidden_size))
        decoder.source_value_head.weight.fill_(1.0)
        decoder.source_value_head.bias.zero_()

    encoded = network(encoded)
    expected_pooled = torch.stack([
        encoded.h_source0[0:2].mean(dim=0),
        encoded.h_source0[2:5].mean(dim=0),
    ])
    expected_context = torch.cat([expected_pooled, torch.zeros(2, 1)], dim=-1)
    torch.testing.assert_close(encoded.h_source0_context, expected_context)

    _, values = decoder(encoded)
    torch.testing.assert_close(values, expected_context.sum(dim=-1, keepdim=True))


def test_quad_meshing_substep0_value_uses_provided_context():
    frontier_node_hidden_size = 4
    frontier_context_hidden_size = 5
    target_hidden_size = 3
    provided_context = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    encoded = QuadMeshEncoding(
        graph0=None,
        graph1=None,
        substep=torch.tensor([0], dtype=torch.uint8),
        h_source0=torch.zeros(2, frontier_node_hidden_size),
        h_source1=torch.empty(0, frontier_node_hidden_size),
        h_source0_context=provided_context,
        h_source0_batch_offset=torch.tensor([0, 2]),
        h_source1_batch_offset=torch.zeros(1, dtype=torch.long),
        source_idx=torch.empty(0, dtype=torch.long),
        h_target=torch.empty(0, target_hidden_size),
        h_target_batch_offset=torch.zeros(1, dtype=torch.long),
    )
    network = QuadMeshingNetwork(
        frontier_node_hidden_size,
        target_hidden_size,
        frontier_context_hidden_size=frontier_context_hidden_size,
    )
    network.source_context_mlp = torch.nn.Identity()
    decoder = QuadMeshingDecoder(
        [1],
        frontier_node_hidden_size,
        target_hidden_size,
        frontier_context_hidden_size=frontier_context_hidden_size,
    )
    with torch.no_grad():
        decoder.source_value_head.weight.fill_(1.0)
        decoder.source_value_head.bias.zero_()

    encoded = network(encoded)
    torch.testing.assert_close(encoded.h_source0_context, provided_context)

    _, values = decoder(encoded)
    torch.testing.assert_close(values, provided_context.sum(dim=-1, keepdim=True))


def test_quad_meshing_target_network_scores_target_features_without_source_concat():
    frontier_node_hidden_size = 4
    frontier_context_hidden_size = 5
    target_hidden_size = 3
    encoded = QuadMeshEncoding(
        graph0=None,
        graph1=None,
        substep=torch.tensor([1], dtype=torch.uint8),
        h_source0=torch.empty(0, frontier_node_hidden_size),
        h_source1=torch.ones(2, frontier_node_hidden_size),
        h_source0_context=torch.empty(0, frontier_context_hidden_size),
        h_source0_batch_offset=torch.zeros(1, dtype=torch.long),
        h_source1_batch_offset=torch.tensor([0, 2]),
        source_idx=torch.tensor([0]),
        h_target=torch.ones(2, target_hidden_size),
        h_target_batch_offset=torch.tensor([0, 2]),
    )
    network = QuadMeshingNetwork(
        frontier_node_hidden_size,
        target_hidden_size,
        frontier_context_hidden_size=frontier_context_hidden_size,
    )
    decoder = QuadMeshingDecoder(
        [1],
        frontier_node_hidden_size,
        target_hidden_size,
        frontier_context_hidden_size=frontier_context_hidden_size,
    )

    encoded = network(encoded)
    logits, values = decoder(encoded)

    assert logits.shape == (1, 2)
    assert values.shape == (1, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()


def test_quad_meshing_decoder_masks_to_suggested_vertex_neighborhood():
    vertices = torch.zeros(4, 2)
    edges = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
    ])
    graph = CSRGraph(
        vertices=vertices,
        batch_offsets=torch.tensor([0, 4]),
        edges=edges,
        edge_features=torch.zeros(4, 2, dtype=torch.bool),
        edge_ptr=torch.tensor([0, 1, 3, 4, 4]),
        target_edge_length=torch.ones(1),
        suggested_vertex_idx=torch.tensor([1]),
    )
    encoded = QuadMeshEncoding(
        graph0=graph,
        graph1=None,
        substep=torch.tensor([0], dtype=torch.uint8),
        h_source0=torch.arange(4, dtype=torch.float32).view(4, 1),
        h_source1=torch.empty(0, 1),
        h_source0_context=torch.zeros(1, 1),
        h_source0_batch_offset=torch.tensor([0, 4]),
        h_source1_batch_offset=torch.zeros(1, dtype=torch.long),
        source_idx=torch.empty(0, dtype=torch.long),
        h_target=torch.empty(0, 1),
        h_target_batch_offset=torch.zeros(1, dtype=torch.long),
    )
    decoder = QuadMeshingDecoder(
        [1],
        frontier_node_hidden_size=1,
        target_hidden_size=1,
        frontier_context_hidden_size=1,
        use_suggested_vertex=True,
    )
    with torch.no_grad():
        decoder.source_head.weight.fill_(1.0)
        decoder.source_head.bias.zero_()

    logits, _ = decoder(encoded)

    assert torch.isfinite(logits[0, :3]).all()
    assert torch.isneginf(logits[0, 3])
