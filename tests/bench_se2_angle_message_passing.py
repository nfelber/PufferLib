import argparse
import math
import time

import torch

from pufferlib.models import CSRGraph, SE2AngleMessagePassing, compute_ring_source_features


def make_directed_ring(num_nodes, device):
    theta = torch.linspace(0, 2 * math.pi, num_nodes + 1, device=device)[:-1]
    vertices = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1).contiguous()

    src = torch.repeat_interleave(torch.arange(num_nodes, device=device), 2)
    dst = torch.empty(2 * num_nodes, device=device, dtype=torch.long)
    edge_features = torch.empty(2 * num_nodes, 2, device=device, dtype=torch.bool)
    edge_cont = torch.empty(2 * num_nodes, 3, device=device)

    for i in range(num_nodes):
        for rank, j in enumerate(((i + 1) % num_nodes, (i - 1) % num_nodes)):
            e = 2 * i + rank
            dst[e] = j
            length = torch.linalg.vector_norm(vertices[j] - vertices[i])
            is_ccw = rank == 0
            edge_features[e, 0] = is_ccw
            edge_features[e, 1] = not is_ccw
            edge_cont[e, 0] = length
            edge_cont[e, 1] = float(is_ccw)
            edge_cont[e, 2] = float(not is_ccw)

    edge_index = torch.stack([src, dst]).contiguous()
    edge_ptr = torch.arange(0, 2 * num_nodes + 1, 2, device=device, dtype=torch.long)
    graph = CSRGraph(
        vertices=vertices,
        batch_offsets=torch.tensor([0, num_nodes], device=device, dtype=torch.long),
        edges=edge_index,
        edge_features=edge_features,
        edge_ptr=edge_ptr,
        target_edge_length=torch.ones(1, device=device, dtype=vertices.dtype),
    )
    return graph, edge_cont


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_forward(fn, device, iters, warmup=10):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            fn()
    synchronize(device)
    return (time.perf_counter() - start) / iters


def time_forward_backward(fn, params, device, iters, warmup=10):
    for _ in range(warmup):
        for param in params:
            param.grad = None
        fn().square().mean().backward()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        for param in params:
            param.grad = None
        fn().square().mean().backward()
    synchronize(device)
    return (time.perf_counter() - start) / iters


def main():
    parser = argparse.ArgumentParser(description="Benchmark SE2AngleMessagePassing on synthetic boundary rings.")
    parser.add_argument("--nodes", type=int, default=4096)
    parser.add_argument("--node-dim", type=int, default=128)
    parser.add_argument("--msg-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    graph, edge_cont = make_directed_ring(args.nodes, device)
    h = torch.randn(args.nodes, args.node_dim, device=device, requires_grad=True)
    edge_cont = edge_cont.requires_grad_(True)
    layer = SE2AngleMessagePassing(
        node_dim=args.node_dim,
        out_dim=args.out_dim,
        msg_dim=args.msg_dim,
        edge_cont_dim=3,
        hidden_dim=args.hidden_dim,
    ).to(device)

    se2_fn = lambda: layer(h, graph.vertices, graph.edges, graph.edge_ptr, edge_cont=edge_cont)
    se2_forward = time_forward(se2_fn, device, args.iters)
    se2_backward = time_forward_backward(se2_fn, [h, edge_cont, *layer.parameters()], device, args.iters)

    print(f"SE2 forward: {se2_forward * 1e3:.3f} ms")
    print(f"SE2 forward+backward: {se2_backward * 1e3:.3f} ms")

    if device.type == "cuda":
        ring_fn = lambda: compute_ring_source_features(graph, args.neighbors)
        ring_forward = time_forward(ring_fn, device, args.iters)
        print(f"ring source features forward: {ring_forward * 1e3:.3f} ms")
    else:
        print("ring source features forward: skipped (Triton kernel requires CUDA)")


if __name__ == "__main__":
    main()
