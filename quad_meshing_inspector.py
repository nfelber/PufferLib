"""Interactive inspector for trained quad_meshing policies.

Examples:
    python quad_meshing_inspector.py --checkpoint checkpoints/quad_meshing/run/0000001000000000.bin
    python quad_meshing_inspector.py --checkpoint latest --export-view view.png

Controls:
    Space/Enter  Step all agents greedily with the current policy
    Shift+Space  Step greedily 20 times
    R            Reset the vector environment
    Left/Right   Select previous/next agent
    M            Export the current view with matplotlib
    P            Toggle probability/logit coloring
    G            Toggle global-residual ablation overlay when available
    C            Toggle comparison-policy delta overlay when available
    Click        Force clicked source/candidate for selected agent
    Q/Escape     Quit
"""

from __future__ import annotations

import argparse
import ast
import configparser
import ctypes
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pufferlib.models as models
from pufferlib import _C


@dataclass
class AgentDiagnostics:
    agent_idx: int
    substep: int
    vertices: torch.Tensor
    edges: torch.Tensor
    edge_ptr: torch.Tensor
    target_positions: torch.Tensor | None
    target_idx: torch.Tensor | None
    source_idx: int | None
    scores: torch.Tensor
    probabilities: torch.Tensor
    values: torch.Tensor
    global_score_delta: torch.Tensor | None
    global_rank_delta: torch.Tensor | None
    comparison_score_delta: torch.Tensor | None


def _literal(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def load_ini_config(config_path: str | Path) -> dict[str, Any]:
    """Load PufferLib-style default + env ini config into a nested dict."""
    root = Path(__file__).resolve().parent
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path

    parser = configparser.ConfigParser()
    parser.read([root / "config" / "default.ini", config_path])

    args: dict[str, Any] = {}
    for section in parser.sections():
        args[section] = {key: _literal(value) for key, value in parser[section].items()}

    args["env_name"] = args.get("base", {}).get("env_name", "quad_meshing")
    return args


def resolve_checkpoint(path: str | None, args: dict[str, Any]) -> str | None:
    if path is None:
        return None
    if path != "latest":
        return str(Path(path).expanduser())

    checkpoint_dir = args.get("base", {}).get("checkpoint_dir", "checkpoints")
    env_name = args.get("env_name", args.get("base", {}).get("env_name", "quad_meshing"))
    pattern = os.path.join(str(checkpoint_dir), str(env_name), "**", "*.bin")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No .bin checkpoints found for pattern {pattern!r}.")
    return max(candidates, key=os.path.getctime)


def get_obs(vec, device: torch.device | None = None) -> torch.Tensor:
    n = vec.total_agents * vec.obs_size
    buf = (ctypes.c_uint8 * n).from_address(vec.obs_ptr)
    obs = torch.frombuffer(buf, dtype=torch.uint8).reshape(vec.total_agents, vec.obs_size)
    if device is not None:
        obs = obs.to(device)
    return obs


def finite_softmax(scores: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(scores)
    mask = torch.isfinite(scores)
    if mask.any():
        out[mask] = torch.softmax(scores[mask], dim=0)
    return out


def action_ranks(scores: torch.Tensor) -> torch.Tensor:
    ranks = torch.empty(scores.numel(), dtype=torch.long, device=scores.device)
    order = torch.argsort(torch.nan_to_num(scores, neginf=-1e30), descending=True)
    ranks[order] = torch.arange(scores.numel(), device=scores.device)
    return ranks


class QuadMeshingInspector:
    """Backend for environment, policy, checkpoint and per-agent diagnostics."""

    def __init__(
        self,
        config_path: str = "config/quad_meshing.ini",
        checkpoint: str | None = None,
        compare_config_path: str | None = None,
        compare_checkpoint: str | None = None,
        device: str | None = None,
        total_agents: int | None = None,
        seed: int | None = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("quad_meshing_inspector currently requires CUDA for observation deserialization.")

        self.args = load_ini_config(config_path)
        if total_agents is not None:
            self.args["vec"]["total_agents"] = int(total_agents)
        if seed is not None:
            self.args["env"]["seed"] = int(seed)

        self.device = torch.device(device or "cuda")
        self.vec = _C.create_vec(self.args)
        self.vec.reset()
        self.obs_cuda = torch.empty(
            (self.vec.total_agents, self.vec.obs_size),
            dtype=torch.uint8,
            device=self.device,
        )
        self.action_cuda = torch.empty(
            (self.vec.total_agents, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.policy = self._load_policy(resolve_checkpoint(checkpoint, self.args))
        self.policy.eval()
        self.policy_state = self.policy.initial_state(self.vec.total_agents, self.device)
        self.compare_args = None
        self.compare_policy = None
        self.compare_policy_state = None
        if compare_config_path is not None or compare_checkpoint is not None:
            if compare_config_path is None or compare_checkpoint is None:
                raise ValueError("Comparison mode requires both --compare-config and --compare-checkpoint.")
            self.compare_args = load_ini_config(compare_config_path)
            compare_path = resolve_checkpoint(compare_checkpoint, self.compare_args)
            self.compare_policy = self._load_policy(compare_path, self.compare_args, label="comparison")
            self.compare_policy.eval()
            self.compare_policy_state = self.compare_policy.initial_state(self.vec.total_agents, self.device)
        self.last_checkpoint = checkpoint
        self.step_count = 0

    def close(self):
        close = getattr(self.vec, "close", None)
        if close is not None:
            close()

    def _load_policy(self, checkpoint: str | None, args: dict[str, Any] | None = None, label: str = "primary"):
        args = self.args if args is None else args
        policy_kwargs = args["policy"]
        network_cls = getattr(models, args["torch"]["network"])
        encoder_cls = getattr(models, args["torch"]["encoder"])
        decoder_cls = getattr(models, args["torch"]["decoder"])

        network = network_cls(**policy_kwargs)
        encoder = encoder_cls(self.vec.obs_size, **policy_kwargs)
        decoder = decoder_cls(self.vec.act_sizes, **policy_kwargs)
        policy = models.Policy(encoder, decoder, network).to(self.device)

        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=self.device)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            policy.load_state_dict(state_dict)
            print(f"Loaded {label} checkpoint: {checkpoint}")
        return policy

    def reset(self):
        self.vec.reset()
        self.policy_state = self.policy.initial_state(self.vec.total_agents, self.device)
        if self.compare_policy is not None:
            self.compare_policy_state = self.compare_policy.initial_state(self.vec.total_agents, self.device)
        self.step_count = 0

    def _current_obs(self) -> torch.Tensor:
        obs_cpu = get_obs(self.vec)
        self.obs_cuda.copy_(obs_cpu, non_blocking=False)
        return self.obs_cuda

    def _policy_outputs(self, ablate_global: bool = False):
        obs = self._current_obs()
        with torch.no_grad():
            if ablate_global:
                return obs, self._with_global_scale(0.0, lambda: self.policy.forward_eval(obs, self.policy_state)[:2])
            logits, values, _ = self.policy.forward_eval(obs, self.policy_state)
            return obs, (logits, values)

    def _comparison_outputs(self, obs: torch.Tensor):
        if self.compare_policy is None:
            return None
        with torch.no_grad():
            logits, values, _ = self.compare_policy.forward_eval(obs, self.compare_policy_state)
            return logits, values

    def _with_global_scale(self, value: float, fn):
        params = []
        for module in self.policy.modules():
            scale = getattr(module, "global_residual_scale", None)
            if scale is not None:
                params.append((scale, scale.detach().clone()))
                scale.fill_(float(value))
        try:
            return fn()
        finally:
            for scale, old in params:
                scale.copy_(old)

    def greedy_step(self):
        obs, (logits, _) = self._policy_outputs()
        choices = torch.argmax(torch.nan_to_num(logits, neginf=-1e30), dim=1).to(torch.float32)
        self._step_choices(choices)

    def manual_step(self, agent_idx: int, action_idx: int):
        _obs, (logits, _) = self._policy_outputs()
        choices = torch.argmax(torch.nan_to_num(logits, neginf=-1e30), dim=1).to(torch.float32)
        choices[agent_idx] = float(action_idx)
        self._step_choices(choices)

    def _step_choices(self, choices: torch.Tensor):
        self.action_cuda[:, 0].copy_(choices)
        self.vec.gpu_step(self.action_cuda.data_ptr())
        torch.cuda.synchronize(self.device)
        self.step_count += 1

    def _decode_agent(self, obs: torch.Tensor, agent_idx: int, substep: int):
        valid = torch.tensor([agent_idx], dtype=torch.long, device=self.device)
        if substep == 0:
            graph = models.deserialize_observation(
                obs,
                valid,
                D=self.args["env"]["max_degree"],
                F_CAP=self.args["policy"]["max_frontier"],
                C_CAP=self.args["policy"]["max_candidates"],
                deserialize_candidates=False,
                exact_output=True,
                copy_obs=True,
                use_cuda_graph=True,
            )
            return graph, None

        graph, targets = models.deserialize_observation(
            obs,
            valid,
            D=self.args["env"]["max_degree"],
            F_CAP=self.args["policy"]["max_frontier"],
            C_CAP=self.args["policy"]["max_candidates"],
            deserialize_candidates=True,
            exact_output=True,
            copy_obs=True,
            use_cuda_graph=True,
        )
        return graph, targets

    def diagnostics(self, agent_idx: int, compute_global_delta: bool = True) -> AgentDiagnostics:
        obs, (logits, values) = self._policy_outputs()
        substep = int(obs[agent_idx, 0].item())
        graph, targets = self._decode_agent(obs, agent_idx, substep)

        if substep == 0:
            count = int(graph.vertices.size(0))
            target_positions = None
            target_idx = None
            source_idx = None
        else:
            count = int(targets.target_positions.size(0))
            target_positions = targets.target_positions.detach().cpu()
            target_idx = targets.target_idx.detach().cpu()
            source_idx = int(targets.source_idx[0].item())

        scores = logits[agent_idx, :count].detach().cpu()
        probs = finite_softmax(scores)
        global_delta = None
        global_rank_delta = None
        if compute_global_delta and substep == 1 and self.has_global_residual():
            _, (ablated_logits, _) = self._policy_outputs(ablate_global=True)
            ablated_scores = ablated_logits[agent_idx, :count]
            # global_delta = (logits[agent_idx, :count] - ablated_scores).detach().cpu()
            global_delta = ablated_scores.detach().cpu() # TODO: this is not global delta, just a quick test
            global_rank_delta = (action_ranks(logits[agent_idx, :count]) - action_ranks(ablated_scores)).detach().cpu()

        comparison_delta = None
        comparison = self._comparison_outputs(obs)
        if comparison is not None:
            comparison_logits, _ = comparison
            n = min(count, comparison_logits.size(1))
            comparison_delta = torch.empty(count, dtype=torch.float32)
            comparison_delta.fill_(float("nan"))
            comparison_delta[:n] = (logits[agent_idx, :n] - comparison_logits[agent_idx, :n]).detach().cpu()

        return AgentDiagnostics(
            agent_idx=agent_idx,
            substep=substep,
            vertices=graph.vertices.detach().cpu(),
            edges=graph.edges.detach().cpu(),
            edge_ptr=graph.edge_ptr.detach().cpu(),
            target_positions=target_positions,
            target_idx=target_idx,
            source_idx=source_idx,
            scores=scores,
            probabilities=probs.detach().cpu(),
            values=values[agent_idx].detach().cpu(),
            global_score_delta=global_delta,
            global_rank_delta=global_rank_delta,
            comparison_score_delta=comparison_delta,
        )

    def has_global_residual(self) -> bool:
        return any(getattr(module, "global_residual_scale", None) is not None for module in self.policy.modules())


class ViewTransform:
    def __init__(self, width: int, height: int, center: torch.Tensor, scale: float):
        self.width = width
        self.height = height
        self.center = center
        self.scale = float(scale)

    @staticmethod
    def fit(width: int, height: int, points: torch.Tensor) -> tuple[torch.Tensor, float]:
        if points.numel() == 0:
            return torch.zeros(2), 1.0
        lo = points.min(dim=0).values
        hi = points.max(dim=0).values
        center = 0.5 * (lo + hi)
        span = (hi - lo).max().item()
        scale = 0.85 * min(width, height) / max(span, 1e-6)
        return center, scale

    def screen(self, p: torch.Tensor) -> tuple[int, int]:
        q = (p - self.center) * self.scale
        return int(q[0].item() + self.width / 2), int(-q[1].item() + self.height / 2)

    def world_delta(self, dx: float, dy: float) -> torch.Tensor:
        return torch.tensor([-dx / self.scale, dy / self.scale], dtype=self.center.dtype)


def score_colors(values: torch.Tensor, use_probs: bool) -> list[tuple[int, int, int]]:
    if values.numel() == 0:
        return []
    finite = torch.isfinite(values)
    if not finite.any():
        return [(120, 120, 120)] * values.numel()
    vals = values.clone()
    if use_probs:
        lo, hi = 0.0, max(float(vals[finite].max().item()), 1e-8)
    else:
        lo = float(vals[finite].min().item())
        hi = float(vals[finite].max().item())
    denom = max(hi - lo, 1e-8)
    colors = []
    for v in vals:
        if not torch.isfinite(v):
            colors.append((70, 70, 70))
            continue
        t = float(torch.clamp((v - lo) / denom, 0.0, 1.0).item())
        colors.append((int(60 + 195 * t), int(80 + 120 * (1.0 - abs(t - 0.5) * 2)), int(255 * (1.0 - t))))
    return colors


class PygameInspectorApp:
    def __init__(self, inspector: QuadMeshingInspector, width: int = 1200, height: int = 900):
        import pygame

        self.pygame = pygame
        self.inspector = inspector
        self.width = width
        self.height = height
        self.agent_idx = 0
        self.use_probs = False
        self.show_global_delta = False
        self.show_comparison_delta = False
        self.view_center: torch.Tensor | None = None
        self.view_scale: float | None = None
        self.dragging = False
        self.last_mouse: tuple[int, int] | None = None
        self.mouse_down_pos: tuple[int, int] | None = None
        self.mouse_moved = False
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Quad Meshing Policy Inspector")
        self.font = pygame.font.SysFont(None, 20)

    def run(self):
        clock = self.pygame.time.Clock()
        running = True
        while running:
            diag = self.inspector.diagnostics(self.agent_idx, self.show_global_delta)
            self.draw(diag)
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    running = False
                elif event.type == self.pygame.KEYDOWN:
                    running = self.handle_key(event.key, diag)
                elif event.type == self.pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_down(event)
                elif event.type == self.pygame.MOUSEBUTTONUP:
                    self.handle_mouse_up(event, diag)
                elif event.type == self.pygame.MOUSEMOTION:
                    self.handle_mouse_motion(event)
                elif event.type == self.pygame.MOUSEWHEEL:
                    self.zoom_at_mouse(event.y)
            clock.tick(30)
        self.pygame.quit()

    def handle_key(self, key: int, diag: AgentDiagnostics) -> bool:
        pg = self.pygame
        if key in (pg.K_q, pg.K_ESCAPE):
            return False
        if key == pg.K_r:
            self.inspector.reset()
        elif key in (pg.K_SPACE, pg.K_RETURN):
            mods = pg.key.get_mods()
            n_steps = 20 if mods & pg.KMOD_SHIFT else 1
            for _ in range(n_steps):
                self.inspector.greedy_step()
        elif key == pg.K_RIGHT:
            self.agent_idx = (self.agent_idx + 1) % self.inspector.vec.total_agents
        elif key == pg.K_LEFT:
            self.agent_idx = (self.agent_idx - 1) % self.inspector.vec.total_agents
        elif key == pg.K_p:
            self.use_probs = not self.use_probs
        elif key == pg.K_g:
            self.show_global_delta = not self.show_global_delta
            if self.show_global_delta:
                self.show_comparison_delta = False
        elif key == pg.K_c and self.inspector.compare_policy is not None:
            self.show_comparison_delta = not self.show_comparison_delta
            if self.show_comparison_delta:
                self.show_global_delta = False
        elif key == pg.K_m:
            export_matplotlib(diag, f"quad_meshing_inspector_agent{diag.agent_idx}_step{self.inspector.step_count}.png", self.use_probs)
        elif key == pg.K_f:
            self.view_center = None
            self.view_scale = None
        return True

    def _ensure_view(self, points: torch.Tensor):
        if self.view_center is None or self.view_scale is None:
            self.view_center, self.view_scale = ViewTransform.fit(self.width, self.height, points)

    def _transform(self) -> ViewTransform:
        assert self.view_center is not None and self.view_scale is not None
        return ViewTransform(self.width, self.height, self.view_center, self.view_scale)

    def handle_mouse_down(self, event):
        if event.button == 1:
            self.dragging = True
            self.last_mouse = event.pos
            self.mouse_down_pos = event.pos
            self.mouse_moved = False
        elif event.button == 4:
            self.zoom_at_mouse(1)
        elif event.button == 5:
            self.zoom_at_mouse(-1)

    def handle_mouse_up(self, event, diag: AgentDiagnostics):
        if event.button == 1:
            if not self.mouse_moved:
                action_idx = self.action_at_mouse(diag, self._transform(), event.pos)
                if action_idx is not None:
                    self.inspector.manual_step(diag.agent_idx, action_idx)
            self.dragging = False
            self.last_mouse = None
            self.mouse_down_pos = None

    def handle_mouse_motion(self, event):
        if not self.dragging or self.last_mouse is None or self.view_center is None:
            return
        tr = self._transform()
        dx = event.pos[0] - self.last_mouse[0]
        dy = event.pos[1] - self.last_mouse[1]
        if self.mouse_down_pos is not None:
            total_dx = event.pos[0] - self.mouse_down_pos[0]
            total_dy = event.pos[1] - self.mouse_down_pos[1]
            self.mouse_moved = self.mouse_moved or (total_dx * total_dx + total_dy * total_dy > 16)
        self.view_center = self.view_center + tr.world_delta(dx, dy)
        self.last_mouse = event.pos

    def zoom_at_mouse(self, wheel_y: int):
        if self.view_scale is None:
            return
        factor = 1.15 ** wheel_y
        self.view_scale = max(self.view_scale * factor, 1e-6)

    def draw_text(self, text: str, xy: tuple[int, int], color=(235, 235, 235)):
        self.screen.blit(self.font.render(text, True, color), xy)

    def draw(self, diag: AgentDiagnostics):
        pg = self.pygame
        self.screen.fill((18, 18, 22))
        points = diag.vertices if diag.target_positions is None else torch.cat([diag.vertices, diag.target_positions], dim=0)
        self._ensure_view(points)
        tr = self._transform()

        for e in range(diag.edges.size(1)):
            s = int(diag.edges[0, e].item())
            d = int(diag.edges[1, e].item())
            pg.draw.line(self.screen, (105, 105, 110), tr.screen(diag.vertices[s]), tr.screen(diag.vertices[d]), 1)

        if diag.substep == 0:
            values = diag.comparison_score_delta if self.show_comparison_delta and diag.comparison_score_delta is not None else diag.probabilities if self.use_probs else diag.scores
            colors = score_colors(values, self.use_probs and not self.show_comparison_delta)
        else:
            colors = [(170, 170, 180)] * diag.vertices.size(0)

        for i, p in enumerate(diag.vertices):
            color = colors[i] if i < len(colors) else (170, 170, 180)
            radius = 7 if diag.source_idx is not None and i == diag.source_idx else 4
            pg.draw.circle(self.screen, color, tr.screen(p), radius)

        if diag.substep == 1 and diag.target_positions is not None:
            values = self.overlay_values(diag)
            colors = score_colors(values, self.use_probs and not (self.show_global_delta or self.show_comparison_delta))
            for i, p in enumerate(diag.target_positions):
                existing = bool(diag.target_idx is not None and int(diag.target_idx[i].item()) >= 0)
                radius = 6 if existing else 4
                pg.draw.circle(self.screen, colors[i], tr.screen(p), radius, 0 if not existing else 2)

        hover = self.hover_text(diag, tr)

        scores = diag.probabilities if self.use_probs else diag.scores
        topk = torch.topk(torch.nan_to_num(scores, neginf=-1e30), k=min(5, scores.numel())) if scores.numel() else None
        mode = "prob" if self.use_probs else "logit"
        overlay = " global-delta" if self.show_global_delta else " compare-delta" if self.show_comparison_delta else ""
        self.draw_text(f"agent={diag.agent_idx} substep={diag.substep} step={self.inspector.step_count} mode={mode}{overlay}", (12, 10))
        self.draw_text(f"value={float(diag.values.flatten()[0]):.4f} nodes={diag.vertices.size(0)} actions={diag.scores.numel()}", (12, 32))
        if diag.source_idx is not None:
            self.draw_text(f"source_idx={diag.source_idx}", (12, 54))
        if topk is not None:
            pairs = ", ".join(f"{int(i)}:{float(v):.3g}" for v, i in zip(topk.values, topk.indices))
            self.draw_text(f"top {mode}: {pairs}", (12, 76))
        if hover is not None:
            self.draw_text(hover, (12, 98), (255, 230, 150))
        if self.show_global_delta and diag.global_rank_delta is not None:
            self.draw_text(self.rank_change_text(diag), (12, 120), (180, 220, 255))
        self.draw_text("Click act | Space step | Shift+Space x20 | drag pan | wheel zoom | F fit | R reset | Left/Right agent | P prob/logit | G global | C compare | M export | Q quit", (12, self.height - 28), (190, 190, 195))
        pg.display.flip()

    def overlay_values(self, diag: AgentDiagnostics) -> torch.Tensor:
        if self.show_global_delta and diag.global_score_delta is not None:
            return diag.global_score_delta
        if self.show_comparison_delta and diag.comparison_score_delta is not None:
            return diag.comparison_score_delta
        return diag.probabilities if self.use_probs else diag.scores

    def rank_change_text(self, diag: AgentDiagnostics) -> str:
        assert diag.global_rank_delta is not None
        changes = diag.global_rank_delta.abs()
        if changes.numel() == 0:
            return "global rank changes: none"
        vals, idx = torch.topk(changes, k=min(5, changes.numel()))
        parts = []
        for change, i in zip(vals, idx):
            if int(change.item()) == 0:
                continue
            delta = int(diag.global_rank_delta[i].item())
            parts.append(f"{int(i)}:{delta:+d}")
        return "global rank changes: " + (", ".join(parts) if parts else "none")

    def action_at_mouse(self, diag: AgentDiagnostics, tr: ViewTransform, pos: tuple[int, int]) -> int | None:
        mx, my = pos
        best = None
        if diag.substep == 0:
            points = diag.vertices
        elif diag.target_positions is not None:
            points = diag.target_positions
        else:
            return None
        for i, p in enumerate(points):
            sx, sy = tr.screen(p)
            d2 = (sx - mx) * (sx - mx) + (sy - my) * (sy - my)
            if best is None or d2 < best[0]:
                best = (d2, i)
        if best is None or best[0] > 14 * 14:
            return None
        return int(best[1])

    def hover_text(self, diag: AgentDiagnostics, tr: ViewTransform) -> str | None:
        mx, my = self.pygame.mouse.get_pos()
        best = None
        for i, p in enumerate(diag.vertices):
            sx, sy = tr.screen(p)
            d2 = (sx - mx) * (sx - mx) + (sy - my) * (sy - my)
            if best is None or d2 < best[0]:
                score = diag.scores[i] if diag.substep == 0 and i < diag.scores.numel() else None
                prob = diag.probabilities[i] if diag.substep == 0 and i < diag.probabilities.numel() else None
                best = (d2, "vertex", i, score, prob)

        if diag.target_positions is not None:
            for i, p in enumerate(diag.target_positions):
                sx, sy = tr.screen(p)
                d2 = (sx - mx) * (sx - mx) + (sy - my) * (sy - my)
                if best is None or d2 < best[0]:
                    best = (d2, "target", i, diag.scores[i], diag.probabilities[i])

        if best is None or best[0] > 14 * 14:
            return None
        _, kind, idx, score, prob = best
        if score is None or prob is None:
            text = f"hover {kind} {idx}"
            if kind == "vertex" and diag.comparison_score_delta is not None and idx < diag.comparison_score_delta.numel():
                text += f" compare_delta={float(diag.comparison_score_delta[idx]):.4g}"
            return text
        text = f"hover {kind} {idx}: logit={float(score):.4g} prob={float(prob):.4g}"
        if kind == "vertex" and diag.comparison_score_delta is not None and idx < diag.comparison_score_delta.numel():
            text += f" compare_delta={float(diag.comparison_score_delta[idx]):.4g}"
        if kind == "target" and diag.target_idx is not None:
            target_idx = int(diag.target_idx[idx].item())
            text += f" target_idx={target_idx}"
            if diag.global_score_delta is not None and idx < diag.global_score_delta.numel():
                text += f" global_delta={float(diag.global_score_delta[idx]):.4g}"
            if diag.global_rank_delta is not None and idx < diag.global_rank_delta.numel():
                text += f" rank_delta={int(diag.global_rank_delta[idx].item()):+d}"
            if diag.comparison_score_delta is not None and idx < diag.comparison_score_delta.numel():
                text += f" compare_delta={float(diag.comparison_score_delta[idx]):.4g}"
        return text


def export_matplotlib(diag: AgentDiagnostics, path: str, use_probs: bool = False):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    x = diag.vertices.numpy()
    for e in range(diag.edges.size(1)):
        s = int(diag.edges[0, e].item())
        d = int(diag.edges[1, e].item())
        ax.plot([x[s, 0], x[d, 0]], [x[s, 1], x[d, 1]], color="0.65", lw=0.8, zorder=1)

    if diag.substep == 0:
        values = (diag.probabilities if use_probs else diag.scores).numpy()
        sc = ax.scatter(x[:, 0], x[:, 1], c=values, cmap="coolwarm", s=28, zorder=3)
        fig.colorbar(sc, ax=ax, label="probability" if use_probs else "logit")
    else:
        ax.scatter(x[:, 0], x[:, 1], color="0.25", s=18, zorder=2)
        if diag.source_idx is not None:
            ax.scatter([x[diag.source_idx, 0]], [x[diag.source_idx, 1]], color="gold", edgecolor="black", s=80, zorder=4, label="source")
        if diag.target_positions is not None:
            t = diag.target_positions.numpy()
            values = (diag.probabilities if use_probs else diag.scores).numpy()
            sc = ax.scatter(t[:, 0], t[:, 1], c=values, cmap="coolwarm", s=24, zorder=3)
            fig.colorbar(sc, ax=ax, label="probability" if use_probs else "logit")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"agent {diag.agent_idx}, substep {diag.substep}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Exported {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect trained quad_meshing policies interactively.")
    parser.add_argument("--config", default="config/quad_meshing.ini", help="Path to quad_meshing ini config.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path, or 'latest'.")
    parser.add_argument("--compare-config", default=None, help="Optional second policy config for disagreement mode.")
    parser.add_argument("--compare-checkpoint", default=None, help="Optional second policy checkpoint for disagreement mode.")
    parser.add_argument("--device", default=None, help="Torch device, default cuda.")
    parser.add_argument("--total-agents", type=int, default=4, help="Number of env agents to inspect.")
    parser.add_argument("--seed", type=int, default=None, help="Override env seed.")
    parser.add_argument("--export-view", default=None, help="Export one matplotlib view and exit.")
    parser.add_argument("--agent", type=int, default=0, help="Agent index for export or initial view.")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def main():
    args = parse_args()
    inspector = QuadMeshingInspector(
        config_path=args.config,
        checkpoint=args.checkpoint,
        compare_config_path=args.compare_config,
        compare_checkpoint=args.compare_checkpoint,
        device=args.device,
        total_agents=args.total_agents,
        seed=args.seed,
    )
    try:
        if args.export_view:
            export_matplotlib(inspector.diagnostics(args.agent), args.export_view)
            return
        app = PygameInspectorApp(inspector, args.width, args.height)
        app.agent_idx = args.agent % inspector.vec.total_agents
        app.run()
    finally:
        inspector.close()


if __name__ == "__main__":
    main()
