"""The frame GNN's network (05 model 2): dense message passing over padded graphs.

Kept apart from prediction/gnn.py so that importing the model wrapper (and so
prediction.cv) doesn't need torch.

Every graph is fully connected, self-loops included, and padding nodes are masked out.
LayerNorm only, never BatchNorm: a row's output must not depend on the other rows in
its batch.
"""

import torch
import torch.nn.functional as F
from torch import nn

from prediction.graphs import IDX, MAX_NODES, VEL_SCALE, X_SCALE, Y_SCALE

EDGE_FEATURES = ("dx", "dy", "dist", "dvx", "dvy", "both_vel")
EDGE_M = 20.0  # dx, dy, distance in units of 20 m
EDGE_V = 10.0  # relative velocity in units of 10 m/s


def edge_features(x: torch.Tensor) -> torch.Tensor:
    """(B, N, F) node features -> (B, N, N, 6) edge features, [b, i, j] = sender j seen
    from receiver i. Positions go back to meters first: x and y are scaled differently,
    so distances on the scaled values would squash the pitch's width."""
    px, py = x[..., IDX["x"]] * X_SCALE, x[..., IDX["y"]] * Y_SCALE
    vx, vy = x[..., IDX["vx"]] * VEL_SCALE, x[..., IDX["vy"]] * VEL_SCALE
    both = x[..., IDX["has_vel"]][:, :, None] * x[..., IDX["has_vel"]][:, None, :]
    dx = px[:, None, :] - px[:, :, None]
    dy = py[:, None, :] - py[:, :, None]
    dvx = (vx[:, None, :] - vx[:, :, None]) * both
    dvy = (vy[:, None, :] - vy[:, :, None]) * both
    dist = torch.sqrt(dx * dx + dy * dy)
    return torch.stack(
        [dx / EDGE_M, dy / EDGE_M, dist / EDGE_M, dvx / EDGE_V, dvy / EDGE_V, both], dim=-1
    )


class MessageLayer(nn.Module):
    """One round: a message per (receiver, sender) pair from both nodes and the edge,
    combined by an attention-weighted mean and a plain sum (counts matter: defenders in
    the box), then a residual update."""

    def __init__(self, d: int, n_edge: int, dropout: float):
        super().__init__()
        self.recv = nn.Linear(d, d)
        self.send = nn.Linear(d, d, bias=False)
        self.edge = nn.Linear(n_edge, d, bias=False)
        self.msg = nn.Linear(d, d)
        self.att = nn.Linear(d, 1)
        self.update = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d)
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.recv(h)[:, :, None] + self.send(h)[:, None, :] + self.edge(e))
        m = self.msg(z)  # (B, N, N, d)
        s = self.att(z).squeeze(-1)  # (B, N, N)
        send = mask[:, None, :]  # padding never sends
        s = s.masked_fill(~send, torch.finfo(s.dtype).min)
        a = torch.softmax(s, dim=-1) * send  # an empty graph gets all zeros, not NaN
        mean = torch.matmul(a.unsqueeze(-2), m).squeeze(-2)
        total = torch.matmul(send.to(m.dtype).unsqueeze(-2), m).squeeze(-2) / MAX_NODES
        h = self.norm(h + self.update(torch.cat([h, mean, total], dim=-1)))
        return h * mask.unsqueeze(-1)


class FrameGNN(nn.Module):
    """Graph readout (mean, max and sum over the nodes, plus the ball node) and the
    global vector, then an MLP to one logit. layers=0 switches the graph off: the same
    head on the global vector alone (05's globals-only arm)."""

    def __init__(
        self, n_node: int, n_glob: int, d: int = 64, layers: int = 3, dropout: float = 0.1
    ):
        super().__init__()
        self.n_layers = layers
        width = 0
        if layers:
            self.encode = nn.Sequential(nn.Linear(n_node, d), nn.GELU(), nn.Linear(d, d))
            self.layers = nn.ModuleList(
                MessageLayer(d, len(EDGE_FEATURES), dropout) for _ in range(layers)
            )
            width += 4 * d
        self.glob = None
        if n_glob:  # each global feature comes with its missing flag
            self.glob = nn.Sequential(nn.Linear(2 * n_glob, d), nn.GELU(), nn.Linear(d, d))
            width += d
        if not width:
            raise ValueError("nothing to read: no message passing and no global features")
        self.head = nn.Sequential(
            nn.Linear(width, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1)
        )

    def forward(self, x: torch.Tensor, n: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """x (B, N, F) nodes, n (B,) node counts, g (B, 2G) globals -> (B,) logits."""
        parts = []
        if self.n_layers:
            mask = torch.arange(x.shape[1], device=x.device)[None, :] < n[:, None]
            e = edge_features(x)
            h = self.encode(x) * mask.unsqueeze(-1)
            for layer in self.layers:
                h = layer(h, e, mask)
            valid = mask.unsqueeze(-1).to(h.dtype)
            count = valid.sum(1).clamp(min=1.0)
            mean = (h * valid).sum(1) / count
            top = h.masked_fill(~mask.unsqueeze(-1), torch.finfo(h.dtype).min).amax(1)
            top = torch.where(mask.any(1, keepdim=True), top, torch.zeros_like(top))
            total = (h * valid).sum(1) / MAX_NODES
            ball = h[:, 0] * x[:, 0, IDX["is_ball"]].unsqueeze(-1)  # the ball is slot 0
            parts += [mean, top, total, ball]
        if self.glob is not None:
            parts.append(self.glob(g))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)
