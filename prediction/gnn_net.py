"""The GNNs' networks: dense message passing over padded graphs, one graph per grid row
(FrameGNN, 05 model 2) or a window of them through a GRU (TemporalGNN, 05 model 3).

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


def readout(h: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    """Mean, max and sum over the real nodes, and the ball node's embedding (slot 0)."""
    valid = mask.unsqueeze(-1).to(h.dtype)
    count = valid.sum(1).clamp(min=1.0)
    mean = (h * valid).sum(1) / count
    top = h.masked_fill(~mask.unsqueeze(-1), torch.finfo(h.dtype).min).amax(1)
    top = torch.where(mask.any(1, keepdim=True), top, torch.zeros_like(top))
    total = (h * valid).sum(1) / MAX_NODES
    ball = h[:, 0] * x[:, 0, IDX["is_ball"]].unsqueeze(-1)
    return [mean, top, total, ball]


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
            parts += readout(h, x, mask)
        if self.glob is not None:
            parts.append(self.glob(g))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


class TemporalGNN(nn.Module):
    """05 model 3: every step of a window through the same message passing and readout
    as the frame GNN, a GRU over the steps (oldest first), then a head on the GRU's last
    state, the anchor step's readout and the global vector. A step with no nodes (no row,
    or nothing visible) keeps the previous state. Written apart from FrameGNN so the
    frame GNN's modules, and so its init and results, stay as they were."""

    def __init__(
        self, n_node: int, n_glob: int, d: int = 64, layers: int = 3, dropout: float = 0.1
    ):
        super().__init__()
        if not layers:
            raise ValueError("the temporal GNN needs message passing (layers >= 1)")
        self.encode = nn.Sequential(nn.Linear(n_node, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList(
            MessageLayer(d, len(EDGE_FEATURES), dropout) for _ in range(layers)
        )
        self.step = nn.Sequential(nn.Linear(4 * d, d), nn.GELU())
        self.gru = nn.GRUCell(d, d)
        width = d + 4 * d
        self.glob = None
        if n_glob:
            self.glob = nn.Sequential(nn.Linear(2 * n_glob, d), nn.GELU(), nn.Linear(d, d))
            width += d
        self.head = nn.Sequential(
            nn.Linear(width, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1)
        )

    def graphs(self, x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """(M, N, F) nodes, (M,) counts -> (M, 4d) readouts."""
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < n[:, None]
        e = edge_features(x)
        h = self.encode(x) * mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, e, mask)
        return torch.cat(readout(h, x, mask), dim=-1)

    def forward(self, x: torch.Tensor, n: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """x (B, T, N, F) windows, anchor last, n (B, T) node counts (0: no step), g (B,
        2G) the anchor's globals -> (B,) logits."""
        b, t = n.shape
        r = self.graphs(x.flatten(0, 1), n.flatten()).view(b, t, -1)
        z = self.step(r)
        h = z.new_zeros(b, z.shape[-1])
        for s in range(t):
            h = torch.where((n[:, s] > 0).unsqueeze(-1), self.gru(z[:, s], h), h)
        parts = [h, r[:, -1]]
        if self.glob is not None:
            parts.append(self.glob(g))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)
