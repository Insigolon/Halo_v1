"""
encoder.py
──────────
Unified Multimodal Encoder.

Architecture:
    adapter.tokenize(tensor)          →  [N, 768]  tokens
    transformer_backbone([N, 768])    →  [N, 768]  contextualised tokens
    pool + projection                 →  [512]     final embedding

The backbone is modality-agnostic — it sees tokens, not sensors.
This means a new sensor only ever needs a new adapter; the backbone
is trained once and reused across all modalities via contrastive loss.

Training strategy (contrastive):
    A LiDAR scan and an RGB photo of the same scene are "positive pairs".
    The loss pushes their 512-d embeddings close together.
    When you add a new sensor, collect ~1000 paired observations,
    train just the new adapter for a few hundred steps, done.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


# ─────────────────────────────────────────────
# Positional Encoding (learnable)
# ─────────────────────────────────────────────

class LearnablePositionalEncoding:
    """
    Simple learnable positional embedding table.
    Supports sequences up to max_len tokens.
    Weights are initialised to small random values and trained end-to-end.
    """

    def __init__(self, max_len: int = 512, dim: int = 768):
        self.weights = np.random.randn(max_len, dim).astype(np.float32) * 0.02

    def __call__(self, n_tokens: int) -> np.ndarray:
        """Returns positional embeddings for a sequence of n_tokens."""
        if n_tokens > len(self.weights):
            # Extend if sequence is longer than pre-allocated table
            extra = np.random.randn(n_tokens - len(self.weights),
                                    self.weights.shape[1]).astype(np.float32) * 0.02
            self.weights = np.concatenate([self.weights, extra], axis=0)
        return self.weights[:n_tokens]


# ─────────────────────────────────────────────
# Modality Token
# ─────────────────────────────────────────────

class ModalityTokenTable:
    """
    One learned [768] vector per modality — prepended to the token sequence
    so the transformer knows which sensor produced the tokens.
    Similar to the [CLS] token in BERT or modality tokens in ImageBind.
    """

    def __init__(self, dim: int = 768):
        self._dim    = dim
        self._tokens: dict[str, np.ndarray] = {}

    def get_or_create(self, modality_id: str) -> np.ndarray:
        if modality_id not in self._tokens:
            self._tokens[modality_id] = (
                np.random.randn(1, self._dim).astype(np.float32) * 0.02
            )
            print(f"[ModalityTokenTable] Created token for modality: {modality_id}")
        return self._tokens[modality_id]   # [1, 768]


# ─────────────────────────────────────────────
# Transformer Backbone (pure NumPy for portability)
# Replace with torch.nn.TransformerEncoder for production
# ─────────────────────────────────────────────

class TransformerBackbone:
    """
    Lightweight multi-head self-attention transformer.
    
    - 4 layers, 8 heads, dim=768, ffn_dim=2048
    - Uses numpy for CPU inference (no torch dependency at runtime)
    - In training, swap this for a standard torch TransformerEncoder
      and export weights to numpy arrays after training.
    
    Weights are initialised randomly here and MUST be replaced
    with trained weights before real use.
    """

    def __init__(self, num_layers: int = 4, dim: int = 768,
                 num_heads: int = 8, ffn_dim: int = 2048):
        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.layers    = [
            self._init_layer(dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ]

    def _init_layer(self, dim, num_heads, ffn_dim) -> dict:
        s = 0.02
        return {
            # Multi-head self attention projections
            "Wq":  np.random.randn(dim, dim).astype(np.float32) * s,
            "Wk":  np.random.randn(dim, dim).astype(np.float32) * s,
            "Wv":  np.random.randn(dim, dim).astype(np.float32) * s,
            "Wo":  np.random.randn(dim, dim).astype(np.float32) * s,
            # Feed-forward
            "W1":  np.random.randn(dim, ffn_dim).astype(np.float32) * s,
            "b1":  np.zeros(ffn_dim, dtype=np.float32),
            "W2":  np.random.randn(ffn_dim, dim).astype(np.float32) * s,
            "b2":  np.zeros(dim, dtype=np.float32),
            # Layer norms
            "ln1_g": np.ones(dim, dtype=np.float32),
            "ln1_b": np.zeros(dim, dtype=np.float32),
            "ln2_g": np.ones(dim, dtype=np.float32),
            "ln2_b": np.zeros(dim, dtype=np.float32),
        }

    def _layer_norm(self, x: np.ndarray, g: np.ndarray, b: np.ndarray,
                    eps: float = 1e-6) -> np.ndarray:
        mu  = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + eps
        return g * (x - mu) / std + b

    def _mhsa(self, x: np.ndarray, layer: dict) -> np.ndarray:
        """Multi-head self-attention (simplified, no masking)."""
        N, D = x.shape
        Q = x @ layer["Wq"]
        K = x @ layer["Wk"]
        V = x @ layer["Wv"]

        # Reshape to [heads, N, head_dim]
        Q = Q.reshape(N, self.num_heads, self.head_dim).transpose(1, 0, 2)
        K = K.reshape(N, self.num_heads, self.head_dim).transpose(1, 0, 2)
        V = V.reshape(N, self.num_heads, self.head_dim).transpose(1, 0, 2)

        scale  = self.head_dim ** -0.5
        scores = Q @ K.transpose(0, 2, 1) * scale   # [heads, N, N]
        # Softmax
        scores = scores - scores.max(axis=-1, keepdims=True)
        attn   = np.exp(scores)
        attn  /= attn.sum(axis=-1, keepdims=True)

        out = attn @ V   # [heads, N, head_dim]
        out = out.transpose(1, 0, 2).reshape(N, D)
        return out @ layer["Wo"]

    def _ffn(self, x: np.ndarray, layer: dict) -> np.ndarray:
        h = np.maximum(0, x @ layer["W1"] + layer["b1"])   # ReLU
        return h @ layer["W2"] + layer["b2"]

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """
        tokens: [N, 768]
        output: [N, 768]  contextualised token representations
        """
        x = tokens
        for layer in self.layers:
            # Self-attention with residual
            x = x + self._mhsa(self._layer_norm(x, layer["ln1_g"], layer["ln1_b"]), layer)
            # FFN with residual
            x = x + self._ffn(self._layer_norm(x, layer["ln2_g"], layer["ln2_b"]), layer)
        return x

    def load_weights(self, path: str) -> None:
        """Load pre-trained weights from a .npz file."""
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            for key in layer:
                k = f"layer{i}_{key}"
                if k in data:
                    layer[key] = data[k]
        print(f"[TransformerBackbone] Weights loaded from {path}")

    def save_weights(self, path: str) -> None:
        """Save weights to a .npz file."""
        payload = {}
        for i, layer in enumerate(self.layers):
            for key, val in layer.items():
                payload[f"layer{i}_{key}"] = val
        np.savez(path, **payload)


# ─────────────────────────────────────────────
# Pooling + Projection head
# ─────────────────────────────────────────────

class EmbeddingHead:
    """
    Collapses [N, 768] token sequence → [512] final embedding.
    Uses the CLS token (first position = modality token) + mean pool,
    then a learned linear projection + L2 normalisation.
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 512):
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * 0.02
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """
        tokens: [N, 768]  (first token is the CLS/modality token)
        output: [512]     L2-normalised embedding
        """
        # CLS token + mean of the rest
        cls_token  = tokens[0]
        mean_token = tokens[1:].mean(axis=0)
        pooled     = (cls_token + mean_token) / 2.0   # [768]

        projected  = pooled @ self.W + self.b          # [512]

        # L2 normalise so cosine similarity == dot product
        norm = np.linalg.norm(projected)
        if norm > 1e-8:
            projected /= norm
        return projected


# ─────────────────────────────────────────────
# Unified Encoder
# ─────────────────────────────────────────────

class UnifiedEncoder:
    """
    The single encoder for all sensor types.
    
    Usage:
        encoder  = UnifiedEncoder()
        embedding = encoder.encode(modality_id="rgb", tensor=preprocessed)
    
    To support a new sensor: register its adapter in AdapterRegistry.
    No changes needed here.
    """

    def __init__(self,
                 backbone:  Optional[TransformerBackbone] = None,
                 head:      Optional[EmbeddingHead]       = None,
                 pos_enc:   Optional[LearnablePositionalEncoding] = None,
                 mod_tokens: Optional[ModalityTokenTable] = None):
        self.backbone   = backbone   or TransformerBackbone()
        self.head       = head       or EmbeddingHead()
        self.pos_enc    = pos_enc    or LearnablePositionalEncoding()
        self.mod_tokens = mod_tokens or ModalityTokenTable()

    def encode(self, modality_id: str, tensor: np.ndarray) -> np.ndarray:
        """
        tensor:     preprocessed sensor data as returned by adapter.preprocess()
        modality_id: string key for the modality token

        Returns [512] L2-normalised embedding.
        """
        # 1. Tokenize via the adapter (already done by caller — tensor IS the tokens)
        tokens = tensor   # [N, 768]

        # 2. Prepend modality token
        mod_tok = self.mod_tokens.get_or_create(modality_id)   # [1, 768]
        tokens  = np.concatenate([mod_tok, tokens], axis=0)    # [N+1, 768]

        # 3. Add positional encoding
        tokens = tokens + self.pos_enc(len(tokens))

        # 4. Transformer backbone
        tokens = self.backbone.forward(tokens)

        # 5. Pool + project to 512-d
        embedding = self.head.forward(tokens)

        return embedding   # [512] float32

    def load_weights(self, backbone_path: str, head_path: str) -> None:
        self.backbone.load_weights(backbone_path)
        data = np.load(head_path)
        self.head.W = data["W"]
        self.head.b = data["b"]
        print("[UnifiedEncoder] All weights loaded.")