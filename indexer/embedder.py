from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class ClipEmbedder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._device = "cpu"
        self._dim = 512

        self._init_model()

    @property
    def dim(self) -> int:
        return self._dim

    def _init_model(self) -> None:
        try:
            import torch  # type: ignore
            import open_clip  # type: ignore
        except Exception:
            LOGGER.warning("OpenCLIP not available. Falling back to deterministic embeddings.")
            return

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )

            LOGGER.info("Using OpenCLIP pretrained='%s'", self.pretrained)

            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._device = "cpu"
            self._dim = int(model.text_projection.shape[1]) if hasattr(model, "text_projection") else 512
            LOGGER.info("OpenCLIP model ready: %s", self.model_name)
        except Exception as exc:
            LOGGER.warning("Failed to load OpenCLIP, using fallback embeddings: %s", exc)
            self._model = None

    def embed_image(self, image_bgr: np.ndarray) -> np.ndarray:
        if self._model is None or self._preprocess is None:
            return self._fallback_embedding(image_bgr)

        try:
            import torch  # type: ignore
            from PIL import Image  # type: ignore
        except Exception:
            return self._fallback_embedding(image_bgr)

        image_rgb = image_bgr[:, :, ::-1]
        pil = Image.fromarray(image_rgb)
        image = self._preprocess(pil).unsqueeze(0)
        with torch.no_grad():
            features = self._model.encode_image(image)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def _fallback_embedding(self, image_bgr: np.ndarray) -> np.ndarray:
        # Deterministic embedding based on image bytes
        digest = hashlib.sha256(image_bgr.tobytes()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        vec = rng.standard_normal(self._dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        return vec
