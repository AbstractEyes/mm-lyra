# ComfyUI-Lyra/nodes.py

"""
ComfyUI Nodes for VAE Lyra
==========================

Nodes:
- LyraLoader: Load VAE Lyra model from HuggingFace or local path
- LyraEncode: Encode CLIP/T5 embeddings to latent space
- LyraDecode: Decode latent to reconstructed embeddings
- LyraFullPass: Full encode-decode pass
"""

import torch
import folder_paths
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from .lyra_loader import (
    load_vae_lyra,
    load_vae_lyra_local,
    get_available_models,
    get_model_info,
    KNOWN_MODELS,
)


# ============================================================================
# LOADER NODE
# ============================================================================

class LyraLoader:
    """Load VAE Lyra model from HuggingFace Hub or local path."""

    CATEGORY = "Lyra"
    FUNCTION = "load"
    RETURN_TYPES = ("LYRA_MODEL", "LYRA_INFO")
    RETURN_NAMES = ("model", "info")

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = list(KNOWN_MODELS.keys())

        return {
            "required": {
                "source": (["huggingface", "local"],),
                "repo_id": (model_choices, {"default": model_choices[-1] if model_choices else ""}),
            },
            "optional": {
                "local_path": ("STRING", {"default": "", "multiline": False}),
                "force_version": (["auto", "v1", "v2"], {"default": "auto"}),
            }
        }

    def load(
            self,
            source: str,
            repo_id: str,
            local_path: str = "",
            force_version: str = "auto"
    ) -> Tuple[Any, Dict]:

        device = "cuda" if torch.cuda.is_available() else "cpu"
        version = None if force_version == "auto" else force_version

        if source == "huggingface":
            print(f"[Lyra] Loading from HuggingFace: {repo_id}")
            model = load_vae_lyra(repo_id, device=device, force_version=version)
            info = get_model_info(repo_id)
        else:
            if not local_path:
                raise ValueError("Local path required when source is 'local'")
            print(f"[Lyra] Loading from local: {local_path}")
            model = load_vae_lyra_local(local_path, device=device, force_version=version)
            info = {
                "source": "local",
                "path": local_path,
                "version": version or "auto-detected"
            }

        return (model, info)


# ============================================================================
# ENCODE NODE
# ============================================================================

class LyraEncode:
    """Encode CLIP/T5 embeddings to Lyra latent space."""

    CATEGORY = "Lyra"
    FUNCTION = "encode"
    RETURN_TYPES = ("LATENT", "LATENT", "LYRA_PER_MOD_MU")
    RETURN_NAMES = ("mu", "logvar", "per_modality_mus")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LYRA_MODEL",),
            },
            "optional": {
                "clip_l": ("CONDITIONING",),
                "clip_g": ("CONDITIONING",),
                "t5_xl_l": ("CONDITIONING",),
                "t5_xl_g": ("CONDITIONING",),
            }
        }

    def encode(
            self,
            model,
            clip_l=None,
            clip_g=None,
            t5_xl_l=None,
            t5_xl_g=None
    ) -> Tuple[Dict, Dict, Optional[Dict]]:

        # Build input dict from provided embeddings
        inputs = {}

        if clip_l is not None:
            inputs["clip_l"] = self._extract_embedding(clip_l)
        if clip_g is not None:
            inputs["clip_g"] = self._extract_embedding(clip_g)
        if t5_xl_l is not None:
            inputs["t5_xl_l"] = self._extract_embedding(t5_xl_l)
        if t5_xl_g is not None:
            inputs["t5_xl_g"] = self._extract_embedding(t5_xl_g)

        if not inputs:
            raise ValueError("At least one embedding input required")

        # Encode
        with torch.no_grad():
            result = model.encode(inputs)

            # v1 returns (mu, logvar, None), v2 returns (mu, logvar, per_mod_mus)
            if len(result) == 2:
                mu, logvar = result
                per_mod_mus = None
            else:
                mu, logvar, per_mod_mus = result

        return (
            {"samples": mu},
            {"samples": logvar},
            per_mod_mus
        )

    def _extract_embedding(self, conditioning):
        """Extract embedding tensor from ComfyUI conditioning format."""
        if isinstance(conditioning, torch.Tensor):
            return conditioning
        if isinstance(conditioning, list) and len(conditioning) > 0:
            # ComfyUI conditioning format: [(embedding, metadata), ...]
            return conditioning[0][0]
        raise ValueError(f"Unknown conditioning format: {type(conditioning)}")


# ============================================================================
# DECODE NODE
# ============================================================================

class LyraDecode:
    """Decode Lyra latent to reconstructed embeddings."""

    CATEGORY = "Lyra"
    FUNCTION = "decode"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("clip_l", "clip_g", "t5_xl_l", "t5_xl_g")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LYRA_MODEL",),
                "mu": ("LATENT",),
                "logvar": ("LATENT",),
            },
            "optional": {
                "sample": ("BOOLEAN", {"default": True}),
                "target_modalities": ("STRING", {"default": "clip_l,clip_g"}),
            }
        }

    def decode(
            self,
            model,
            mu: Dict,
            logvar: Dict,
            sample: bool = True,
            target_modalities: str = "clip_l,clip_g"
    ) -> Tuple:

        mu_tensor = mu["samples"]
        logvar_tensor = logvar["samples"]

        # Parse target modalities
        targets = [t.strip() for t in target_modalities.split(",") if t.strip()]
        if not targets:
            targets = ["clip_l", "clip_g"]

        with torch.no_grad():
            if sample:
                z = model.reparameterize(mu_tensor, logvar_tensor)
            else:
                z = mu_tensor

            reconstructions = model.decode(z, target_modalities=targets)

        # Build output tuple (always 4 outputs, None for missing)
        outputs = []
        for modality in ["clip_l", "clip_g", "t5_xl_l", "t5_xl_g"]:
            if modality in reconstructions:
                # Convert to ComfyUI conditioning format
                outputs.append([(reconstructions[modality], {})])
            else:
                outputs.append(None)

        return tuple(outputs)


# ============================================================================
# FULL PASS NODE
# ============================================================================

class LyraFullPass:
    """Full encode-decode pass through Lyra."""

    CATEGORY = "Lyra"
    FUNCTION = "process"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "LATENT")
    RETURN_NAMES = ("clip_l_recon", "clip_g_recon", "mu", "logvar")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LYRA_MODEL",),
            },
            "optional": {
                "clip_l": ("CONDITIONING",),
                "clip_g": ("CONDITIONING",),
                "t5_xl_l": ("CONDITIONING",),
                "t5_xl_g": ("CONDITIONING",),
                "target_modalities": ("STRING", {"default": "clip_l,clip_g"}),
            }
        }

    def process(
            self,
            model,
            clip_l=None,
            clip_g=None,
            t5_xl_l=None,
            t5_xl_g=None,
            target_modalities: str = "clip_l,clip_g"
    ) -> Tuple:

        # Build inputs
        inputs = {}
        if clip_l is not None:
            inputs["clip_l"] = self._extract_embedding(clip_l)
        if clip_g is not None:
            inputs["clip_g"] = self._extract_embedding(clip_g)
        if t5_xl_l is not None:
            inputs["t5_xl_l"] = self._extract_embedding(t5_xl_l)
        if t5_xl_g is not None:
            inputs["t5_xl_g"] = self._extract_embedding(t5_xl_g)

        if not inputs:
            raise ValueError("At least one embedding input required")

        # Parse targets
        targets = [t.strip() for t in target_modalities.split(",") if t.strip()]
        if not targets:
            targets = ["clip_l", "clip_g"]

        # Full forward pass
        with torch.no_grad():
            result = model(inputs, target_modalities=targets)

            # v1: (recons, mu, logvar), v2: (recons, mu, logvar, per_mod_mus)
            if len(result) == 3:
                reconstructions, mu, logvar = result
            else:
                reconstructions, mu, logvar, _ = result

        # Build outputs
        clip_l_out = [(reconstructions["clip_l"], {})] if "clip_l" in reconstructions else None
        clip_g_out = [(reconstructions["clip_g"], {})] if "clip_g" in reconstructions else None

        return (
            clip_l_out,
            clip_g_out,
            {"samples": mu},
            {"samples": logvar}
        )

    def _extract_embedding(self, conditioning):
        if isinstance(conditioning, torch.Tensor):
            return conditioning
        if isinstance(conditioning, list) and len(conditioning) > 0:
            return conditioning[0][0]
        raise ValueError(f"Unknown conditioning format: {type(conditioning)}")


# ============================================================================
# INFO NODE
# ============================================================================

class LyraInfo:
    """Display Lyra model information."""

    CATEGORY = "Lyra"
    FUNCTION = "show_info"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info_text",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LYRA_MODEL",),
            }
        }

    def show_info(self, model) -> Tuple[str]:
        lines = ["=== VAE Lyra Model Info ==="]

        # Get config
        if hasattr(model, 'config'):
            config = model.config
            lines.append(f"Fusion: {config.fusion_strategy}")
            lines.append(f"Latent dim: {config.latent_dim}")
            lines.append(f"Modalities: {list(config.modality_dims.keys())}")

            if hasattr(config, 'modality_seq_lens'):
                lines.append(f"Seq lens: {config.modality_seq_lens}")

            if hasattr(config, 'binding_config') and config.binding_config:
                lines.append("Binding groups:")
                for target, sources in config.binding_config.items():
                    if sources:
                        lines.append(f"  {target} ↔ {list(sources.keys())}")

        # Get learned params
        if hasattr(model, 'get_fusion_params'):
            params = model.get_fusion_params()
            if params:
                if 'alphas' in params:
                    lines.append("Alphas (visibility):")
                    for name, alpha in params['alphas'].items():
                        lines.append(f"  {name}: {torch.sigmoid(alpha).item():.4f}")
                if 'betas' in params:
                    lines.append("Betas (capacity):")
                    for name, beta in params['betas'].items():
                        lines.append(f"  {name}: {torch.sigmoid(beta).item():.4f}")

        # Param count
        total_params = sum(p.numel() for p in model.parameters())
        lines.append(f"Parameters: {total_params:,}")

        info_text = "\n".join(lines)
        print(info_text)

        return (info_text,)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LyraLoader": LyraLoader,
    "LyraEncode": LyraEncode,
    "LyraDecode": LyraDecode,
    "LyraFullPass": LyraFullPass,
    "LyraInfo": LyraInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LyraLoader": "Lyra Loader",
    "LyraEncode": "Lyra Encode",
    "LyraDecode": "Lyra Decode",
    "LyraFullPass": "Lyra Full Pass",
    "LyraInfo": "Lyra Info",
}