# ComfyUI-Lyra/nodes.py

"""
ComfyUI Nodes for VAE Lyra
==========================

Nodes:
- EncoderLoader: Load T5/Qwen/Llama text encoders
- LyraLoader: Load VAE Lyra model
- LyraEncodeConfiguration: Settings for encode behavior
- LyraEncode: Encode with optional cond inputs
- LyraEncodeSummary: Ease-of-use encode with summary support
- LyraDecode: Decode latents back to conds
"""

import torch
from typing import Dict, Any, Tuple, Optional, List
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .lyra_loader import (
    load_vae_lyra,
    load_vae_lyra_local,
    KNOWN_MODELS,
)


# ============================================================================
# ENCODER LOADER
# ============================================================================

class LyraEncoderLoader:
    """
    Load text encoder (T5, Qwen, Llama) from HuggingFace.

    Outputs ENCODER_PIPE containing model, tokenizer, and config.
    """

    CATEGORY = "Lyra/Loaders"
    FUNCTION = "load"
    RETURN_TYPES = ("ENCODER_PIPE",)
    RETURN_NAMES = ("encoder_pipe",)

    # Common presets for convenience
    ENCODER_PRESETS = [
        "google/flan-t5-base",
        "google/flan-t5-xl",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
        "meta-llama/Llama-3.2-1B",
        "custom",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (cls.ENCODER_PRESETS, {"default": "google/flan-t5-xl"}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            },
            "optional": {
                "custom_repo_id": ("STRING", {"default": "", "multiline": False}),
                "trust_remote_code": ("BOOLEAN", {"default": False}),
                "torch_dtype": (["float32", "float16", "bfloat16"], {"default": "float32"}),
            }
        }

    def load(
            self,
            preset: str,
            device: str,
            custom_repo_id: str = "",
            trust_remote_code: bool = False,
            torch_dtype: str = "float32"
    ) -> Tuple[Dict]:

        # Determine repo_id
        repo_id = custom_repo_id.strip() if preset == "custom" else preset
        if not repo_id:
            raise ValueError("Must specify custom_repo_id when preset is 'custom'")

        print(f"[EncoderLoader] Loading: {repo_id}")
        print(f"[EncoderLoader] Device: {device}")

        # Determine dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[torch_dtype]

        # Detect encoder type and load appropriately
        repo_lower = repo_id.lower()

        if "t5" in repo_lower or "flan" in repo_lower:
            encoder_type = "t5"
            print(f"[EncoderLoader] Detected T5 encoder")
            tokenizer = T5Tokenizer.from_pretrained(repo_id)
            model = T5EncoderModel.from_pretrained(
                repo_id,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code
            )
        elif "qwen" in repo_lower:
            encoder_type = "qwen"
            print(f"[EncoderLoader] Detected Qwen encoder")
            tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                trust_remote_code=trust_remote_code
            )
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code
            )
        elif "llama" in repo_lower:
            encoder_type = "llama"
            print(f"[EncoderLoader] Detected Llama encoder")
            tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                trust_remote_code=trust_remote_code
            )
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code
            )
        else:
            # Fallback: try T5 first, then AutoModel
            encoder_type = "auto"
            print(f"[EncoderLoader] Auto-detecting encoder type...")
            try:
                tokenizer = T5Tokenizer.from_pretrained(repo_id)
                model = T5EncoderModel.from_pretrained(repo_id, torch_dtype=dtype)
                encoder_type = "t5"
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    repo_id,
                    trust_remote_code=trust_remote_code
                )
                model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    torch_dtype=dtype,
                    trust_remote_code=trust_remote_code
                )

        model.to(device).eval()

        param_count = sum(p.numel() for p in model.parameters())
        print(f"[EncoderLoader] Loaded: {param_count:,} params")

        encoder_pipe = {
            "model": model,
            "tokenizer": tokenizer,
            "repo_id": repo_id,
            "encoder_type": encoder_type,
            "device": device,
            "dtype": dtype,
        }

        return (encoder_pipe,)


# ============================================================================
# LYRA LOADER
# ============================================================================

class LyraLoader:
    """
    Load VAE Lyra model from HuggingFace Hub or local path.

    Outputs LYRA_PIPE containing model and config.
    """

    CATEGORY = "Lyra/Loaders"
    FUNCTION = "load"
    RETURN_TYPES = ("LYRA_PIPE",)
    RETURN_NAMES = ("lyra_pipe",)

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = list(KNOWN_MODELS.keys()) + ["custom_hf", "custom_local"]

        return {
            "required": {
                "source": (model_choices, {"default": model_choices[0] if model_choices else "custom_hf"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                "custom_repo_id": ("STRING", {"default": "", "multiline": False}),
                "local_model_path": ("STRING", {"default": "", "multiline": False}),
                "local_config_path": ("STRING", {"default": "", "multiline": False}),
                "force_version": (["auto", "v1", "v2"], {"default": "auto"}),
            }
        }

    def load(
            self,
            source: str,
            device: str,
            custom_repo_id: str = "",
            local_model_path: str = "",
            local_config_path: str = "",
            force_version: str = "auto"
    ) -> Tuple[Dict]:

        version = None if force_version == "auto" else force_version

        if source == "custom_local":
            if not local_model_path:
                raise ValueError("local_model_path required for custom_local source")
            print(f"[LyraLoader] Loading from local: {local_model_path}")
            config_path = local_config_path if local_config_path else None
            model = load_vae_lyra_local(
                local_model_path,
                config_path=config_path,
                device=device,
                force_version=version
            )
            repo_id = local_model_path
        elif source == "custom_hf":
            if not custom_repo_id:
                raise ValueError("custom_repo_id required for custom_hf source")
            print(f"[LyraLoader] Loading from HuggingFace: {custom_repo_id}")
            model = load_vae_lyra(custom_repo_id, device=device, force_version=version)
            repo_id = custom_repo_id
        else:
            # Known model from registry
            print(f"[LyraLoader] Loading: {source}")
            model = load_vae_lyra(source, device=device, force_version=version)
            repo_id = source

        # Extract config info
        config = model.config if hasattr(model, 'config') else None

        lyra_pipe = {
            "model": model,
            "config": config,
            "repo_id": repo_id,
            "device": device,
            "version": "v2" if hasattr(model, 'encoders') else "v1",
            "modality_dims": config.modality_dims if config else {},
            "modality_seq_lens": getattr(config, 'modality_seq_lens', {}),
            "binding_config": getattr(config, 'binding_config', {}),
        }

        return (lyra_pipe,)


# ============================================================================
# LYRA ENCODE CONFIGURATION
# ============================================================================

class LyraEncodeConfiguration:
    """
    Configuration for Lyra encoding behavior.

    Controls target model, output modalities, seed, and separator settings.
    """

    CATEGORY = "Lyra/Config"
    FUNCTION = "configure"
    RETURN_TYPES = ("LYRA_CONFIG",)
    RETURN_NAMES = ("config",)

    TARGET_MODELS = [
        "sd15",  # clip_l only
        "sdxl",  # clip_l + clip_g
        "sdxl_with_t5",  # clip_l + clip_g + t5 (for SD3/Flux style)
        "flux",  # clip_l + t5
        "custom",  # user-defined
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_model": (cls.TARGET_MODELS, {"default": "sdxl"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                # Custom modality selection (for target_model="custom")
                "output_clip_l": ("BOOLEAN", {"default": True}),
                "output_clip_g": ("BOOLEAN", {"default": True}),
                "output_t5_xl_l": ("BOOLEAN", {"default": False}),
                "output_t5_xl_g": ("BOOLEAN", {"default": False}),

                # Summary separator settings
                "summary_separator": ("STRING", {"default": "¶"}),
                "use_separator": ("BOOLEAN", {"default": True}),

                # Sampling behavior
                "sample_latent": ("BOOLEAN", {"default": True}),

                # Lyra experimental toggles
                "use_hard_masking": ("BOOLEAN", {"default": True}),
            }
        }

    def configure(
            self,
            target_model: str,
            seed: int,
            output_clip_l: bool = True,
            output_clip_g: bool = True,
            output_t5_xl_l: bool = False,
            output_t5_xl_g: bool = False,
            summary_separator: str = "¶",
            use_separator: bool = True,
            sample_latent: bool = True,
            use_hard_masking: bool = True,
    ) -> Tuple[Dict]:

        # Determine output modalities based on target model
        if target_model == "sd15":
            output_modalities = ["clip_l"]
        elif target_model == "sdxl":
            output_modalities = ["clip_l", "clip_g"]
        elif target_model == "sdxl_with_t5":
            output_modalities = ["clip_l", "clip_g", "t5_xl_l"]
        elif target_model == "flux":
            output_modalities = ["clip_l", "t5_xl_l"]
        elif target_model == "custom":
            output_modalities = []
            if output_clip_l:
                output_modalities.append("clip_l")
            if output_clip_g:
                output_modalities.append("clip_g")
            if output_t5_xl_l:
                output_modalities.append("t5_xl_l")
            if output_t5_xl_g:
                output_modalities.append("t5_xl_g")
            if not output_modalities:
                output_modalities = ["clip_l"]  # Fallback
        else:
            output_modalities = ["clip_l", "clip_g"]

        config = {
            "target_model": target_model,
            "output_modalities": output_modalities,
            "seed": seed,
            "summary_separator": summary_separator,
            "use_separator": use_separator,
            "sample_latent": sample_latent,
            "use_hard_masking": use_hard_masking,
        }

        print(f"[LyraConfig] Target: {target_model}")
        print(f"[LyraConfig] Output modalities: {output_modalities}")
        print(f"[LyraConfig] Separator: '{summary_separator}' (active: {use_separator})")

        return (config,)


# ============================================================================
# LYRA ENCODE
# ============================================================================

class LyraEncode:
    """
    Encode inputs through Lyra VAE.

    Handles ComfyUI SDXL conditioning format:
    - Splits merged [B, 77, 2048] back to clip_l [768] + clip_g [1280]
    - Re-merges outputs back to SDXL format
    """

    CATEGORY = "Lyra/Process"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LYRA_LATENT", "LYRA_STATE")
    RETURN_NAMES = ("cond_positive", "cond_negative", "latent", "state")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyra_pipe": ("LYRA_PIPE",),
                "lyra_config": ("LYRA_CONFIG",),
            },
            "optional": {
                "encoder_pipe": ("ENCODER_PIPE",),
                "prompt_positive": ("STRING", {"default": "", "multiline": True}),
                "prompt_negative": ("STRING", {"default": "", "multiline": True}),
                # Pre-computed SDXL conds (merged format)
                "cond_positive": ("CONDITIONING",),
                "cond_negative": ("CONDITIONING",),
            }
        }

    def encode(
            self,
            lyra_pipe: Dict,
            lyra_config: Dict,
            encoder_pipe: Optional[Dict] = None,
            prompt_positive: str = "",
            prompt_negative: str = "",
            cond_positive=None,
            cond_negative=None,
    ) -> Tuple:

        model = lyra_pipe["model"]
        device = lyra_pipe["device"]
        output_modalities = lyra_config["output_modalities"]
        seed = lyra_config["seed"]
        sample_latent = lyra_config["sample_latent"]

        generator = torch.Generator(device=device).manual_seed(seed)

        # Extract positive inputs
        pos_inputs, pos_metadata = self._build_inputs(
            encoder_pipe=encoder_pipe,
            prompt=prompt_positive,
            sdxl_cond=cond_positive,
            lyra_config=lyra_config,
            device=device
        )

        # Extract negative inputs
        neg_inputs, neg_metadata = self._build_inputs(
            encoder_pipe=encoder_pipe,
            prompt=prompt_negative,
            sdxl_cond=cond_negative,
            lyra_config=lyra_config,
            device=device
        )

        if not pos_inputs:
            raise ValueError(
                "No positive inputs provided - need either cond_positive or prompt_positive with encoder_pipe")

        # Encode positive
        with torch.no_grad():
            pos_result = model(
                pos_inputs,
                target_modalities=output_modalities,
                generator=generator
            )

            if len(pos_result) == 3:
                pos_recons, pos_mu, pos_logvar = pos_result
                pos_per_mod = None
            else:
                pos_recons, pos_mu, pos_logvar, pos_per_mod = pos_result

        # Encode negative
        with torch.no_grad():
            if neg_inputs:
                neg_result = model(
                    neg_inputs,
                    target_modalities=output_modalities,
                    generator=generator
                )
                if len(neg_result) == 3:
                    neg_recons, neg_mu, neg_logvar = neg_result
                else:
                    neg_recons, neg_mu, neg_logvar, _ = neg_result
            else:
                neg_recons = {k: torch.zeros_like(v) for k, v in pos_recons.items()}
                neg_mu = torch.zeros_like(pos_mu)
                neg_logvar = torch.zeros_like(pos_logvar)

        # Build output conds (re-merged to SDXL format, on CPU)
        cond_pos_out = self._build_output_cond(pos_recons, output_modalities, pos_metadata)
        cond_neg_out = self._build_output_cond(neg_recons, output_modalities, neg_metadata)

        # Build latent output
        latent = {
            "mu": pos_mu.cpu(),
            "logvar": pos_logvar.cpu(),
            "sampled": sample_latent,
        }

        # Build state
        state = {
            "pos_recons": {k: v.cpu() for k, v in pos_recons.items()},
            "neg_recons": {k: v.cpu() for k, v in neg_recons.items()},
            "pos_mu": pos_mu.cpu(),
            "pos_logvar": pos_logvar.cpu(),
            "neg_mu": neg_mu.cpu(),
            "neg_logvar": neg_logvar.cpu(),
            "pos_metadata": pos_metadata,
            "neg_metadata": neg_metadata,
            "output_modalities": output_modalities,
            "config": lyra_config,
        }

        return (cond_pos_out, cond_neg_out, latent, state)

    def _build_inputs(
            self,
            encoder_pipe: Optional[Dict],
            prompt: str,
            sdxl_cond,
            lyra_config: Dict,
            device: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Build Lyra input dict from SDXL conditioning or prompt.

        Returns:
            (inputs_dict, original_metadata)
        """
        inputs = {}
        metadata = {}

        # Priority 1: SDXL conditioning (split merged back to components)
        if sdxl_cond is not None:
            extracted = self._extract_sdxl_embeddings(sdxl_cond)

            if "clip_l" in extracted:
                inputs["clip_l"] = extracted["clip_l"].to(device)
            if "clip_g" in extracted:
                inputs["clip_g"] = extracted["clip_g"].to(device)
            if "pooled_output" in extracted:
                metadata["pooled_output"] = extracted["pooled_output"]

            # Preserve other metadata
            if isinstance(sdxl_cond, list) and len(sdxl_cond) > 0 and len(sdxl_cond[0]) > 1:
                for k, v in sdxl_cond[0][1].items():
                    if k not in metadata:
                        metadata[k] = v

        # Priority 2: Encode T5 from prompt
        if prompt.strip() and encoder_pipe is not None:
            t5_embed = self._encode_t5(encoder_pipe, prompt, device)
            inputs["t5_xl_l"] = t5_embed
            inputs["t5_xl_g"] = t5_embed

        return inputs, metadata

    def _extract_sdxl_embeddings(self, conditioning) -> Dict[str, torch.Tensor]:
        """
        Extract CLIP-L and CLIP-G from ComfyUI SDXL conditioning.

        ComfyUI SDXL format:
            [[merged_embed, {"pooled_output": pooled, ...}], ...]

        Where merged_embed is [B, 77, 2048] = clip_l (768) + clip_g (1280)
        """
        if isinstance(conditioning, list) and len(conditioning) > 0:
            merged = conditioning[0][0]  # [B, 77, 2048]
            metadata = conditioning[0][1] if len(conditioning[0]) > 1 else {}
        elif isinstance(conditioning, torch.Tensor):
            merged = conditioning
            metadata = {}
        else:
            raise ValueError(f"Unknown conditioning format: {type(conditioning)}")

        result = {}

        # Detect format based on last dimension
        last_dim = merged.shape[-1]

        if last_dim == 2048:
            # SDXL merged format: split back
            result["clip_l"] = merged[..., :768]  # [B, 77, 768]
            result["clip_g"] = merged[..., 768:]  # [B, 77, 1280]
        elif last_dim == 768:
            # SD1.5 format: just clip_l
            result["clip_l"] = merged
        elif last_dim == 1280:
            # Just clip_g somehow
            result["clip_g"] = merged
        else:
            # Unknown format - pass through as clip_l
            print(f"[LyraEncode] Warning: unexpected embedding dim {last_dim}, treating as clip_l")
            result["clip_l"] = merged

        # Preserve pooled output
        if "pooled_output" in metadata:
            result["pooled_output"] = metadata["pooled_output"]

        return result

    def _encode_t5(
            self,
            encoder_pipe: Dict,
            prompt: str,
            device: str
    ) -> torch.Tensor:
        """Encode prompt with T5/Qwen/Llama encoder."""
        model = encoder_pipe["model"]
        tokenizer = encoder_pipe["tokenizer"]
        encoder_type = encoder_pipe["encoder_type"]
        encoder_device = encoder_pipe["device"]

        tokens = tokenizer(
            prompt,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(encoder_device)

        with torch.no_grad():
            if encoder_type == "t5":
                outputs = model(**tokens)
                embedding = outputs.last_hidden_state
            else:
                outputs = model(**tokens, output_hidden_states=True)
                embedding = outputs.hidden_states[-1]

        return embedding.to(device)

    def _build_output_cond(
            self,
            recons: Dict[str, torch.Tensor],
            modalities: List[str],
            original_metadata: Optional[Dict] = None
    ) -> List:
        """
        Build ComfyUI SDXL conditioning from Lyra reconstructions.

        Re-merges clip_l + clip_g back to [B, 77, 2048] format.
        """
        clip_l = recons.get("clip_l")
        clip_g = recons.get("clip_g")

        if clip_l is not None and clip_g is not None:
            # Re-merge for SDXL: [B, 77, 768] + [B, 77, 1280] -> [B, 77, 2048]
            merged = torch.cat([clip_l.cpu(), clip_g.cpu()], dim=-1)
        elif clip_l is not None:
            merged = clip_l.cpu()
        elif clip_g is not None:
            merged = clip_g.cpu()
        else:
            return []

        # Build metadata
        metadata = {}

        # Preserve original metadata
        if original_metadata:
            metadata.update(original_metadata)

        # Ensure pooled_output exists (required for SDXL)
        if "pooled_output" not in metadata and clip_g is not None:
            # Use first token of clip_g as pooled (approximation)
            metadata["pooled_output"] = clip_g[:, 0, :].cpu()

        # Store Lyra info
        metadata["lyra_modalities"] = modalities
        metadata["lyra_recons"] = {k: v.cpu() for k, v in recons.items()}

        return [(merged, metadata)]


# ============================================================================
# LYRA ENCODE SUMMARY (FIXED)
# ============================================================================

class LyraEncodeSummary:
    """
    Ease-of-use Lyra encode with primary prompt + summary support.

    - CLIP sees: tags only (via upstream SDXL conds)
    - T5 sees: tags + separator + summary
    """

    CATEGORY = "Lyra/Process"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LYRA_LATENT", "LYRA_STATE")
    RETURN_NAMES = ("cond_positive", "cond_negative", "latent", "state")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyra_pipe": ("LYRA_PIPE",),
                "lyra_config": ("LYRA_CONFIG",),
                "encoder_pipe": ("ENCODER_PIPE",),
            },
            "optional": {
                # Tag prompt (for T5, and description of what CLIP should see)
                "prompt_tags": ("STRING", {
                    "default": "masterpiece, 1girl, blue hair, school uniform",
                    "multiline": True
                }),
                # Summary (only T5 sees this, after separator)
                "prompt_summary": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "prompt_negative": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                # Pre-computed SDXL conds (merged format) - these supersede tag encoding for CLIP
                "cond_positive": ("CONDITIONING",),
                "cond_negative": ("CONDITIONING",),
            }
        }

    def encode(
            self,
            lyra_pipe: Dict,
            lyra_config: Dict,
            encoder_pipe: Dict,
            prompt_tags: str = "",
            prompt_summary: str = "",
            prompt_negative: str = "",
            cond_positive=None,
            cond_negative=None,
    ) -> Tuple:

        model = lyra_pipe["model"]
        device = lyra_pipe["device"]
        output_modalities = lyra_config["output_modalities"]
        seed = lyra_config["seed"]
        separator = lyra_config["summary_separator"]
        use_separator = lyra_config["use_separator"]
        sample_latent = lyra_config["sample_latent"]

        generator = torch.Generator(device=device).manual_seed(seed)

        # Build T5 input: tags + separator + summary
        if prompt_summary.strip() and use_separator:
            t5_prompt_pos = f"{prompt_tags} {separator} {prompt_summary}"
        elif prompt_summary.strip():
            t5_prompt_pos = f"{prompt_tags} {prompt_summary}"
        else:
            t5_prompt_pos = prompt_tags

        print(f"[LyraEncodeSummary] T5 input: {t5_prompt_pos[:80]}...")

        # Build positive inputs
        pos_inputs = {}
        pos_metadata = {}

        # CLIP from upstream SDXL conds (if provided)
        if cond_positive is not None:
            extracted = self._extract_sdxl_embeddings(cond_positive)
            if "clip_l" in extracted:
                pos_inputs["clip_l"] = extracted["clip_l"].to(device)
            if "clip_g" in extracted:
                pos_inputs["clip_g"] = extracted["clip_g"].to(device)
            if "pooled_output" in extracted:
                pos_metadata["pooled_output"] = extracted["pooled_output"]
            # Preserve other metadata
            if isinstance(cond_positive, list) and len(cond_positive) > 0 and len(cond_positive[0]) > 1:
                for k, v in cond_positive[0][1].items():
                    if k not in pos_metadata:
                        pos_metadata[k] = v

        # T5 from prompt + summary
        if t5_prompt_pos.strip():
            t5_pos = self._encode_t5(encoder_pipe, t5_prompt_pos, device)
            pos_inputs["t5_xl_l"] = t5_pos
            pos_inputs["t5_xl_g"] = t5_pos

        if not pos_inputs:
            raise ValueError("No positive inputs - need cond_positive or prompt_tags with encoder")

        # Build negative inputs
        neg_inputs = {}
        neg_metadata = {}

        if cond_negative is not None:
            extracted = self._extract_sdxl_embeddings(cond_negative)
            if "clip_l" in extracted:
                neg_inputs["clip_l"] = extracted["clip_l"].to(device)
            if "clip_g" in extracted:
                neg_inputs["clip_g"] = extracted["clip_g"].to(device)
            if "pooled_output" in extracted:
                neg_metadata["pooled_output"] = extracted["pooled_output"]

        if prompt_negative.strip():
            t5_neg = self._encode_t5(encoder_pipe, prompt_negative, device)
            neg_inputs["t5_xl_l"] = t5_neg
            neg_inputs["t5_xl_g"] = t5_neg

        # Encode through Lyra
        with torch.no_grad():
            pos_result = model(
                pos_inputs,
                target_modalities=output_modalities,
                generator=generator
            )

            if len(pos_result) == 3:
                pos_recons, pos_mu, pos_logvar = pos_result
            else:
                pos_recons, pos_mu, pos_logvar, _ = pos_result

        # Negative pass
        with torch.no_grad():
            if neg_inputs:
                neg_result = model(
                    neg_inputs,
                    target_modalities=output_modalities,
                    generator=generator
                )
                if len(neg_result) == 3:
                    neg_recons, neg_mu, neg_logvar = neg_result
                else:
                    neg_recons, neg_mu, neg_logvar, _ = neg_result
            else:
                neg_recons = {k: torch.zeros_like(v) for k, v in pos_recons.items()}
                neg_mu = torch.zeros_like(pos_mu)
                neg_logvar = torch.zeros_like(pos_logvar)

        # Build outputs (re-merged SDXL format, CPU)
        cond_pos_out = self._build_output_cond(pos_recons, output_modalities, pos_metadata)
        cond_neg_out = self._build_output_cond(neg_recons, output_modalities, neg_metadata)

        latent = {
            "mu": pos_mu.cpu(),
            "logvar": pos_logvar.cpu(),
            "sampled": sample_latent,
        }

        state = {
            "pos_recons": {k: v.cpu() for k, v in pos_recons.items()},
            "neg_recons": {k: v.cpu() for k, v in neg_recons.items()},
            "pos_mu": pos_mu.cpu(),
            "neg_mu": neg_mu.cpu(),
            "output_modalities": output_modalities,
            "t5_prompt_pos": t5_prompt_pos,
            "t5_prompt_neg": prompt_negative,
            "tags_prompt": prompt_tags,
            "summary_prompt": prompt_summary,
            "pos_metadata": pos_metadata,
            "neg_metadata": neg_metadata,
        }

        return (cond_pos_out, cond_neg_out, latent, state)

    def _extract_sdxl_embeddings(self, conditioning) -> Dict[str, torch.Tensor]:
        """Extract CLIP-L and CLIP-G from ComfyUI SDXL merged conditioning."""
        if isinstance(conditioning, list) and len(conditioning) > 0:
            merged = conditioning[0][0]
            metadata = conditioning[0][1] if len(conditioning[0]) > 1 else {}
        elif isinstance(conditioning, torch.Tensor):
            merged = conditioning
            metadata = {}
        else:
            raise ValueError(f"Unknown conditioning format: {type(conditioning)}")

        result = {}
        last_dim = merged.shape[-1]

        if last_dim == 2048:
            result["clip_l"] = merged[..., :768]
            result["clip_g"] = merged[..., 768:]
        elif last_dim == 768:
            result["clip_l"] = merged
        elif last_dim == 1280:
            result["clip_g"] = merged
        else:
            result["clip_l"] = merged

        if "pooled_output" in metadata:
            result["pooled_output"] = metadata["pooled_output"]

        return result

    def _encode_t5(self, encoder_pipe: Dict, prompt: str, device: str) -> torch.Tensor:
        """Encode prompt with T5/Qwen/Llama encoder."""
        model = encoder_pipe["model"]
        tokenizer = encoder_pipe["tokenizer"]
        encoder_type = encoder_pipe["encoder_type"]
        encoder_device = encoder_pipe["device"]

        tokens = tokenizer(
            prompt,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(encoder_device)

        with torch.no_grad():
            if encoder_type == "t5":
                outputs = model(**tokens)
                embedding = outputs.last_hidden_state
            else:
                outputs = model(**tokens, output_hidden_states=True)
                embedding = outputs.hidden_states[-1]

        return embedding.to(device)

    def _build_output_cond(
            self,
            recons: Dict[str, torch.Tensor],
            modalities: List[str],
            original_metadata: Optional[Dict] = None
    ) -> List:
        """Build ComfyUI SDXL conditioning from Lyra reconstructions."""
        clip_l = recons.get("clip_l")
        clip_g = recons.get("clip_g")

        if clip_l is not None and clip_g is not None:
            merged = torch.cat([clip_l.cpu(), clip_g.cpu()], dim=-1)
        elif clip_l is not None:
            merged = clip_l.cpu()
        elif clip_g is not None:
            merged = clip_g.cpu()
        else:
            return []

        metadata = {}
        if original_metadata:
            metadata.update(original_metadata)

        if "pooled_output" not in metadata and clip_g is not None:
            metadata["pooled_output"] = clip_g[:, 0, :].cpu()

        metadata["lyra_modalities"] = modalities
        metadata["lyra_recons"] = {k: v.cpu() for k, v in recons.items()}

        return [(merged, metadata)]

# ============================================================================
# LYRA DECODE
# ============================================================================

class LyraDecode:
    """
    Decode Lyra latent/state back to conditioning.

    Takes LYRA_STATE from encode and produces final conds.
    """

    CATEGORY = "Lyra/Process"
    FUNCTION = "decode"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("cond_positive", "cond_negative")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyra_pipe": ("LYRA_PIPE",),
                "state": ("LYRA_STATE",),
            },
            "optional": {
                "latent": ("LYRA_LATENT",),
                "resample": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    def decode(
            self,
            lyra_pipe: Dict,
            state: Dict,
            latent: Optional[Dict] = None,
            resample: bool = False,
            seed: int = 42
    ) -> Tuple:

        model = lyra_pipe["model"]
        device = lyra_pipe["device"]
        output_modalities = state["output_modalities"]

        if resample and latent is not None:
            # Resample from latent
            mu = latent["mu"].to(device)
            logvar = latent["logvar"].to(device)

            generator = torch.Generator(device=device).manual_seed(seed)

            with torch.no_grad():
                z = model.reparameterize(mu, logvar, generator=generator)
                pos_recons = model.decode(z, target_modalities=output_modalities)

                # For negative, use stored or zeros
                if "neg_mu" in state:
                    neg_mu = state["neg_mu"].to(device)
                    neg_logvar = state.get("neg_logvar", torch.zeros_like(neg_mu)).to(device)
                    z_neg = model.reparameterize(neg_mu, neg_logvar, generator=generator)
                    neg_recons = model.decode(z_neg, target_modalities=output_modalities)
                else:
                    neg_recons = {k: torch.zeros_like(v) for k, v in pos_recons.items()}
        else:
            # Use stored reconstructions
            pos_recons = {k: v.to(device) for k, v in state["pos_recons"].items()}
            neg_recons = {k: v.to(device) for k, v in state["neg_recons"].items()}

        # Build output conds (CPU)
        cond_pos = self._build_cond(pos_recons, output_modalities)
        cond_neg = self._build_cond(neg_recons, output_modalities)

        return (cond_pos, cond_neg)

    def _build_cond(
            self,
            recons: Dict[str, torch.Tensor],
            modalities: List[str]
    ) -> List:
        tensors = [recons[m].cpu() for m in modalities if m in recons]
        if not tensors:
            return []

        primary = tensors[0]
        metadata = {
            "lyra_modalities": modalities,
            "lyra_all_recons": {k: v.cpu() for k, v in recons.items()},
        }

        return [(primary, metadata)]


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LyraEncoderLoader": LyraEncoderLoader,
    "LyraLoader": LyraLoader,
    "LyraEncodeConfiguration": LyraEncodeConfiguration,
    "LyraEncode": LyraEncode,
    "LyraEncodeSummary": LyraEncodeSummary,
    "LyraDecode": LyraDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LyraEncoderLoader": "Lyra Encoder Loader (T5/Qwen/Llama)",
    "LyraLoader": "Lyra Loader",
    "LyraEncodeConfiguration": "Lyra Encode Config",
    "LyraEncode": "Lyra Encode",
    "LyraEncodeSummary": "Lyra Encode (Summary)",
    "LyraDecode": "Lyra Decode",
}