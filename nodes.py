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

class EncoderLoader:
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

    Accepts lyra_pipe, encoder_pipe, and optional pre-computed conds.
    Outputs reconstructed conds and latent state.
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
                # Pre-computed conds (supersede prompts if provided)
                "cond_positive": ("CONDITIONING",),
                "cond_negative": ("CONDITIONING",),
                # Individual modality inputs (for advanced use)
                "clip_l_cond": ("CONDITIONING",),
                "clip_g_cond": ("CONDITIONING",),
                "t5_cond": ("CONDITIONING",),
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
            clip_l_cond=None,
            clip_g_cond=None,
            t5_cond=None,
    ) -> Tuple:

        model = lyra_pipe["model"]
        device = lyra_pipe["device"]
        output_modalities = lyra_config["output_modalities"]
        seed = lyra_config["seed"]
        sample_latent = lyra_config["sample_latent"]

        # Set seed for reproducibility
        generator = torch.Generator(device=device).manual_seed(seed)

        # Build positive inputs
        pos_inputs = self._build_inputs(
            encoder_pipe=encoder_pipe,
            prompt=prompt_positive,
            precomputed_cond=cond_positive,
            clip_l_cond=clip_l_cond,
            clip_g_cond=clip_g_cond,
            t5_cond=t5_cond,
            lyra_config=lyra_config,
            device=device
        )

        # Build negative inputs
        neg_inputs = self._build_inputs(
            encoder_pipe=encoder_pipe,
            prompt=prompt_negative,
            precomputed_cond=cond_negative,
            clip_l_cond=None,  # Negatives don't use individual modality inputs
            clip_g_cond=None,
            t5_cond=None,
            lyra_config=lyra_config,
            device=device
        )

        # Encode positive
        with torch.no_grad():
            if pos_inputs:
                pos_result = model(
                    pos_inputs,
                    target_modalities=output_modalities,
                    generator=generator
                )
                # v1: 3 values, v2: 4 values
                if len(pos_result) == 3:
                    pos_recons, pos_mu, pos_logvar = pos_result
                    pos_per_mod = None
                else:
                    pos_recons, pos_mu, pos_logvar, pos_per_mod = pos_result
            else:
                raise ValueError("No positive inputs provided")

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
                # Empty negative
                neg_recons = {k: torch.zeros_like(v) for k, v in pos_recons.items()}
                neg_mu = torch.zeros_like(pos_mu)
                neg_logvar = torch.zeros_like(pos_logvar)

        # Build output conds (on CPU for transport)
        cond_pos_out = self._build_output_cond(pos_recons, output_modalities)
        cond_neg_out = self._build_output_cond(neg_recons, output_modalities)

        # Build latent output
        latent = {
            "mu": pos_mu.cpu(),
            "logvar": pos_logvar.cpu(),
            "sampled": sample_latent,
        }

        # Build state (for downstream use)
        state = {
            "pos_recons": {k: v.cpu() for k, v in pos_recons.items()},
            "neg_recons": {k: v.cpu() for k, v in neg_recons.items()},
            "pos_mu": pos_mu.cpu(),
            "pos_logvar": pos_logvar.cpu(),
            "neg_mu": neg_mu.cpu(),
            "neg_logvar": neg_logvar.cpu(),
            "output_modalities": output_modalities,
            "config": lyra_config,
        }

        return (cond_pos_out, cond_neg_out, latent, state)

    def _build_inputs(
            self,
            encoder_pipe: Optional[Dict],
            prompt: str,
            precomputed_cond,
            clip_l_cond,
            clip_g_cond,
            t5_cond,
            lyra_config: Dict,
            device: str
    ) -> Dict[str, torch.Tensor]:
        """Build input dict for Lyra from various sources."""
        inputs = {}

        # Priority 1: Individual modality conds
        if clip_l_cond is not None:
            inputs["clip_l"] = self._extract_embedding(clip_l_cond).to(device)
        if clip_g_cond is not None:
            inputs["clip_g"] = self._extract_embedding(clip_g_cond).to(device)
        if t5_cond is not None:
            emb = self._extract_embedding(t5_cond).to(device)
            inputs["t5_xl_l"] = emb
            inputs["t5_xl_g"] = emb

        # Priority 2: Precomputed cond (assumes SDXL-style pooled)
        if precomputed_cond is not None and not inputs:
            emb = self._extract_embedding(precomputed_cond).to(device)
            # Assume it's clip_l sized, could be smarter here
            inputs["clip_l"] = emb

        # Priority 3: Encode from prompt using encoder_pipe
        if not inputs and prompt.strip() and encoder_pipe is not None:
            inputs = self._encode_prompt(encoder_pipe, prompt, lyra_config, device)

        return inputs

    def _encode_prompt(
            self,
            encoder_pipe: Dict,
            prompt: str,
            lyra_config: Dict,
            device: str
    ) -> Dict[str, torch.Tensor]:
        """Encode prompt using the encoder_pipe."""
        model = encoder_pipe["model"]
        tokenizer = encoder_pipe["tokenizer"]
        encoder_type = encoder_pipe["encoder_type"]
        encoder_device = encoder_pipe["device"]

        # Tokenize
        tokens = tokenizer(
            prompt,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(encoder_device)

        # Encode
        with torch.no_grad():
            if encoder_type == "t5":
                outputs = model(**tokens)
                embedding = outputs.last_hidden_state
            else:
                # Qwen/Llama - use hidden states
                outputs = model(**tokens, output_hidden_states=True)
                embedding = outputs.hidden_states[-1]

        # Move to target device
        embedding = embedding.to(device)

        # For T5, we create both t5_xl_l and t5_xl_g
        return {
            "t5_xl_l": embedding,
            "t5_xl_g": embedding,
        }

    def _extract_embedding(self, conditioning) -> torch.Tensor:
        """Extract embedding tensor from ComfyUI conditioning format."""
        if isinstance(conditioning, torch.Tensor):
            return conditioning
        if isinstance(conditioning, list) and len(conditioning) > 0:
            return conditioning[0][0]
        if isinstance(conditioning, dict) and "samples" in conditioning:
            return conditioning["samples"]
        raise ValueError(f"Unknown conditioning format: {type(conditioning)}")

    def _build_output_cond(
            self,
            recons: Dict[str, torch.Tensor],
            modalities: List[str]
    ) -> List:
        """Build ComfyUI conditioning format from reconstructions."""
        # Concatenate requested modalities
        tensors = []
        for mod in modalities:
            if mod in recons:
                tensors.append(recons[mod].cpu())

        if not tensors:
            return []

        # Use first modality as primary (ComfyUI expects single tensor)
        primary = tensors[0]

        # Store all in metadata
        metadata = {
            "lyra_modalities": modalities,
            "lyra_all_recons": {k: v.cpu() for k, v in recons.items()},
        }

        return [(primary, metadata)]


# ============================================================================
# LYRA ENCODE SUMMARY
# ============================================================================

class LyraEncodeSummary:
    """
    Ease-of-use Lyra encode with primary prompt + summary support.

    Automatically concatenates summary with separator for T5 input,
    while CLIP only sees the primary prompt (tags).
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
                "prompt_tags": ("STRING", {
                    "default": "masterpiece, 1girl, blue hair, school uniform",
                    "multiline": True
                }),
            },
            "optional": {
                "prompt_summary": ("STRING", {
                    "default": "A cheerful schoolgirl with blue hair smiling warmly",
                    "multiline": True
                }),
                "prompt_negative": ("STRING", {
                    "default": "lowres, bad anatomy, bad hands",
                    "multiline": True
                }),
                # Optional pre-computed CLIP conds
                "clip_l_positive": ("CONDITIONING",),
                "clip_g_positive": ("CONDITIONING",),
                "clip_l_negative": ("CONDITIONING",),
                "clip_g_negative": ("CONDITIONING",),
            }
        }

    def encode(
            self,
            lyra_pipe: Dict,
            lyra_config: Dict,
            encoder_pipe: Dict,
            prompt_tags: str,
            prompt_summary: str = "",
            prompt_negative: str = "",
            clip_l_positive=None,
            clip_g_positive=None,
            clip_l_negative=None,
            clip_g_negative=None,
    ) -> Tuple:

        model = lyra_pipe["model"]
        device = lyra_pipe["device"]
        output_modalities = lyra_config["output_modalities"]
        seed = lyra_config["seed"]
        separator = lyra_config["summary_separator"]
        use_separator = lyra_config["use_separator"]
        sample_latent = lyra_config["sample_latent"]

        generator = torch.Generator(device=device).manual_seed(seed)

        # Build T5 input: tags + separator + summary (if enabled)
        if prompt_summary.strip() and use_separator:
            t5_prompt_pos = f"{prompt_tags} {separator} {prompt_summary}"
        elif prompt_summary.strip():
            t5_prompt_pos = f"{prompt_tags} {prompt_summary}"
        else:
            t5_prompt_pos = prompt_tags

        print(f"[LyraEncodeSummary] CLIP sees: {prompt_tags[:60]}...")
        print(f"[LyraEncodeSummary] T5 sees: {t5_prompt_pos[:80]}...")

        # Encode T5 inputs
        t5_pos = self._encode_t5(encoder_pipe, t5_prompt_pos, device)
        t5_neg = self._encode_t5(encoder_pipe, prompt_negative, device) if prompt_negative.strip() else None

        # Build positive inputs
        pos_inputs = {}

        # CLIP inputs (from upstream or leave empty for Lyra to handle)
        if clip_l_positive is not None:
            pos_inputs["clip_l"] = self._extract_embedding(clip_l_positive).to(device)
        if clip_g_positive is not None:
            pos_inputs["clip_g"] = self._extract_embedding(clip_g_positive).to(device)

        # T5 inputs
        pos_inputs["t5_xl_l"] = t5_pos
        pos_inputs["t5_xl_g"] = t5_pos

        # Build negative inputs
        neg_inputs = {}

        if clip_l_negative is not None:
            neg_inputs["clip_l"] = self._extract_embedding(clip_l_negative).to(device)
        if clip_g_negative is not None:
            neg_inputs["clip_g"] = self._extract_embedding(clip_g_negative).to(device)

        if t5_neg is not None:
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

        # Build outputs (CPU for transport)
        cond_pos = self._build_output_cond(pos_recons, output_modalities)
        cond_neg = self._build_output_cond(neg_recons, output_modalities)

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
        }

        return (cond_pos, cond_neg, latent, state)

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

    def _extract_embedding(self, conditioning) -> torch.Tensor:
        if isinstance(conditioning, torch.Tensor):
            return conditioning
        if isinstance(conditioning, list) and len(conditioning) > 0:
            return conditioning[0][0]
        raise ValueError(f"Unknown conditioning format: {type(conditioning)}")

    def _build_output_cond(
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
    "EncoderLoader": EncoderLoader,
    "LyraLoader": LyraLoader,
    "LyraEncodeConfiguration": LyraEncodeConfiguration,
    "LyraEncode": LyraEncode,
    "LyraEncodeSummary": LyraEncodeSummary,
    "LyraDecode": LyraDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EncoderLoader": "Encoder Loader (T5/Qwen/Llama)",
    "LyraLoader": "Lyra Loader",
    "LyraEncodeConfiguration": "Lyra Encode Config",
    "LyraEncode": "Lyra Encode",
    "LyraEncodeSummary": "Lyra Encode (Summary)",
    "LyraDecode": "Lyra Decode",
}