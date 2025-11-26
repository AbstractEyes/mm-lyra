# ComfyUI-Lyra

VAE Lyra nodes for ComfyUI - Multi-modal VAE with Cantor fusion for SDXL text encoders.

## Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AbstractPhil/ComfyUI-Lyra.git
cd ComfyUI-Lyra
pip install -r requirements.txt
```

## Available Models

| Model | Version | Fusion | CLIP Weights |
|-------|---------|--------|--------------|
| `AbstractPhil/vae-lyra` | v1 | cantor | standard |
| `AbstractPhil/vae-lyra-sdxl-t5xl` | v1 | cantor | standard |
| `AbstractPhil/vae-lyra-xl-adaptive-cantor` | v2 | adaptive_cantor | standard |
| `AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious` | v2 | adaptive_cantor | illustrious |

## Nodes

### Lyra Loader
Load model from HuggingFace Hub or local path.

### Lyra Encode  
Encode CLIP-L, CLIP-G, T5-XL embeddings to latent space.

### Lyra Decode
Decode latent back to reconstructed embeddings.

### Lyra Full Pass
Complete encode-decode cycle.

### Lyra Info
Display model configuration and learned parameters.

## V1 vs V2

**V1**: Single encoder, single output projection. `forward()` returns 3 values.

**V2 (adaptive_cantor)**: Per-modality encoders with hard masking between binding groups. `forward()` returns 4 values.

## License

MIT