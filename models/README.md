# Local model layout

Model weights are intentionally not included in the repository. Place trusted
local model files under this directory, or override the corresponding relative
path in `generate.py`.

Expected default layout:

```text
models/
├── flux1-dev/
│   ├── flux1-dev.safetensors
│   ├── ae.safetensors
│   ├── text_encoder/
│   ├── text_encoder_2/
│   ├── tokenizer/
│   └── tokenizer_2/
├── sd3.5/
│   ├── sd3.5_large.safetensors
│   ├── clip_g.safetensors
│   ├── clip_l.safetensors
│   ├── t5xxl.safetensors
│   ├── tokenizer/
│   └── tokenizer_3/
├── pixart/
│   ├── PixArt-XL-2-256x256.pth
│   ├── sd-vae-ft-ema/
│   └── t5-v1_1-xxl/
├── sdxl/
└── sd1.5/
```

The SDXL and SD1.5 directories use the standard Diffusers directory layout.
Prefer `safetensors` weights. Only load checkpoints and legacy feature files
from sources you trust.
