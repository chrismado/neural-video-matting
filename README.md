# neural-video-matting

Temporal video matting — extract high-quality alpha mattes from video without a green screen. Given a video and a rough first-frame mask, produces per-frame alpha mattes with temporal consistency for production compositing.

Built from scratch in PyTorch — no pretrained backbone, no external matting libraries. Recurrent temporal propagation via ConvGRU for stable, flicker-free mattes.

## Architecture

```
Frame t + Mask ──→ Encoder (ResNet-34) ──→ Multi-scale Features
                                                    │
                                            f1 (H/4, 64ch)
                                                    │
                                               ConvGRU ←── Hidden State (from frame t-1)
                                                    │
                                            Temporal Features
                                                    │
                                     Decoder ←── Skip Connections (f2, f3, f4)
                                         │
                                    Coarse Alpha (H, W)
                                         │
                                  Optional: Refiner ──→ Fine Alpha (hair/edges)
```

**Encoder**: ResNet-34 style backbone (from scratch) producing features at 4 scales: H/4, H/8, H/16, H/32. Input is RGB + guidance channel (rough mask or trimap).

**ConvGRU**: Convolutional Gated Recurrent Unit operating at H/4 resolution. Propagates temporal context between frames — edge memory, motion boundaries, matte consistency. This is what makes it *video* matting, not just per-frame matting.

**Decoder**: Top-down feature pyramid with skip connections from the encoder. Produces coarse alpha matte + optional foreground RGB prediction.

**Refiner**: Lightweight patch-based network for recovering fine details (hair strands, semi-transparent edges) at full resolution.

## Why This Architecture

**Why recurrent**: Per-frame matting flickers. Adjacent frames produce inconsistent mattes at soft boundaries, causing visible jitter when composited. ConvGRU carries temporal context so the model "remembers" how it resolved ambiguous edges in previous frames.

**Why trimap-free**: Traditional matting requires a trimap (manually marked foreground/background/unknown). In production, you get a rough mask from SAM or a user click — not a precise trimap. This network operates in mask-guided mode: give it an approximate selection and it refines to a clean matte.

**Production connection**: I built a neural green screen removal system for a feature film, replacing traditional keying workflows. This project demonstrates the same architectural approach as a general-purpose video matting tool.

## Results

<!-- TODO: Add after training -->
<!-- [Side-by-side: per-frame matting (flickering) vs recurrent (stable)] -->
<!-- [Demo: extract matte → composite onto new background] -->
<!-- [SAD/MSE metrics on VideoMatte240K test set] -->

## Quick Start

### Training

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/train.py --config configs/train_config.yaml
```

### Inference

```bash
# Extract mattes from video
python scripts/matte_video.py --video input.mp4 --mask first_frame_mask.png --output mattes/

# Composite onto new background
python scripts/composite.py --fg input.mp4 --alpha mattes/ --bg beach.jpg --output composited.mp4
```

### Serving

```bash
python scripts/serve.py --checkpoint checkpoints/best.pt
docker compose up --build
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/matte_frame` | Image + mask → alpha matte PNG |
| `POST` | `/matte_video` | Video + mask → ZIP of alpha frames |
| `POST` | `/composite` | FG + alpha + BG → composited video |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |

## Project Structure

```
src/
├── model/
│   ├── encoder.py          # ResNet-34 multi-scale feature encoder
│   ├── decoder.py          # Top-down alpha decoder with skip connections
│   ├── recurrent.py        # ConvGRU temporal propagation
│   ├── refiner.py          # Patch-based detail refinement (hair/edges)
│   └── matting_network.py  # Full pipeline with sequential frame processing
├── data/
│   ├── dataset.py          # VideoMatte240K loader with composite generation
│   ├── augmentation.py     # Spatiotemporal augmentations (consistent across clips)
│   └── trimap_gen.py       # Trimap and rough mask generation from GT alpha
├── training/
│   ├── losses.py           # Alpha L1, Laplacian pyramid, gradient, temporal
│   ├── trainer.py          # TBPTT training with recurrent state management
│   └── evaluate.py         # SAD, MSE, gradient, connectivity metrics
└── serving/
    ├── app.py              # FastAPI matting + compositing endpoints
    ├── inference.py        # Sequential video processing with recurrent state
    └── composite.py        # FG × α + BG × (1-α) with spill suppression
```

## What I'd Do Differently at Scale

- **MatAnyone memory propagation**: CVPR 2025 approach — memory bank of reference mattes instead of fixed-size hidden state, handles longer videos without degradation
- **SAM 2 integration**: Automatic mask generation from a single click, eliminating manual first-frame selection
- **Attention-based temporal**: Replace ConvGRU with cross-frame attention (more parameters but better at handling large motions and occlusions)
- **Real-time optimization**: BiVM-style binarized networks for 30+ FPS inference on consumer hardware — required for on-set live preview
- **Multi-instance matting**: Handle multiple subjects simultaneously with instance-aware hidden states

## Stack

PyTorch · FastAPI · Docker · Weights & Biases · VideoMatte240K · CUDA
