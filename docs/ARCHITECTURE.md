# Architecture Overview

This document describes the architectural design of the neural video matting system, the rationale behind the recurrent temporal approach, and how each component fits into the full pipeline.

## System Overview

The system extracts high-quality alpha mattes from video without a green screen. Given a video and a rough first-frame mask, it produces per-frame alpha mattes with temporal consistency suitable for production compositing.

```
                    Input Video (T frames)
                           |
                    +------v------+
                    |   Encoder   |  ResNet-34 from scratch
                    | (per-frame) |  4 input channels: RGB + guidance
                    +------+------+
                           |
                    [f1, f2, f3, f4]  Multi-scale features
                    H/4  H/8  H/16  H/32
                           |
                    +------v------+
                    |   ConvGRU   |  Temporal recurrence at H/4
                    | (frame-by-  |  Hidden state carries edge memory
                    |   frame)    |
                    +------+------+
                           |
                    +------v------+
                    |   Decoder   |  Top-down FPN with skip connections
                    |             |  Produces alpha + optional foreground
                    +------+------+
                           |
                    +------v------+
                    |   Refiner   |  Optional patch-based detail recovery
                    | (full res)  |  Hair strands, semi-transparent edges
                    +------+------+
                           |
                    Alpha Matte (H, W) per frame
```

## Component Details

### Encoder (`src/model/encoder.py`)

A ResNet-34 style backbone built from scratch (no pretrained weights). Takes 4-channel input: 3 RGB channels concatenated with 1 guidance channel (rough mask or trimap).

**Output**: Feature maps at 4 scales:
- `f1`: (B, 64, H/4, W/4) -- finest features, fed to ConvGRU
- `f2`: (B, 128, H/8, W/8) -- skip connection to decoder
- `f3`: (B, 256, H/16, W/16) -- skip connection to decoder
- `f4`: (B, 512, H/32, W/32) -- coarsest features, top of decoder

The stem uses a 7x7 convolution with stride 2 followed by max pooling (stride 2), matching standard ResNet design. Each subsequent stage uses BasicBlocks with residual connections. Block counts follow ResNet-34: [3, 4, 6, 3].

### ConvGRU Temporal Module (`src/model/recurrent.py`)

A Convolutional Gated Recurrent Unit that replaces fully-connected GRU gates with convolutional gates to preserve spatial structure. Operates at H/4 resolution with 64 channels.

**GRU equations (convolutional form)**:
```
z_t = sigmoid(Conv([x_t, h_{t-1}]))     -- update gate
r_t = sigmoid(Conv([x_t, h_{t-1}]))     -- reset gate
h_t = (1 - z_t) * h_{t-1} + z_t * tanh(Conv([x_t, r_t * h_{t-1}]))
```

The hidden state `h` carries temporal context across frames: edge memory, motion boundaries, and matte consistency information. For the first frame, `h` is initialized to zeros.

**Why H/4 resolution**: Operating at full resolution would require 16x more memory per hidden state. H/4 provides a good balance between spatial detail and memory efficiency for typical 720p-1080p inputs.

### Decoder (`src/model/decoder.py`)

A top-down feature pyramid decoder that fuses multi-scale encoder features with the temporally-enriched ConvGRU output.

**Path**: f4 (H/32) -> upsample + skip f3 -> upsample + skip f2 -> upsample + skip ConvGRU output -> 4x upsample -> alpha head (sigmoid)

The decoder also optionally predicts foreground RGB through a separate head, which is used in the alpha-weighted foreground L1 loss during training.

### Refiner (`src/model/refiner.py`)

A lightweight patch-based network for recovering fine alpha details (hair strands, fur, semi-transparent edges) at full resolution. The main decoder operates on downsampled features and can miss these high-frequency details.

At inference, the refiner processes overlapping patches with weighted blending. At training, random patches are sampled from the "unknown" alpha region.

### Loss Functions (`src/training/losses.py`)

The training objective combines five loss terms:

1. **Alpha L1**: Direct pixel-wise L1 between predicted and GT alpha
2. **Laplacian Pyramid L1**: Multi-scale frequency-domain loss that balances sharp edges and smooth regions
3. **Gradient Loss**: Sobel-based gradient matching to preserve edge sharpness
4. **Temporal Consistency**: Penalizes alpha changes between frames in regions where the input frames are similar (low motion)
5. **Foreground L1**: Alpha-weighted L1 loss on predicted foreground RGB

## Why Recurrent Temporal Design

### The Flickering Problem

Per-frame matting methods process each frame independently. At soft boundaries (hair, transparent materials, motion blur), the matting prediction is inherently ambiguous. Different frames resolve this ambiguity differently, causing visible jitter/flicker when the alpha matte sequence is used for compositing.

### How ConvGRU Solves It

The ConvGRU hidden state acts as temporal memory. When the model encounters an ambiguous soft boundary, it can refer to how it resolved the same region in previous frames. This produces temporally coherent decisions even at uncertain boundaries.

The update gate `z_t` controls how much of the previous state to retain vs. how much to update from the current frame. In practice:
- **Static regions**: The gate learns to retain previous hidden state (high temporal consistency)
- **Moving regions**: The gate learns to update from current features (track motion)
- **Newly appearing regions**: With no prior context, the gate allows fresh features through

### Alternatives Considered

**Optical flow warping**: Warp previous frame's alpha to current frame using estimated flow. Problem: flow estimation errors accumulate and create artifacts at occlusion boundaries -- exactly where matting is hardest.

**3D convolutions**: Process short temporal windows (e.g., 5 frames) with 3D kernels. Problem: fixed temporal receptive field, high memory cost, cannot handle arbitrary-length sequences.

**Attention-based temporal**: Cross-frame attention (as in Transformer architectures). Better at handling large motions and occlusions, but significantly more parameters and memory. This is noted as a future direction.

**Memory bank approach**: Store reference mattes in a memory bank (as in MatAnyone, CVPR 2025). Better for very long videos where fixed-size hidden state degrades. Also noted as a future direction.

## Training Strategy

### Truncated Backpropagation Through Time (TBPTT)

The model processes video clips of length T (default: 8 frames) with full backpropagation through the recurrent chain. Gradients flow from the loss at frame T back through all T ConvGRU steps, allowing the model to learn useful temporal representations.

Clip length is limited by GPU memory -- each frame's features and gradients must be stored. The default T=8 balances temporal learning with memory constraints.

### Guidance Mask Generation

During training, ground-truth alpha mattes are degraded into rough masks via random morphological operations (erosion + dilation). This simulates the imprecise masks a user would provide at inference time (e.g., from SAM, manual selection, or a previous frame's prediction).

## Inference Pipeline

### Sequential Processing

At inference, frames are processed one at a time with explicit hidden state management:

```python
h = None
for frame, mask in zip(frames, masks):
    alpha, fg, h = model.forward_single(frame, mask, h)
    save(alpha)
```

This allows processing arbitrarily long videos with constant memory (only the current frame's features + the fixed-size hidden state need to be in memory).

### Serving Architecture

The FastAPI serving layer (`src/serving/app.py`) exposes three endpoints:
- `/matte_frame`: Single image matting (stateless)
- `/matte_video`: Full video matting with recurrent state
- `/composite`: Alpha compositing onto new backgrounds with spill suppression

Video uploads are validated for format and frame count limits before processing.

## Data Pipeline

### VideoMatte240K Format

Training data follows the VideoMatte240K convention:
- `fgr/`: Foreground RGB frames organized by clip
- `pha/`: Alpha matte frames (ground truth)
- `bgr_video/`: Background video clips for random compositing
- `bgr_image/`: Background images (alternative to video backgrounds)

Training composites are generated on-the-fly: `composite = fg * alpha + bg * (1 - alpha)`, giving the model diverse background contexts.

### Augmentation

Spatiotemporal augmentations are applied consistently across all T frames in a clip:
- Random spatial crop (same crop for all frames)
- Random horizontal flip (same decision for all frames)
- Random affine transform (same matrix for all frames)

This consistency is critical -- augmenting frames independently would break temporal coherence in the training signal.
