"""FastAPI serving application for neural video matting."""

import io
import logging
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from src.serving.composite import composite_video
from src.serving.inference import MattingInference, load_video_frames

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Neural Video Matting API",
    description="Video and image matting with recurrent neural networks.",
    version="1.0.0",
)

# Global inference engine (initialized on startup)
_engine: Optional[MattingInference] = None
_start_time: float = 0
_request_count: int = 0


@app.on_event("startup")
def startup():
    global _engine, _start_time
    _start_time = time.time()

    config_path = os.environ.get("CONFIG_PATH", "configs/inference_config.yaml")
    if Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    checkpoint = config.get("checkpoint", "checkpoints/final.pth")
    device = config.get("device", "cuda")
    output_size = config.get("output_size", None)
    if output_size is not None:
        output_size = tuple(output_size)

    _engine = MattingInference(
        checkpoint_path=checkpoint,
        device=device,
        output_size=output_size,
    )
    logger.info("Matting inference engine initialized.")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _engine is not None,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.get("/metrics")
def metrics():
    """Basic metrics endpoint."""
    return {
        "uptime_seconds": round(time.time() - _start_time, 1),
        "total_requests": _request_count,
    }


@app.post("/matte_video")
async def matte_video(
    video: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    """Process a video with a guidance mask and return alpha matte PNGs as ZIP.

    Args:
        video: Input video file.
        mask: Guidance mask video/image. If a single image, it is used for all frames.

    Returns:
        ZIP file containing alpha matte PNG sequence.
    """
    global _request_count
    _request_count += 1

    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded files
        video_path = os.path.join(tmpdir, "input_video" + Path(video.filename).suffix)
        mask_path = os.path.join(tmpdir, "input_mask" + Path(mask.filename).suffix)

        with open(video_path, "wb") as f:
            f.write(await video.read())
        with open(mask_path, "wb") as f:
            f.write(await mask.read())

        # Load frames
        frames = load_video_frames(video_path)
        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="Could not read video frames.")

        # Load mask(s)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            # Try as video
            mask_frames_raw = load_video_frames(mask_path)
            if len(mask_frames_raw) == 0:
                raise HTTPException(status_code=400, detail="Could not read mask.")
            masks = [cv2.cvtColor(m, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                     for m in mask_frames_raw]
            # Pad if shorter than video
            while len(masks) < len(frames):
                masks.append(masks[-1])
        else:
            masks = [mask_img.astype(np.float32) / 255.0] * len(frames)

        # Run inference
        alphas = _engine.process_video(frames, masks)

        # Package as ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, alpha in enumerate(alphas):
                alpha_uint8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
                _, png_data = cv2.imencode(".png", alpha_uint8)
                zf.writestr(f"alpha_{i:06d}.png", png_data.tobytes())

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=alpha_mattes.zip"},
        )


@app.post("/matte_frame")
async def matte_frame(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    """Process a single image with a guidance mask and return the alpha matte as PNG.

    Args:
        image: Input RGB image.
        mask: Guidance mask image.

    Returns:
        PNG alpha matte image.
    """
    global _request_count
    _request_count += 1

    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Read image
    img_bytes = await image.read()
    img_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Read mask
    mask_bytes = await mask.read()
    mask_arr = np.frombuffer(mask_bytes, np.uint8)
    mask_img = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise HTTPException(status_code=400, detail="Could not decode mask.")
    mask_float = mask_img.astype(np.float32) / 255.0

    # Inference
    alpha = _engine.process_single_image(img, mask_float)

    # Encode to PNG
    alpha_uint8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    _, png_data = cv2.imencode(".png", alpha_uint8)

    return Response(content=png_data.tobytes(), media_type="image/png")


@app.post("/composite")
async def composite_endpoint(
    fg_video: UploadFile = File(...),
    alpha_zip: UploadFile = File(...),
    bg_video: UploadFile = File(...),
):
    """Composite foreground video with alpha mattes onto a new background.

    Args:
        fg_video: Foreground video file.
        alpha_zip: ZIP of alpha matte PNGs.
        bg_video: New background video or image.

    Returns:
        ZIP of composited frames as PNGs.
    """
    global _request_count
    _request_count += 1

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploads
        fg_path = os.path.join(tmpdir, "fg" + Path(fg_video.filename).suffix)
        alpha_zip_path = os.path.join(tmpdir, "alphas.zip")
        bg_path = os.path.join(tmpdir, "bg" + Path(bg_video.filename).suffix)

        with open(fg_path, "wb") as f:
            f.write(await fg_video.read())
        with open(alpha_zip_path, "wb") as f:
            f.write(await alpha_zip.read())
        with open(bg_path, "wb") as f:
            f.write(await bg_video.read())

        # Load foreground frames
        fg_frames_bgr = load_video_frames(fg_path)

        # Note: load_video_frames returns RGB, composite expects uint8

        # Load alpha mattes from ZIP
        alphas = []
        alpha_dir = os.path.join(tmpdir, "alphas")
        os.makedirs(alpha_dir, exist_ok=True)
        with zipfile.ZipFile(alpha_zip_path, "r") as zf:
            zf.extractall(alpha_dir)
        alpha_files = sorted(Path(alpha_dir).glob("*.png"))
        for af in alpha_files:
            a = cv2.imread(str(af), cv2.IMREAD_GRAYSCALE)
            alphas.append(a.astype(np.float32) / 255.0)

        # Load background
        bg_img = cv2.imread(bg_path)
        if bg_img is not None:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_frames = [bg_img] * len(fg_frames_bgr)
        else:
            bg_frames = load_video_frames(bg_path)

        if len(fg_frames_bgr) == 0 or len(alphas) == 0 or len(bg_frames) == 0:
            raise HTTPException(status_code=400, detail="Could not load inputs.")

        # Composite
        results = composite_video(
            fg_frames=fg_frames_bgr,
            alphas=alphas,
            bg_frames=bg_frames,
        )

        # Package as ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, frame in enumerate(results):
                if frame.dtype == np.float32:
                    frame = (frame * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, png_data = cv2.imencode(".png", frame_bgr)
                zf.writestr(f"composite_{i:06d}.png", png_data.tobytes())

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=composited.zip"},
        )
