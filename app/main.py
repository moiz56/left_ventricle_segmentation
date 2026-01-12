import time
import torch
import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pathlib import Path
import tempfile
import nibabel as nib
from fastapi.concurrency import run_in_threadpool
from typing import List


from src.inference.inference import load_model
from src.data.preprocessing import pad_or_resize_to_224
from logs.logger import get_logger
from src.utils.load_yaml import get_yaml
from app.utils import save_slice_overlays

cfg = get_yaml()
model_name = cfg["inference_model"]["name"]
registry_dir = os.path.join(Path(__file__).resolve().parents[1], "registry")
save_dir = os.path.join(Path(__file__).resolve().parent, "inference_results")
os.makedirs(save_dir, exist_ok=True)

logger = get_logger(name="api", log_file="logs/api.log")

app = FastAPI(title="CMR Segmentation API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(model_name, registry_dir, device)
model.to(device)
model.eval()



@app.post("/predict/")
async def predict_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...), batch_size: int = 8):
    try:
        if not file.filename.endswith((".nii", ".nii.gz")):
            raise HTTPException(status_code=400, detail="Only .nii or .nii.gz files are supported")

        content = await file.read()

        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
            tmp.write(content)
            tmp.flush()

            # Load NIfTI volume (threadpool to avoid blocking)
            nii = await run_in_threadpool(nib.load, tmp.name)
            volume = await run_in_threadpool(lambda x: nib.as_closest_canonical(x).get_fdata(), nii)

            if volume.ndim != 3:
                raise HTTPException(status_code=400, detail="Expecting a 3D volume")

            start_time = time.time()

            predicted_slices = []
            original_slices = []

            # Prepare all slices first
            slices = []
            for idx in range(volume.shape[2]):
                slice_2d = volume[:, :, idx]
                slice_2d = pad_or_resize_to_224(slice_2d)
                slice_2d = np.clip(slice_2d, np.percentile(slice_2d, 0.5), np.percentile(slice_2d, 99.5))
                slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
                slices.append(slice_2d)
                original_slices.append(slice_2d.copy())

            # Batch inference
            num_slices = len(slices)
            for i in range(0, num_slices, batch_size):
                batch = slices[i:i+batch_size]
                tensor_batch = torch.from_numpy(np.stack(batch)).unsqueeze(1).to(device, dtype=torch.float32)  # [B,1,H,W]

                with torch.no_grad():
                    output = model(tensor_batch)  # [B, out_channels, H, W]
                    preds = torch.sigmoid(output)[:, 0, :, :]  # [B, H, W]

                    for pred_mask in preds:
                        pred_mask = (pred_mask > 0.5).float()
                        if pred_mask.mean() > 0.5:
                            pred_mask = 1 - pred_mask
                        predicted_slices.append(pred_mask.cpu().numpy())

            elapsed = time.time() - start_time
            logger.info(f"Inference completed for {num_slices} slices in {elapsed:.2f} seconds")

        # Save overlays in background
        background_tasks.add_task(save_slice_overlays, predicted_slices, original_slices, save_dir)

        return {
            "message": f"Inference completed for {num_slices} slices; saving overlays in background",
            "num_slices": num_slices,
            "inference_time_sec": elapsed
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.post("/batch_predict/")
async def predict_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    batch_size: int = 8
):
    all_slices = []           # To store all slices across files
    original_slices_map = {}  # Maps filename -> original slices
    slice_file_map = []       # Maps slice index to filename

    #Load all files and prepare slices
    for file in files:
        try:
            if not file.filename.endswith((".nii", ".nii.gz")):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not .nii or .nii.gz")

            content = await file.read()
            with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
                tmp.write(content)
                tmp.flush()

                nii = await run_in_threadpool(nib.load, tmp.name)
                volume = await run_in_threadpool(lambda x: nib.as_closest_canonical(x).get_fdata(), nii)

                if volume.ndim != 3:
                    raise HTTPException(status_code=400, detail=f"File {file.filename} is not a 3D volume")

                original_slices_map[file.filename] = []
                for idx in range(volume.shape[2]):
                    slice_2d = volume[:, :, idx]
                    slice_2d = pad_or_resize_to_224(slice_2d)
                    slice_2d = np.clip(slice_2d, np.percentile(slice_2d, 0.5), np.percentile(slice_2d, 99.5))
                    slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)

                    all_slices.append(slice_2d)
                    original_slices_map[file.filename].append(slice_2d.copy())
                    slice_file_map.append(file.filename)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to process file {file.filename}: {e}")
            return {"error": f"Failed to process {file.filename}: {e}"}

    #Batch inference over all slices
    predicted_slices_map = {fname: [] for fname in original_slices_map}
    num_slices_total = len(all_slices)
    start_time = time.time()

    for i in range(0, num_slices_total, batch_size):
        batch = all_slices[i:i + batch_size]
        tensor_batch = torch.from_numpy(np.stack(batch)).unsqueeze(1).to(device, dtype=torch.float32)

        with torch.no_grad():
            output = model(tensor_batch)
            preds = torch.sigmoid(output)[:, 0, :, :]

            for idx, pred_mask in enumerate(preds):
                pred_mask = (pred_mask > 0.5).float()
                if pred_mask.mean() > 0.5:
                    pred_mask = 1 - pred_mask
                filename = slice_file_map[i + idx]
                predicted_slices_map[filename].append(pred_mask.cpu().numpy())

    elapsed = time.time() - start_time
    logger.info(f"Total inference for {num_slices_total} slices completed in {elapsed:.2f} sec")

    #Save overlays per file in background
    for fname in files:
        background_tasks.add_task(
            save_slice_overlays,
            predicted_slices_map[fname.filename],
            original_slices_map[fname.filename],
            save_dir
        )

    # Return summary
    return {
        "message": "Inference completed; saving overlays in background",
        "num_files": len(files),
        "num_slices_total": num_slices_total,
        "inference_time_sec": elapsed,
        "slices_per_file": {fname.filename: len(original_slices_map[fname.filename]) for fname in files}
    }
