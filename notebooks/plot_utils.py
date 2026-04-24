import glob
import json
import os
import random
import re
import subprocess
import tempfile

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import dilation, disk, erosion
from skimage.segmentation import watershed

random.seed(42)


def _sample_sort_key(path):
    m = re.search(r"sample_(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else -1


def run_sample_inference(filenames, idx, checkpoint, test_dir):
    stacked_tif = f"{test_dir}/sample_{idx}_stacked.tif"
    pred_tif = f"{test_dir}/sample_{idx}_pred.tif"
    with rasterio.open(filenames["window_b"]) as src:
        profile, data_b = src.profile.copy(), src.read()
    with rasterio.open(filenames["window_a"]) as src:
        data_a = src.read()
    stack = np.vstack([data_b, data_a])
    # FTW tiles are south-up (transform.e > 0); `ftw inference run` assumes north-up for
    # patch placement, so flip rows + rewrite the transform before writing the input.
    t = profile["transform"]
    flipped = t.e > 0
    if flipped:
        stack = stack[:, ::-1, :]
        profile["transform"] = Affine(
            t.a, t.b, t.c, t.d, -t.e, t.f + t.e * stack.shape[1]
        )
    profile.update(count=stack.shape[0])
    with rasterio.open(stacked_tif, "w", **profile) as dst:
        dst.write(stack)
    result = subprocess.run(
        [
            "ftw",
            "inference",
            "run",
            stacked_tif,
            "-m",
            checkpoint,
            "-o",
            pred_tif,
            "-r",
            "1",
            "--gpu=-1",
            "-ps",
            "256",
            "-bs",
            "1",
            "--num_workers",
            "1",
            "-f",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(result.stdout)
        print("STDERR:\n", result.stderr)
        raise RuntimeError(f"ftw inference run failed for sample {idx}")
    with rasterio.open(pred_tif) as src:
        pred = src.read(1)
    # Flip prediction back to match the original south-up orientation of mask/image.
    if flipped:
        pred = pred[::-1, :]
    return pred


def plot_sample_predictions(ds, indices, checkpoint, test_dir, cmap):
    indices = sorted(indices)
    fig, axes = plt.subplots(len(indices), 4, figsize=(18, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[np.newaxis, :]
    mask_paths = {}
    for row, idx in enumerate(indices):
        sample = ds[idx]
        img, mask = sample["image"], sample["mask"].numpy()
        pred = run_sample_inference(ds.filenames[idx], idx, checkpoint, test_dir)
        mask_paths[idx] = ds.filenames[idx]["mask"]
        rgb = img[:3].permute(1, 2, 0).numpy()
        rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8), 0, 1)
        overlay = np.zeros((*pred.shape, 4), dtype=np.float32)
        overlay[pred == 1] = [0.3, 0.8, 0.3, 0.4]
        overlay[pred == 2] = [0.9, 0.2, 0.2, 0.7]
        panels = [
            (rgb, f"RGB (sample {idx})", {}),
            (
                mask,
                "Ground Truth",
                {"cmap": cmap, "vmin": 0, "vmax": 3, "interpolation": "nearest"},
            ),
            (
                pred,
                "Prediction",
                {"cmap": cmap, "vmin": 0, "vmax": 3, "interpolation": "nearest"},
            ),
        ]
        for col, (data, title, kw) in enumerate(panels):
            axes[row, col].imshow(data, **kw)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
        axes[row, 3].imshow(rgb)
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Prediction Overlay")
        axes[row, 3].axis("off")
    with open(os.path.join(test_dir, "sample_mask_paths.json"), "w") as f:
        json.dump({str(k): v for k, v in mask_paths.items()}, f)
    legend_patches = [
        mpatches.Patch(color="#888888", label="Background"),
        mpatches.Patch(color="#4CAF50", label="Crop"),
        mpatches.Patch(color="#F44336", label="Boundary"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(f"{test_dir}/sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()


def _polygonize_tif(tif_path, parquet_path, simplify, min_size):
    result = subprocess.run(
        [
            "ftw",
            "inference",
            "polygonize",
            tif_path,
            "-o",
            parquet_path,
            "-f",
            "--simplify",
            str(simplify),
            "--min_size",
            str(min_size),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(result.stdout)
        print("STDERR:\n", result.stderr)
        raise RuntimeError(f"ftw inference polygonize failed for {tif_path}")
    return gpd.read_parquet(parquet_path)


def plot_polygonized_results(
    test_dir, cmap, simplify=15, min_size=500, postprocess=None
):
    """Plot GT mask, pred polygons, and optionally post-processed polygons.

    Args:
        postprocess: Optional suffix of a post-processed raster to polygonize
            alongside the raw prediction (e.g. "morph" or "watershed").
            Expects files named sample_*_{postprocess}.tif in test_dir.
            Watershed instance labels (uint16) are binarized before polygonizing.
    """
    pred_tifs = sorted(glob.glob(f"{test_dir}/sample_*_pred.tif"), key=_sample_sort_key)
    if not pred_tifs:
        print(f"No pred TIFs found in {test_dir}. Run section 2 first.")
        return

    mask_paths_file = os.path.join(test_dir, "sample_mask_paths.json")
    mask_paths = {}
    if os.path.exists(mask_paths_file):
        with open(mask_paths_file) as f:
            mask_paths = json.load(f)

    ncols = 3 if postprocess else 2
    fig, axes = plt.subplots(
        len(pred_tifs), ncols, figsize=(6 * ncols, 5 * len(pred_tifs))
    )
    if len(pred_tifs) == 1:
        axes = axes[np.newaxis, :]

    for row, pred_tif in enumerate(pred_tifs):
        name = os.path.basename(pred_tif).replace("_pred.tif", "")
        idx_str = name.replace("sample_", "")
        fields_parquet = pred_tif.replace("_pred.tif", "_fields.parquet")
        fields = _polygonize_tif(pred_tif, fields_parquet, simplify, min_size)

        with rasterio.open(pred_tif) as src:
            north_up = src.transform.e < 0

        # Col 0: GT mask
        mask_path = mask_paths.get(idx_str)
        if mask_path and os.path.exists(mask_path):
            with rasterio.open(mask_path) as src:
                gt_mask = src.read(1)
            axes[row, 0].imshow(
                gt_mask, cmap=cmap, vmin=0, vmax=3, interpolation="nearest"
            )
        else:
            axes[row, 0].text(
                0.5,
                0.5,
                "No GT mask",
                ha="center",
                va="center",
                transform=axes[row, 0].transAxes,
            )
        axes[row, 0].set_title(f"{name} — GT Mask")
        axes[row, 0].axis("off")

        # Col 1: pred polygons
        fields.plot(
            ax=axes[row, 1], facecolor="#4CAF5066", edgecolor="#F44336", linewidth=0.5
        )
        if north_up:
            axes[row, 1].invert_yaxis()
        axes[row, 1].set_title(f"{name} — Polygons (n={len(fields)})")
        axes[row, 1].axis("off")

        if postprocess:
            pp_tif = pred_tif.replace("_pred.tif", f"_{postprocess}.tif")
            if not os.path.exists(pp_tif):
                axes[row, 2].text(
                    0.5,
                    0.5,
                    f"No {postprocess} TIF",
                    ha="center",
                    va="center",
                    transform=axes[row, 2].transAxes,
                )
                axes[row, 2].axis("off")
                continue

            with rasterio.open(pp_tif) as src:
                pp_raster = src.read(1)
                pp_profile = src.profile.copy()

            # Watershed output has instance labels (uint16); binarize for polygonize.
            poly_tif = pp_tif
            if postprocess == "watershed":
                tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                poly_tif = tmp.name
                tmp.close()
                binary = (pp_raster > 0).astype(np.uint8)
                pp_profile.update(dtype="uint8", nodata=0)
                with rasterio.open(poly_tif, "w", **pp_profile) as dst:
                    dst.write(binary[np.newaxis, :])

            pp_parquet = pred_tif.replace("_pred.tif", f"_{postprocess}_fields.parquet")
            pp_fields = _polygonize_tif(poly_tif, pp_parquet, simplify, min_size)

            label = postprocess.replace("_", " ").title()
            pp_fields.plot(
                ax=axes[row, 2],
                facecolor="#4CAF5066",
                edgecolor="#F44336",
                linewidth=0.5,
            )
            if north_up:
                axes[row, 2].invert_yaxis()
            axes[row, 2].set_title(f"{name} — {label} Polygons (n={len(pp_fields)})")
            axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(f"{test_dir}/full_scene_results.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_polygon_size_distribution(test_dir, postprocess):
    """Compare raw vs post-processed field size distributions.

    Args:
        postprocess: Suffix of post-processed parquet files (e.g. "morph",
            "watershed"). Expects files named sample_*_{postprocess}_fields.parquet.
    """

    def _load_areas(pattern, exact=False):
        candidates = sorted(glob.glob(pattern), key=_sample_sort_key)
        # when exact=True, keep only sample_{digits}_fields.parquet (no extra suffix)
        parquets = (
            [f for f in candidates if re.search(r"sample_\d+_fields\.parquet$", f)]
            if exact
            else candidates
        )
        if not parquets:
            return None
        gdf = gpd.GeoDataFrame(
            pd.concat([gpd.read_parquet(f) for f in parquets], ignore_index=True)
        )
        gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
        return gdf_utm.geometry.area / 10_000

    areas_ha = _load_areas(f"{test_dir}/sample_*_fields.parquet", exact=True)
    if areas_ha is None:
        print("No fields parquets found. Run section 3 first.")
        return

    pp_areas_ha = _load_areas(f"{test_dir}/sample_*_{postprocess}_fields.parquet")
    if pp_areas_ha is None:
        print(f"No {postprocess} parquets found. Run post-processing first.")
        return

    label = postprocess.replace("_", " ").title()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, data, title, color in [
        (axes[0], areas_ha, "Raw Pred — Field Size Distribution", "#4CAF50"),
        (axes[1], pp_areas_ha, f"{label} — Field Size Distribution", "#2196F3"),
    ]:
        ax.hist(data, bins=50, color=color, edgecolor="white")
        ax.set(xlabel="Field area (ha)", ylabel="Count", title=title)

    plt.tight_layout()
    plt.show()

    for areas, name in [(areas_ha, "Raw pred"), (pp_areas_ha, label)]:
        print(
            f"{name}: Total={len(areas)}  Median={areas.median():.2f} ha  "
            f"Mean={areas.mean():.2f} ha  Max={areas.max():.2f} ha"
        )


def morphological_opening(input_raster, output_raster, kernel_size=5):
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()
        data = src.read(1)

    field = (data == 1).astype(np.uint8)
    boundary = (data == 2).astype(np.uint8)
    kernel = disk(kernel_size)
    opened = dilation(erosion(field, kernel), kernel)

    result = np.zeros_like(data)
    result[opened == 1] = 1
    result[boundary == 1] = 2

    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(result[np.newaxis, :])
    return result


def watershed_segmentation(input_raster, output_raster, kernel_size=5):
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()
        data = src.read(1)

    field = (data == 1).astype(np.uint8)
    distance = ndimage.distance_transform_edt(field)
    coords = peak_local_max(distance, min_distance=kernel_size, labels=field)
    peak_mask = np.zeros(distance.shape, dtype=bool)
    if coords.size > 0:
        peak_mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(peak_mask)
    labels = watershed(-distance, markers, mask=field)

    profile.update(dtype="uint16", nodata=None)
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(labels.astype(np.uint16)[np.newaxis, :])
    return labels


def plot_post_processing_results(
    test_dir, cmap, morph_kernel=None, watershed_kernel=None, chain=False
):
    """Plot post-processing results for each pred TIF.

    Args:
        morph_kernel: Disk radius for morphological opening. Skipped if None.
        watershed_kernel: min_distance for watershed peak detection. Skipped if None.
        chain: If True, watershed runs on the morph output instead of raw pred
            (requires morph_kernel to also be set).
    """
    pred_tifs = sorted(glob.glob(f"{test_dir}/sample_*_pred.tif"), key=_sample_sort_key)
    if not pred_tifs:
        print(f"No pred TIFs found in {test_dir}. Run section 2 first.")
        return

    run_morph = morph_kernel is not None
    run_watershed = watershed_kernel is not None
    ncols = 2 + run_morph + run_watershed
    if ncols == 2:
        print("No post-processing specified. Set morph_kernel and/or watershed_kernel.")
        return

    mask_paths_file = os.path.join(test_dir, "sample_mask_paths.json")
    mask_paths = {}
    if os.path.exists(mask_paths_file):
        with open(mask_paths_file) as f:
            mask_paths = json.load(f)

    fig, axes = plt.subplots(
        len(pred_tifs), ncols, figsize=(5 * ncols, 5 * len(pred_tifs))
    )
    if len(pred_tifs) == 1:
        axes = axes[np.newaxis, :]

    for row, pred_tif in enumerate(pred_tifs):
        name = os.path.basename(pred_tif).replace("_pred.tif", "")
        idx_str = name.replace("sample_", "")

        with rasterio.open(pred_tif) as src:
            pred = src.read(1)
            # pred TIF is stored north-up (e < 0) because run_sample_inference
            # flips south-up FTW tiles before inference but doesn't flip back on disk.
            # Flip to match the south-up orientation of the GT mask.
            north_up = src.transform.e < 0
        if north_up:
            pred = pred[::-1, :]

        col = 0

        # Load GT mask once; used for IoU computation across all columns
        gt_mask = None
        mask_path = mask_paths.get(idx_str)
        if mask_path and os.path.exists(mask_path):
            with rasterio.open(mask_path) as src:
                gt_mask = src.read(1)

        def field_iou(pred_arr, gt):
            if gt is None:
                return None
            p = (pred_arr > 0)
            g = (gt == 1)
            union = (p | g).sum()
            return (p & g).sum() / union if union > 0 else 0.0

        def iou_str(iou):
            return f"  IoU={iou:.3f}" if iou is not None else ""

        # Col 0: GT mask
        if gt_mask is not None:
            axes[row, col].imshow(gt_mask, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        else:
            axes[row, col].text(0.5, 0.5, "No GT mask", ha="center", va="center",
                                transform=axes[row, col].transAxes)
        axes[row, col].set_title(f"{name} — GT Mask")
        axes[row, col].axis("off")
        col += 1

        # Col 1: raw prediction
        iou = field_iou(pred == 1, gt_mask)
        axes[row, col].imshow(pred, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        axes[row, col].set_title(f"{name} — Prediction{iou_str(iou)}")
        axes[row, col].axis("off")
        col += 1

        # Optional col: morphological opening
        if run_morph:
            morph_tif = pred_tif.replace("_pred.tif", "_morph.tif")
            morphological_opening(pred_tif, morph_tif, kernel_size=morph_kernel)
            with rasterio.open(morph_tif) as src:
                morph = src.read(1)
            if north_up:
                morph = morph[::-1, :]
            iou = field_iou(morph == 1, gt_mask)
            axes[row, col].imshow(morph, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
            axes[row, col].set_title(f"{name} — Morph (k={morph_kernel}){iou_str(iou)}")
            axes[row, col].axis("off")
            col += 1

        # Optional col: watershed
        if run_watershed:
            watershed_tif = pred_tif.replace("_pred.tif", "_watershed.tif")
            # when chaining, watershed runs on morph output; otherwise directly on pred
            watershed_input = morph_tif if (chain and run_morph) else pred_tif
            watershed_segmentation(watershed_input, watershed_tif, kernel_size=watershed_kernel)
            with rasterio.open(watershed_tif) as src:
                wshed = src.read(1)
            if north_up:
                wshed = wshed[::-1, :]
            iou = field_iou(wshed > 0, gt_mask)
            wshed_title = (
                f"Morph→Watershed (k={watershed_kernel})"
                if (chain and run_morph)
                else f"Watershed (k={watershed_kernel})"
            )
            axes[row, col].imshow(wshed, interpolation="nearest")
            axes[row, col].set_title(f"{name} — {wshed_title}{iou_str(iou)}")
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(f"{test_dir}/post_processing_results.png", dpi=150, bbox_inches="tight")
    plt.show()
