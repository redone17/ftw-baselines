"""Utility functions for SAM2 field-boundary inference notebooks."""
import warnings
warnings.filterwarnings("ignore")

import json
import os

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image
from rasterio.features import geometry_mask as rio_geom_mask
from rasterio.features import shapes as rio_shapes
from rasterio.transform import from_origin
from scipy.ndimage import binary_erosion
from shapely.geometry import shape as shapely_shape

from plot_utils import morphological_opening, watershed_segmentation

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
CMAP = ListedColormap(["#888888", "#4CAF50", "#F44336"])
LEGEND_PATCHES = [
    mpatches.Patch(color="#888888", label="Background (0)"),
    mpatches.Patch(color="#4CAF50", label="Field interior (1)"),
    mpatches.Patch(color="#F44336", label="Boundary (2)"),
]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def resize_keep_aspect(rgb, max_side):
    h, w = rgb.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale == 1.0:
        return rgb
    nh, nw = int(h * scale), int(w * scale)
    return np.array(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))


def make_grid_points(h, w, n):
    xs = np.linspace(w * 0.05, w * 0.95, n, dtype=np.float32)
    ys = np.linspace(h * 0.05, h * 0.95, n, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    return np.stack([xv.ravel(), yv.ravel()], axis=1)


def mask_iou(m1, m2):
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter) / float(union + 1e-6)


def nms_masks(masks_scores, iou_thresh):
    order = sorted(range(len(masks_scores)), key=lambda i: masks_scores[i][1], reverse=True)
    kept = []
    for idx in order:
        m_i = masks_scores[idx][0]
        if not any(mask_iou(m_i, masks_scores[j][0]) > iou_thresh for j in kept):
            kept.append(idx)
    return kept


def masks_to_semantic(masks_scores, h, w, boundary_px=2):
    """Return (semantic uint8, score_map float32) from [(mask, score)] list."""
    semantic  = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.float32)
    struct    = np.ones((boundary_px * 2 + 1,) * 2, dtype=bool)
    for mask, score in masks_scores:
        m        = mask.astype(bool)
        eroded   = binary_erosion(m, structure=struct)
        boundary = m & ~eroded
        semantic[m]        = 1
        semantic[boundary] = 2
        score_map[m]       = score
    return semantic, score_map


def predict_single(pred, rgb, grid_n, min_area_ratio, iou_thresh, min_score):
    """Return list of (mask, score) tuples after grid prompting + NMS."""
    h, w     = rgb.shape[:2]
    min_area = max(1, int(min_area_ratio * h * w))
    pred.set_image(rgb)
    grid_pts = make_grid_points(h, w, grid_n)
    raw_ms   = []
    for px, py in grid_pts:
        masks, scores, _ = pred.predict(
            point_coords=np.array([[px, py]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        best  = int(scores.argmax())
        mask  = masks[best].astype(bool)
        score = float(scores[best])
        if mask.sum() >= min_area and score >= min_score:
            raw_ms.append((mask, score))
    kept_idx = nms_masks(raw_ms, iou_thresh)
    return [raw_ms[i] for i in kept_idx]


def predict_satellite_patches(pred, rgb, patch_size, overlap, grid_n,
                               min_area_ratio, iou_thresh, boundary_px, min_score):
    """Return (semantic uint8, score_map float32) via tiled SAM2 inference."""
    h, w    = rgb.shape[:2]
    stride  = patch_size - overlap
    semantic  = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.float32)

    def tile_starts(size, ps, st):
        starts = list(range(0, size - ps, st))
        starts.append(max(0, size - ps))
        return sorted(set(starts))

    ys    = tile_starts(h, patch_size, stride)
    xs    = tile_starts(w, patch_size, stride)
    total = len(ys) * len(xs)
    done  = 0

    for y0 in ys:
        y1 = min(y0 + patch_size, h)
        for x0 in xs:
            x1    = min(x0 + patch_size, w)
            patch = rgb[y0:y1, x0:x1]
            ph, pw = patch.shape[:2]

            kept                 = predict_single(pred, patch, grid_n, min_area_ratio,
                                                  iou_thresh, min_score)
            patch_sem, patch_scr = masks_to_semantic(kept, ph, pw, boundary_px)

            vy0 = overlap // 2 if y0 > 0 else 0
            vy1 = ph - overlap // 2 if y1 < h else ph
            vx0 = overlap // 2 if x0 > 0 else 0
            vx1 = pw - overlap // 2 if x1 < w else pw
            semantic[y0 + vy0: y0 + vy1, x0 + vx0: x0 + vx1] = patch_sem[vy0:vy1, vx0:vx1]
            score_map[y0 + vy0: y0 + vy1, x0 + vx0: x0 + vx1] = patch_scr[vy0:vy1, vx0:vx1]

            done += 1
            print(f"  patch {done}/{total}  [{y0}:{y1}, {x0}:{x1}]  fields={len(kept)}")

    return semantic, score_map


# ---------------------------------------------------------------------------
# GeoTIFF I/O
# ---------------------------------------------------------------------------

def save_semantic_geotiff(semantic, out_tif, origin_x, origin_y, pixel_size, crs):
    h, w      = semantic.shape
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)
    profile   = dict(driver="GTiff", dtype="uint8", width=w, height=h,
                     count=1, crs=crs, transform=transform, compress="lzw")
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(semantic[np.newaxis, :])


def save_score_geotiff(score_map, out_tif, origin_x, origin_y, pixel_size, crs):
    h, w      = score_map.shape
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)
    profile   = dict(driver="GTiff", dtype="float32", width=w, height=h,
                     count=1, crs=crs, transform=transform, compress="lzw")
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(score_map[np.newaxis, :])


def semantic_to_rgb(semantic):
    out = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    out[semantic == 0] = [136, 136, 136]
    out[semantic == 1] = [ 76, 175,  80]
    out[semantic == 2] = [244,  67,  54]
    return out


def polygonize_morph_tif(morph_tif, min_size_m2):
    with rasterio.open(morph_tif) as src:
        data      = src.read(1)
        transform = src.transform
        crs       = src.crs
    field_mask = (data == 1).astype("uint8")
    geoms = [
        shapely_shape(geom)
        for geom, val in rio_shapes(field_mask, mask=field_mask, transform=transform)
        if val == 1
    ]
    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    return gdf[gdf.geometry.area >= min_size_m2].reset_index(drop=True)


def geom_to_pixel_patches(geom, origin_x, origin_y, pixel_size):
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    patches = []
    for poly in polys:
        xs, ys = poly.exterior.xy
        px = [(x - origin_x) / pixel_size for x in xs]
        py = [(origin_y - y) / pixel_size  for y in ys]
        patches.append(MplPolygon(list(zip(px, py))))
    return patches


# ---------------------------------------------------------------------------
# Post-processing pipeline
# ---------------------------------------------------------------------------

def _attach_confidence(gdf, score_tif):
    """Add mean SAM2 score per polygon in-place; 0.0 if score TIF is absent."""
    if not os.path.exists(score_tif) or len(gdf) == 0:
        gdf["confidence"] = 0.0
        return
    with rasterio.open(score_tif) as src:
        sd = src.read(1)
        xf = src.transform
    confs = []
    for geom in gdf.geometry:
        try:
            m = ~rio_geom_mask([geom], out_shape=sd.shape, transform=xf)
            v = sd[m]
            confs.append(float(v[v > 0].mean()) if (v > 0).any() else 0.0)
        except Exception:
            confs.append(0.0)
    gdf["confidence"] = confs


def run_postprocessing(results, output_dir, morph_kernel, wshed_kernel):
    """Run morphological opening + watershed on all pred TIFs.

    Returns (morph_tif_paths, wshed_tif_paths, postproc_arrs) where
    postproc_arrs[stem] = {'morph': ndarray, 'wshed': ndarray}.
    """
    morph_tif_paths, wshed_tif_paths, postproc_arrs = {}, {}, {}
    for stem, res in results.items():
        morph_tif = os.path.join(output_dir, f"{stem}_morph.tif")
        wshed_tif = os.path.join(output_dir, f"{stem}_watershed.tif")
        morph = morphological_opening(res["pred_tif"], morph_tif, kernel_size=morph_kernel)
        wshed = watershed_segmentation(morph_tif, wshed_tif, kernel_size=wshed_kernel)
        morph_tif_paths[stem] = morph_tif
        wshed_tif_paths[stem] = wshed_tif
        postproc_arrs[stem]   = {"morph": morph, "wshed": wshed}
        print(f"{stem}: morph field px={int((morph == 1).sum())}  "
              f"watershed instances={int(wshed.max())}")
    return morph_tif_paths, wshed_tif_paths, postproc_arrs


def run_polygonize(morph_tif_paths, output_dir, poly_min_size, pixel_size_m):
    """Polygonize morph TIFs; attach per-polygon SAM2 confidence if available."""
    polygon_results = {}
    for stem, morph_tif in morph_tif_paths.items():
        gdf       = polygonize_morph_tif(morph_tif, min_size_m2=poly_min_size)
        score_tif = os.path.join(output_dir, f"{stem}_score.tif")
        _attach_confidence(gdf, score_tif)
        polygon_results[stem] = gdf
        print(f"  {stem}: {len(gdf)} polygons  "
              f"(conf: {gdf['confidence'].mean():.3f} avg)")
    return polygon_results


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_sam2_predictions(results, images, output_dir):
    """3-panel figure: original RGB | semantic mask | colour overlay."""
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (stem, res) in enumerate(results.items()):
        rgb      = res["display_rgb"]
        semantic = res["semantic"]
        group    = images[stem]["group"]

        overlay = np.zeros((*semantic.shape, 4), dtype=np.float32)
        overlay[semantic == 1] = [0.3, 0.8, 0.3, 0.45]
        overlay[semantic == 2] = [0.9, 0.2, 0.2, 0.70]

        for col, (data, title, kw) in enumerate([
            (rgb,      f"{stem} [{group}]\nOriginal RGB", {}),
            (semantic, f"{stem}\nSAM2 Semantic",
             {"cmap": CMAP, "vmin": 0, "vmax": 2, "interpolation": "nearest"}),
        ]):
            axes[row, col].imshow(data, **kw)
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")

        axes[row, 2].imshow(rgb)
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"{stem}\nSAM2 Overlay", fontsize=9)
        axes[row, 2].axis("off")

    fig.legend(handles=LEGEND_PATCHES, loc="lower center", ncol=3, fontsize=11)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(os.path.join(output_dir, "sam2_raw_predictions.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_postprocessing(results, postproc_arrs, output_dir, morph_k, wshed_k):
    """4-panel figure: RGB | SAM2 raw | morphological | watershed."""
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(22, 6 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (stem, res) in enumerate(results.items()):
        arrs  = postproc_arrs[stem]
        rgb   = res["display_rgb"]
        sem   = res["semantic"]
        morph = arrs["morph"]
        wshed = arrs["wshed"]

        panels = [
            (rgb,   f"{stem}\nOriginal RGB",        {}),
            (sem,   "SAM2 Raw",                     {"cmap": CMAP, "vmin": 0, "vmax": 2, "interpolation": "nearest"}),
            (morph, f"Morph Opening (k={morph_k})", {"cmap": CMAP, "vmin": 0, "vmax": 2, "interpolation": "nearest"}),
            (wshed, f"Watershed (k={wshed_k})",     {"interpolation": "nearest"}),
        ]
        for col, (data, title, kw) in enumerate(panels):
            axes[row, col].imshow(data, **kw)
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")

    fig.legend(handles=LEGEND_PATCHES, loc="lower center", ncol=3, fontsize=11)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(os.path.join(output_dir, "postprocessing_results.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_polygon_overlays(polygon_results, results, images, origin_x, origin_y,
                          pixel_size, output_dir):
    """2-panel figure: original RGB | polygon overlay."""
    n = len(polygon_results)
    fig, axes = plt.subplots(n, 2, figsize=(16, 7 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (stem, gdf) in enumerate(polygon_results.items()):
        rgb  = results[stem]["display_rgb"]
        h, w = rgb.shape[:2]

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f'{stem} [{images[stem]["group"]}]\nOriginal RGB', fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(rgb)
        if len(gdf) > 0:
            patches = []
            for geom in gdf.geometry:
                if geom is not None and not geom.is_empty:
                    patches.extend(geom_to_pixel_patches(geom, origin_x, origin_y, pixel_size))
            if patches:
                pc = PatchCollection(patches, facecolor="#4CAF5033", edgecolor="#F44336",
                                     linewidth=0.8, match_original=False)
                axes[row, 1].add_collection(pc)
        axes[row, 1].set_xlim(0, w)
        axes[row, 1].set_ylim(h, 0)
        axes[row, 1].set_title(f"{stem}\nField Polygons (n={len(gdf)})", fontsize=9)
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "polygon_overlay.png"), dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Interactive HTML viewer
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAM2 Field Boundary — __TITLE__</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f172a;color:#e2e8f0}
header{padding:12px 18px;background:#1e293b;border-bottom:1px solid #334155;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1{font-size:14px;font-weight:500;color:#94a3b8}
h2{font-size:19px;font-weight:700;color:#4ade80}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{padding:6px 15px;border:none;border-radius:5px;cursor:pointer;font-size:13px;font-weight:600;transition:background .15s}
.btn-on{background:#16a34a;color:#fff}.btn-on:hover{background:#15803d}
.btn-off{background:#475569;color:#cbd5e1}.btn-off:hover{background:#334155}
.badge{font-size:12px;color:#94a3b8;padding:4px 10px;background:#1e293b;border-radius:4px;border:1px solid #334155}
.wrap{padding:14px;overflow:auto}
.frame{position:relative;display:inline-block}
.frame img{display:block;max-width:100%;height:auto}
.frame svg{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
.poly{fill:rgba(74,222,128,.12);stroke:#f87171;stroke-width:1.5;pointer-events:all;cursor:pointer;transition:fill .1s}
.poly:hover{fill:rgba(74,222,128,.40)}
.poly.sel{fill:rgba(251,191,36,.35);stroke:#fbbf24}
#pop{position:fixed;background:rgba(15,23,42,.96);border:1px solid #475569;border-radius:8px;padding:12px 15px;font-size:13px;line-height:1.65;box-shadow:0 8px 28px rgba(0,0,0,.55);z-index:999;display:none;pointer-events:none;min-width:190px}
.pl{color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.pv{color:#e2e8f0;font-weight:600}
.pt{color:#4ade80;font-weight:700;font-size:14px;margin-bottom:7px}
</style>
</head>
<body>
<header>
  <div>
    <div style="margin-bottom:3px"><h1>SAM2.1 Field Boundary&nbsp;&nbsp;</h1><h2>__TITLE__</h2></div>
    <div class="controls">
      <button id="tb" class="btn-on" onclick="tog()">Hide Polygons</button>
      <span class="badge">__N__ polygons</span>
      <span class="badge">__SENSOR__&nbsp;&middot;&nbsp;__DIMS__</span>
    </div>
  </div>
</header>
<div class="wrap">
  <div class="frame">
    <img src="__IMG__" alt="__TITLE__">
    <svg id="ov" viewBox="0 0 __W__ __H__" preserveAspectRatio="none">
__SVGS__
    </svg>
  </div>
</div>
<div id="pop"></div>
<script>
const D=__DATA__;
let on=true,sel=null;
function tog(){
  on=!on;
  document.getElementById("ov").style.display=on?"":"none";
  const b=document.getElementById("tb");
  b.textContent=on?"Hide Polygons":"Show Polygons";
  b.className=on?"btn-on":"btn-off";
}
const pop=document.getElementById("pop");
document.querySelectorAll(".poly").forEach(el=>{
  el.addEventListener("click",e=>{
    if(sel)sel.classList.remove("sel");
    el.classList.add("sel");sel=el;
    const d=D[+el.dataset.i];
    const cf=d.c>0?(d.c*100).toFixed(1)+"%":"N/A";
    pop.innerHTML="<div class=\\"pt\\">Field #"+(+el.dataset.i+1)+"</div>"
      +"<div><span class=\\"pl\\">Area</span><br>"
      +"<span class=\\"pv\\">"+d.p.toLocaleString()+" px²</span>"
      +"&nbsp;<span style=\\"color:#64748b\\">("+d.m.toLocaleString()+" m²)</span></div>"
      +"<div style=\\"margin-top:7px\\"><span class=\\"pl\\">SAM2 Confidence</span><br>"
      +"<span class=\\"pv\\">"+cf+"</span></div>";
    pop.style.display="block";pos(e);e.stopPropagation();
  });
});
function pos(e){
  const vw=innerWidth,vh=innerHeight,pw=210,ph=115;
  let x=e.clientX+14,y=e.clientY+14;
  if(x+pw>vw)x=e.clientX-pw-6;
  if(y+ph>vh)y=e.clientY-ph-6;
  pop.style.left=x+"px";pop.style.top=y+"px";
}
document.addEventListener("click",e=>{
  if(!e.target.classList.contains("poly")){
    pop.style.display="none";
    if(sel){sel.classList.remove("sel");sel=null;}
  }
});
</script>
</body>
</html>"""


def _geom_to_svg(geom, ox, oy, ps):
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        poly   = poly.simplify(ps, preserve_topology=True)
        coords = list(poly.exterior.coords)
        yield " ".join(f"{(x - ox) / ps:.1f},{(oy - y) / ps:.1f}" for x, y in coords)


def make_interactive_html(stem, display_rgb, gdf, out_dir, sensor,
                          origin_x, origin_y, pixel_size):
    """Write *_interactive.html + *_display.jpg to out_dir; return html path."""
    H, W = display_rgb.shape[:2]

    img_file = f"{stem}_display.jpg"
    Image.fromarray(display_rgb).save(
        os.path.join(out_dir, img_file), format="JPEG", quality=90
    )

    svg_lines, poly_data = [], []
    for idx, row in enumerate(gdf.itertuples(index=False)):
        geom    = row.geometry
        area_m2 = int(geom.area)
        area_px = int(geom.area / (pixel_size ** 2))
        conf    = float(getattr(row, "confidence", 0.0))
        for pts in _geom_to_svg(geom, origin_x, origin_y, pixel_size):
            svg_lines.append(f'      <polygon class="poly" data-i="{idx}" points="{pts}"/>')
        poly_data.append({"p": area_px, "m": area_m2, "c": round(conf, 4)})

    subs = {
        "__TITLE__":  stem,
        "__SENSOR__": sensor,
        "__DIMS__":   f"{W}x{H} px",
        "__N__":      str(len(gdf)),
        "__IMG__":    img_file,
        "__W__":      str(W),
        "__H__":      str(H),
        "__SVGS__":   "\n".join(svg_lines),
        "__DATA__":   json.dumps(poly_data),
    }
    html = _HTML_TEMPLATE
    for k, v in subs.items():
        html = html.replace(k, v)

    out_path = os.path.join(out_dir, f"{stem}_interactive.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
