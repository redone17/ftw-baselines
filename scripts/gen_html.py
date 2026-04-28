"""Generate interactive HTML viewers from existing SAM2 output TIFs."""
import os
import json
import numpy as np
from PIL import Image
import rasterio
from rasterio.features import shapes as rio_shapes, geometry_mask as rio_geom_mask
import geopandas as gpd
from shapely.geometry import shape as shapely_shape

OUTPUT_DIR     = 'outputs/sam2_results'
ORIGIN_X       = 500_000.0
ORIGIN_Y       = 5_400_000.0
PIXEL_SIZE_M   = 10.0
POLY_MIN_SIZE  = 500.0
DRONE_MAX_SIDE = 1024

_HTML = """\
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


def polygonize(morph_tif, min_size):
    with rasterio.open(morph_tif) as src:
        data = src.read(1)
        t    = src.transform
        crs  = src.crs
    fm    = (data == 1).astype('uint8')
    geoms = [shapely_shape(g) for g, v in rio_shapes(fm, mask=fm, transform=t) if v == 1]
    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    return gdf[gdf.geometry.area >= min_size].reset_index(drop=True)


def attach_confidence(gdf, score_tif):
    if not os.path.exists(score_tif) or len(gdf) == 0:
        gdf['confidence'] = 0.0
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
    gdf['confidence'] = confs


def svg_pts(geom, ox, oy, ps):
    polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
    for p in polys:
        p = p.simplify(ps, preserve_topology=True)
        yield ' '.join(f'{(x - ox) / ps:.1f},{(oy - y) / ps:.1f}' for x, y in p.exterior.coords)


stems = [
    ('C081_orginal', 'drone',     'data/sga-test/drone/C081_orginal.jpg'),
    ('H6_orginal',   'drone',     'data/sga-test/drone/H6_orginal.jpg'),
    ('S10_orginal',  'drone',     'data/sga-test/drone/S10_orginal.jpg'),
    ('demo_area01',  'satellite', 'data/sga-test/satellite/demo_area01.jpg'),
    ('demo_area02',  'satellite', 'data/sga-test/satellite/demo_area02.jpg'),
]

for stem, sensor, src_path in stems:
    morph_tif = os.path.join(OUTPUT_DIR, f'{stem}_morph.tif')
    score_tif = os.path.join(OUTPUT_DIR, f'{stem}_score.tif')
    if not os.path.exists(morph_tif):
        print(f'SKIP {stem}: no morph TIF')
        continue

    gdf = polygonize(morph_tif, POLY_MIN_SIZE)
    attach_confidence(gdf, score_tif)

    rgb = np.array(Image.open(src_path).convert('RGB'))
    if sensor == 'drone':
        h, w  = rgb.shape[:2]
        scale = min(DRONE_MAX_SIDE / max(h, w), 1.0)
        if scale < 1.0:
            nh, nw = int(h * scale), int(w * scale)
            rgb = np.array(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    H, W = rgb.shape[:2]

    img_file = f'{stem}_display.jpg'
    Image.fromarray(rgb).save(os.path.join(OUTPUT_DIR, img_file), format='JPEG', quality=90)

    svgl, pdata = [], []
    for idx, row in enumerate(gdf.itertuples(index=False)):
        geom    = row.geometry
        area_m2 = int(geom.area)
        area_px = int(geom.area / PIXEL_SIZE_M ** 2)
        conf    = float(getattr(row, 'confidence', 0.0))
        for pts in svg_pts(geom, ORIGIN_X, ORIGIN_Y, PIXEL_SIZE_M):
            svgl.append(f'      <polygon class="poly" data-i="{idx}" points="{pts}"/>')
        pdata.append({'p': area_px, 'm': area_m2, 'c': round(conf, 4)})

    subs = {
        '__TITLE__':  stem,
        '__SENSOR__': sensor,
        '__DIMS__':   f'{W}x{H} px',
        '__N__':      str(len(gdf)),
        '__IMG__':    img_file,
        '__W__':      str(W),
        '__H__':      str(H),
        '__SVGS__':   '\n'.join(svgl),
        '__DATA__':   json.dumps(pdata),
    }
    html = _HTML
    for k, v in subs.items():
        html = html.replace(k, v)
    out = os.path.join(OUTPUT_DIR, f'{stem}_interactive.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    conf_avg = gdf['confidence'].mean()
    print(f'  {os.path.basename(out)}  ({len(gdf)} polygons, conf={conf_avg:.3f})')

print('Done.')
