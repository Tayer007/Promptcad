"""
SmartScaler: Rescales generated CadQuery code to match original STL dimensions.
Pipeline:
  1. Load original STL + generated STL
  2. Compute per-axis scale factors: sorted largest→largest so axis orientation
     of the generated model does not matter.
  3. PRIMARY: directly scale the mesh vertices — always exact, no LLM needed.
  4. SECONDARY: ask LLM to rewrite the code numbers for display (approximate).
  5. Export corrected STL.
"""

import re
import json
import trimesh
import numpy as np
import requests


# ─────────────────────────────────────────────
# 1. Extent helpers
# ─────────────────────────────────────────────

def compute_extents(mesh):
    """Return [dx, dy, dz] bounding-box extents from a trimesh Mesh."""
    b = mesh.bounds          # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    return b[1] - b[0]


# ─────────────────────────────────────────────
# 2. Scale factor computation  (3 independent factors)
# ─────────────────────────────────────────────

def compute_scale_factors(gt_mesh, gen_mesh):
    """
    Compute per-axis scale factors that map gen → gt.

    Strategy: sort both extents largest→smallest, compute ratio per rank,
    then map back to world X/Y/Z.  This makes the result independent of
    which world axis the generated model happens to use for each dimension.
    """
    gt_ext  = compute_extents(gt_mesh)   # [dx, dy, dz]
    gen_ext = compute_extents(gen_mesh)

    # Sort indices of gen extents largest→smallest
    order = np.argsort(gen_ext)[::-1]          # e.g. [1, 0, 2]
    gen_sorted = gen_ext[order]
    gt_sorted  = np.sort(gt_ext)[::-1]         # sort gt the same way

    gen_safe = np.where(gen_sorted < 1e-6, 1e-6, gen_sorted)
    scale_sorted = gt_sorted / gen_safe        # [s_large, s_mid, s_small]

    # Map sorted scales back to world X/Y/Z axes
    scale_xyz = np.empty(3)
    for rank, axis in enumerate(order):
        scale_xyz[axis] = scale_sorted[rank]

    uniform = float(np.mean(scale_sorted))
    is_uniform = bool(np.std(scale_sorted) < 0.05 * np.mean(scale_sorted))

    print(f"GT  extents (X Y Z): {[round(v,2) for v in gt_ext]}")
    print(f"Gen extents (X Y Z): {[round(v,2) for v in gen_ext]}")
    print(f"Scale factors  X={scale_xyz[0]:.4f}  Y={scale_xyz[1]:.4f}  Z={scale_xyz[2]:.4f}")
    print(f"Uniform approx: {uniform:.4f}  ({'uniform' if is_uniform else 'axis-specific'})")

    return {
        'gt_extents':    gt_ext.tolist(),
        'gen_extents':   gen_ext.tolist(),
        'scale_x':       float(scale_xyz[0]),
        'scale_y':       float(scale_xyz[1]),
        'scale_z':       float(scale_xyz[2]),
        'is_uniform':    is_uniform,
        'uniform_scale': uniform,
        'mode':          'uniform' if is_uniform else 'axis',
    }


# ─────────────────────────────────────────────
# 3. Direct STL mesh scaling  (primary — always correct)
# ─────────────────────────────────────────────

def rescale_stl_direct(gen_stl_path, scale_info, output_stl_path):
    """
    Scale each axis of the mesh vertices independently.
    This is mathematically exact — no LLM, no approximation.
    """
    mesh = trimesh.load_mesh(gen_stl_path)
    mesh.vertices[:, 0] *= scale_info['scale_x']
    mesh.vertices[:, 1] *= scale_info['scale_y']
    mesh.vertices[:, 2] *= scale_info['scale_z']
    mesh.export(output_stl_path)
    print(f"Direct mesh rescaling saved: {output_stl_path}")


# ─────────────────────────────────────────────
# 4. LLM-based code rescaling  (secondary — for code display only)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a CadQuery code editor. You receive CadQuery Python code and three scale factors,
one per world axis (X, Y, Z).  Your job: rewrite every SIZE/DIMENSION value using the
correct per-axis scale factor, following the workplane axis mapping rules below.

═══ WORKPLANE → WORLD AXIS MAPPING ═══
  Workplane('XY'):  box(X, Y, Z)   extrude → Z   sketch coords → XY
  Workplane('XZ'):  box(X, Z, Y)   extrude → Y   sketch coords → XZ
  Workplane('ZX'):  box(Z, X, Y)   extrude → Y   sketch coords → ZX
  Workplane('YZ'):  box(Y, Z, X)   extrude → X   sketch coords → YZ

═══ WHAT TO SCALE ═══
  .box(a, b, c)        → scale each arg by its world axis (see mapping above)
  .cylinder(h, r)      → scale h by extrude-axis, r by average of the two lateral axes
  .sphere(r)           → scale r by uniform_scale
  .extrude(dist)       → scale dist by the extrude-axis of the current workplane
  .hole(d, depth)      → d → avg lateral, depth → extrude-axis
  .rect(w, h)          → scale by the two lateral axes of the current workplane
  .circle(r)           → r → avg of the two lateral axes
  .fillet(r)           → scale r by uniform_scale
  .chamfer(d)          → scale d by uniform_scale
  Segment / sketch coordinates → scale each coord by its world axis

═══ DO NOT SCALE ═══
  origin=(...) in Workplane()    workplane(offset=...)    .translate(...)
  Plane name strings             Count / boolean arguments

═══ FORMAT RULES ═══
  Write every result as a plain integer (no expressions).
  BAD:  int(72 * 0.515)   →   GOOD:  37
  Return ONLY the corrected Python code, no markdown, no explanation.\
"""

def rescale_with_llm(code, scale_info, llm_url, model):
    sx = round(scale_info['scale_x'], 4)
    sy = round(scale_info['scale_y'], 4)
    sz = round(scale_info['scale_z'], 4)
    su = round(scale_info['uniform_scale'], 4)

    user_msg = (
        f"Scale factors:\n"
        f"  scale_x = {sx}  (world X axis)\n"
        f"  scale_y = {sy}  (world Y axis)\n"
        f"  scale_z = {sz}  (world Z axis)\n"
        f"  uniform_scale = {su}  (use for sphere/fillet/chamfer)\n\n"
        f"CadQuery code to rescale:\n{code}\n\n"
        f"Return only the rescaled Python code with plain integer values."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }

    resp = requests.post(llm_url, json=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()['choices'][0]['message']['content'].strip()

    # Strip markdown fences if present
    raw = re.sub(r'^```python\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^```\s*',       '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$',       '', raw).strip()
    return raw


# ─────────────────────────────────────────────
# 5. Regex-based code rescaling  (no LLM needed)
# ─────────────────────────────────────────────

def rescale_code_regex(code, uniform_scale):
    """
    Apply a uniform scale factor to every numeric literal in the CadQuery code.
    Skips import lines, comment lines, and origin=/offset= kwargs.
    """
    def scale_num(m):
        val = float(m.group(0))
        return f"{val * uniform_scale:.4g}"

    _NUM = r"(?<!['\"\w])(-?\d+\.?\d*(?:[eE][+-]?\d+)?)(?!['\"\w])"

    lines = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('import ') or stripped.startswith('#'):
            lines.append(line)
            continue
        # Protect origin=(...) and offset=... from scaling
        protected = re.sub(r'(origin\s*=\s*\([^)]*\))', r'\1', line)
        # Split on origin/offset tokens, scale only the non-protected parts
        parts = re.split(r'(origin\s*=\s*\([^)]*\)|offset\s*=\s*[^,)]+)', line)
        scaled_parts = []
        for i, part in enumerate(parts):
            if re.match(r'origin\s*=|offset\s*=', part):
                scaled_parts.append(part)  # leave as-is
            else:
                scaled_parts.append(re.sub(_NUM, scale_num, part))
        lines.append(''.join(scaled_parts))
    return '\n'.join(lines)


# ─────────────────────────────────────────────
# 6. Main pipeline
# ─────────────────────────────────────────────

def smart_scale(
    gt_stl_path,
    gen_stl_path,
    gen_code,
    corrected_stl_path,
    corrected_py_path,
    llm_url="http://127.0.0.1:1234/v1/chat/completions",
    llm_model="local-model",
):
    """
    Full smart scaling pipeline.
    Returns dict with scale info and corrected_code.

    The corrected STL is produced by direct vertex scaling (exact).
    The corrected code is produced by the LLM (best-effort for display).
    """
    gt_mesh  = trimesh.load_mesh(gt_stl_path)
    gen_mesh = trimesh.load_mesh(gen_stl_path)

    scale_info = compute_scale_factors(gt_mesh, gen_mesh)

    # ── PRIMARY: direct mesh scaling — always correct ──────────────────
    rescale_stl_direct(gen_stl_path, scale_info, corrected_stl_path)
    print("Scale applied (direct mesh): STL dimensions are now exact.")

    # ── SECONDARY: regex-based code rescaling — uniform scale on all literals ──
    corrected_code = rescale_code_regex(gen_code, scale_info['uniform_scale'])
    print(f"Regex code rescaling applied (uniform={scale_info['uniform_scale']:.4f})")

    with open(corrected_py_path, 'w') as f:
        f.write(corrected_code)

    scale_info['llm_used']       = False
    scale_info['corrected_code'] = corrected_code
    return scale_info


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: python smart_scaler.py <gt.stl> <gen.stl> <gen_code.py>")
        sys.exit(1)

    gt_stl  = sys.argv[1]
    gen_stl = sys.argv[2]
    gen_py  = sys.argv[3]

    with open(gen_py) as f:
        gen_code = f.read()

    result = smart_scale(
        gt_stl, gen_stl, gen_code,
        gen_stl.replace('.stl', '_corrected.stl'),
        gen_py.replace('.py',  '_corrected.py'),
    )
    print(json.dumps({k: v for k, v in result.items() if k != 'corrected_code'}, indent=2))
