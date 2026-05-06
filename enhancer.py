"""
enhancer.py - Supervisor-model CadQuery code correction.
Sends the current code (+ optional canvas screenshots for vision-capable models)
to a large Ollama model (qwen3.5:122b) and returns improved CadQuery code.
Called from app.py in a background thread.
"""

import re
import requests
# cadquery is imported lazily inside execute_and_export so that this module
# can be imported by Flask even when cadquery is only available in cad_env.

OLLAMA_URL    = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.5:122b"

# ── Compact CadQuery cheat-sheet (~500 tokens, replaces 21K JSON) ──────────
# Groups methods by task so the model sees only what it needs.
CQ_CHEATSHEET = """
# CadQuery Quick Reference

## Workplane init
r = cq.Workplane("XY")           # start on XY plane
  .workplane(offset=N)           # new plane N mm above current
  .transformed(offset=(x,y,z), rotate=(rx,ry,rz))
  .workplaneFromTagged("name")   # jump back to a tagged state

## 2-D drawing (all relative to current workplane)
  .moveTo(x, y)                  # lift pen, no line
  .lineTo(x, y)                  # absolute line
  .line(dx, dy)                  # relative line
  .hLine(d) / .vLine(d)          # horiz / vert
  .polarLine(dist, angle)
  .arc(p1, p2, p3)               # three-point arc
  .radiusArc(end, radius)
  .tangentArcPoint(end)
  .threePointArc(p1, p2)
  .spline([(x,y), ...])
  .close()                       # close the wire

## 2-D primitives
  .rect(w, h)                    # centred rectangle
  .circle(r)
  .ellipse(rx, ry)
  .polygon(nSides, diameter)
  .slot2D(length, diameter, angle)
  .polyline([(x,y), ...])
  .offset2D(d)                   # offset the current wire

## 3-D primitives (each at every stack point)
  .box(l, w, h)
  .cylinder(h, r)
  .sphere(r)
  .wedge(dx,dy,dz, xmin,zmin,xmax,zmax)

## 3-D operations
  .extrude(dist)                 # extrude pending wires
  .extrude(dist, both=True)
  .cutBlind(dist)                # cut inward
  .cutThruAll()
  .revolve(angleDeg, axisStart, axisEnd)
  .loft()                        # loft between wires
  .sweep(path)
  .twistExtrude(dist, angleDeg)
  .shell(thickness)              # hollow out selected faces

## Boolean
  .union(other) / .cut(other) / .intersect(other)
  .combine()

## Selection
  .faces(">Z") / .faces("<Z")    # top / bottom
  .faces(">X") / .faces("<X")   # front / back  ">Y"/"<Y" for sides
  .faces("#Z")                   # faces perpendicular to Z
  .edges("|Z")                   # edges parallel to Z
  .edges(">Z") / .edges("<Z")
  .vertices(">Z")
  .solids() / .shells() / .wires() / .compounds()

## Features on faces (call after .faces(...).workplane())
  .hole(diameter, depth)
  .cboreHole(d, cboreD, cboreDepth)
  .cskHole(d, cskD, cskAngle)
  .fillet(r)                     # round edges (select edges first)
  .chamfer(d)                    # bevel edges

## Transforms
  .translate((x,y,z))
  .rotate(axisStart, axisEnd, angleDeg)
  .rotateAboutCenter(axis, angleDeg)
  .mirror("XY") / .mirrorX() / .mirrorY()
  .center(x, y)

## Arrays
  .rarray(xSpacing, ySpacing, xCount, yCount)
  .polarArray(radius, startAngle, totalAngle, count)
  .pushPoints([(x1,y1), (x2,y2), ...])

## Tags / navigation
  .tag("name")                   # save this state
  .end(n)                        # pop n levels up the chain
"""

# ── System prompt with hard rules ──────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a senior CadQuery code reviewer. You receive the current generated code "
    "and must correct it so the 3-D result better matches the original part.\n\n"
    "MANDATORY RULES — violating any rule makes the code fail:\n"
    "  1. First line must be exactly:  import cadquery as cq\n"
    "  2. Final result variable must be named exactly:  r\n"
    "     Example:  r = (cq.Workplane('XY').box(100, 50, 30))\n"
    "  3. Use ONLY the methods listed in the cheat-sheet below.\n"
    "  4. NEVER call: show_object, display, print, export, save, or import anything else.\n"
    "  5. ALL numeric values must be plain numbers — no variables, no expressions.\n"
    "  5b. Fillet/chamfer radii must be ≤ 10% of the shortest edge length to avoid BRep_API failures.\n"
    "  6. Chain all operations with dots — do not reassign r mid-chain.\n"
    "  7. String plane names must be quoted: 'XY', 'XZ', 'YZ', or '>Z', '<X', etc.\n"
    "  8. Return ONLY the Python code. No markdown, no explanation, no comments.\n\n"
    + CQ_CHEATSHEET
)


def extract_code(raw: str) -> str:
    """
    Robustly extract Python code from the model response.
    Handles: plain code, ```python fences, text before/after the block,
    different result variable names, and truncated output.
    """
    print(f"[enhancer] raw response ({len(raw)} chars):\n{raw}", flush=True)

    # 1. Prefer ```python ... ``` fence
    m = re.search(r"```python\s*\n(.*?)(?:```|$)", raw, re.DOTALL)
    if m:
        code = m.group(1).strip()
    else:
        # 2. Any ``` fence
        m = re.search(r"```\s*\n(.*?)(?:```|$)", raw, re.DOTALL)
        if m:
            code = m.group(1).strip()
        else:
            # 3. Find first Python-looking line and take everything from there
            lines = raw.splitlines()
            start = 0
            for i, line in enumerate(lines):
                s = line.strip()
                if s.startswith("import ") or re.match(r"^[a-zA-Z_]\w*\s*=\s*", s):
                    start = i
                    break
            code = "\n".join(lines[start:]).strip()

    # 4. Ensure cadquery import is present
    if "import cadquery" not in code:
        code = "import cadquery as cq\n" + code

    # 5. Fix truncated output — balance unclosed parentheses
    opens = code.count("(") - code.count(")")
    if opens > 0:
        code = code.rstrip().rstrip("\\").rstrip(",")
        code += ")" * opens

    # 6. If the result isn't assigned to 'r', find what variable it IS in
    #    and add  r = <that_var>  at the end
    if not re.search(r"^\s*r\s*=", code, re.MULTILINE):
        # Look for the last top-level assignment to a Workplane chain
        candidates = re.findall(
            r"^([a-zA-Z_]\w*)\s*=\s*(?:\(?\s*cq\.Workplane|\(?\s*[a-zA-Z_]\w*\.)",
            code, re.MULTILINE
        )
        if candidates:
            last_var = candidates[-1]
            if last_var != "r":
                code += f"\nr = {last_var}"
                print(f"[enhancer] aliased '{last_var}' → r", flush=True)

    print(f"[enhancer] extracted code ({len(code)} chars):\n{code}", flush=True)
    return code.strip()


def _precompute_int_exprs(code: str) -> str:
    """
    Replace int(X * Y) and int(X / Y) expressions with their computed integer values
    so the supervisor model receives clean plain-number code.
    e.g.  int(20.62 * 1.9399)  →  40
    """
    def eval_int_expr(m):
        try:
            return str(int(eval(m.group(0))))  # safe: only numbers + * / - +
        except Exception:
            return m.group(0)

    # Match int( <number> <op> <number> ) with optional nested int()
    pattern = r'int\(\s*-?\d+(?:\.\d+)?\s*[*/+-]\s*-?\d+(?:\.\d+)?\s*(?:[*/+-]\s*\d+(?:\.\d+)?\s*)?\)'
    return re.sub(pattern, eval_int_expr, code)


def call_ollama_supervisor(orig_b64: str, corr_b64: str, current_code: str,
                           model: str, ollama_url: str) -> str:
    """
    Send the current CadQuery code to the supervisor model for correction.
    Images are included only if the model is vision-capable; qwen3.5:122b
    is text-only so the images field is omitted for it automatically.
    """
    # Pre-evaluate int(X*Y) expressions so the supervisor sees plain integers
    clean_code = _precompute_int_exprs(current_code)

    # Models known to be text-only — skip sending images for these
    TEXT_ONLY_MODELS = {"qwen3.5:122b", "qwen2.5:72b", "qwen2.5:32b", "deepseek-r1:671b"}
    is_vision = model.lower() not in TEXT_ONLY_MODELS

    user_content = (
        "You are reviewing auto-generated CadQuery code that was produced by a small model "
        "from a point-cloud scan of a mechanical part, then dimension-corrected by a scaler.\n\n"
        f"Current code:\n{clean_code}\n\n"
        "Your task: fix only clear geometric errors in the EXISTING code.\n"
        "Rules:\n"
        "  - Do NOT add new holes, pockets, cutouts, slots, ribs, or bosses that are not "
        "already present in the code above.\n"
        "  - Do NOT remove existing features.\n"
        "  - You MAY add fillets/chamfers to existing edges if they are clearly missing.\n"
        "  - Keep all dimensions exactly as they are.\n"
        "  - Follow all rules in the system prompt.\n"
        "Return only the corrected Python code."
    )

    user_msg: dict = {"role": "user", "content": user_content}
    if is_vision and orig_b64 and corr_b64:
        user_msg["images"] = [orig_b64, corr_b64]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_msg,
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 4096},
    }

    resp = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=600)
    resp.raise_for_status()
    raw = resp.json()["message"]["content"].strip()

    code = extract_code(raw)
    return code


def _find_result(local: dict):
    """Return the first CadQuery Workplane object from exec() locals."""
    result = local.get("r")
    if result is None:
        for name in ("result", "part", "shape", "solid", "model", "obj", "output", "body"):
            if name in local and hasattr(local[name], "val"):
                result = local[name]
                print(f"[enhancer] found result in variable '{name}'", flush=True)
                break
    if result is None:
        for name, val in local.items():
            if not name.startswith("_") and hasattr(val, "val"):
                result = val
                print(f"[enhancer] found result in variable '{name}'", flush=True)
                break
    return result


def _strip_fillets_chamfers(code: str) -> str:
    """Remove fillet/chamfer calls and any selector that feeds directly into them.

    Patterns handled:
      .edges(...).fillet(r)     → remove both
      .edges(...).chamfer(d)    → remove both
      .fillet(r)                → remove standalone
      .chamfer(d)               → remove standalone
    """
    # 1. selector(..).fillet/chamfer(..)  — strip the whole selector+operation pair
    stripped = re.sub(
        r"\s*\.(?:edges|faces|vertices|wires|shells|solids)\([^)]*\)\s*\.(?:fillet|chamfer)\([^)]*\)",
        "", code
    )
    # 2. Any remaining standalone .fillet / .chamfer not preceded by a selector
    stripped = re.sub(r"\s*\.fillet\([^)]*\)", "", stripped)
    stripped = re.sub(r"\s*\.chamfer\([^)]*\)", "", stripped)
    return stripped


def execute_and_export(code: str, stl_path: str, py_path: str):
    """Execute CadQuery code and export the result to an STL file.

    If the geometry kernel raises BRep_API (typically from oversized fillets/chamfers),
    automatically retry with those operations stripped out.
    """
    import cadquery as cq
    from cadquery import exporters

    def _run(src: str) -> object:
        local = {}
        exec(src, {"cq": cq}, local)
        result = _find_result(local)
        if result is None:
            print(f"[enhancer] no result var found. locals: {list(local.keys())}", flush=True)
            raise RuntimeError("Corrected code did not produce result variable 'r'")
        return result

    try:
        result = _run(code)
        final_code = code
    except Exception as e:
        if "BRep_API" in str(e) or "command not done" in str(e):
            print(f"[enhancer] BRep_API error — retrying without fillets/chamfers: {e}", flush=True)
            stripped = _strip_fillets_chamfers(code)
            print(f"[enhancer] stripped code:\n{stripped}", flush=True)
            result = _run(stripped)
            final_code = stripped
            print("[enhancer] retry without fillets/chamfers succeeded", flush=True)
        else:
            raise

    exporters.export(result, stl_path)
    with open(py_path, "w") as f:
        f.write(final_code)


def run_enhancement(job_id: str, jobs: dict, results_dir: str,
                    orig_b64: str, corr_b64: str,
                    model: str, ollama_url: str):
    """
    Full enhancement pipeline — runs in a background thread started by app.py.
    Updates jobs[job_id] with progress and result.
    """
    job = jobs[job_id]
    enhanced_stl = f"{results_dir}/{job_id}_enhanced.stl"
    enhanced_py  = f"{results_dir}/{job_id}_enhanced.py"

    try:
        job["enhance_status"]  = "calling_llm"
        job["enhance_message"] = f"Sending to supervisor model ({model})…"

        current_code = (
            job.get("result", {}).get("corrected_code")
            or job.get("result", {}).get("generated_code", "")
        )

        corrected_code = call_ollama_supervisor(
            orig_b64, corr_b64, current_code, model, ollama_url
        )

        job["enhance_status"]  = "executing"
        job["enhance_message"] = "Executing supervisor-corrected code…"

        print(f"[enhancer] full corrected code to execute:\n{'='*60}\n{corrected_code}\n{'='*60}", flush=True)
        execute_and_export(corrected_code, enhanced_stl, enhanced_py)

        # ── Re-apply SmartScaler ─────────────────────────────────────────
        job["enhance_status"]  = "rescaling"
        job["enhance_message"] = "Applying SmartScaler to corrected part…"

        rescaled_stl     = f"{results_dir}/{job_id}_enhanced_rescaled.stl"
        rescaled_py      = f"{results_dir}/{job_id}_enhanced_rescaled.py"
        rescaled_stl_url = None
        scale_info       = {}
        import os
        original_stl = job.get("stl_path", "")
        if original_stl and os.path.exists(original_stl) and os.path.exists(enhanced_stl):
            try:
                from smart_scaler import smart_scale
                scale_info = smart_scale(
                    gt_stl_path=original_stl,
                    gen_stl_path=enhanced_stl,
                    gen_code=corrected_code,
                    corrected_stl_path=rescaled_stl,
                    corrected_py_path=rescaled_py,
                )
                rescaled_stl_url = f"/results/{job_id}_enhanced_rescaled.stl"
            except Exception as se:
                print(f"SmartScaler on enhanced part failed: {se}", flush=True)

        job["enhance_status"]  = "done"
        job["enhance_message"] = "Supervisor correction complete"
        job["enhance_result"]  = {
            "code":             scale_info.get("corrected_code", corrected_code),
            "stl_url":          f"/results/{job_id}_enhanced.stl",
            "rescaled_stl_url": rescaled_stl_url,
            "scale_info":       {k: v for k, v in scale_info.items() if k != "corrected_code"},
        }

    except Exception as e:
        job["enhance_status"]  = "error"
        job["enhance_message"] = str(e)[:300]
