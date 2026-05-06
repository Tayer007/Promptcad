"""
inference.py - called by app.py:
  python inference.py <input.stl> <out.stl> <out.py> <corrected.stl> <corrected.py> <result.json>
"""

import sys, os, json, torch, torch.nn as nn, numpy as np
from transformers import Qwen2ForCausalLM, Qwen2Model, PreTrainedModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
import cadquery as cq
from cadquery import exporters
import trimesh, trimesh.sample

# ── Add project dir to path for smart_scaler ──
sys.path.insert(0, os.path.dirname(__file__))
from smart_scaler import smart_scale

# ── FPS ───────────────────────────────────────
def farthest_point_sampling(points, n=256):
    idx = np.zeros(n, dtype=int)
    dist = np.full(len(points), np.inf)
    idx[0] = np.random.randint(len(points))
    for i in range(1, n):
        d = np.sum((points - points[idx[i-1]])**2, axis=1)
        dist = np.minimum(dist, d)
        idx[i] = np.argmax(dist)
    return points[idx]

# ── Architecture ──────────────────────────────
class FourierPointEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.projection = nn.Linear(51, hidden_size)

    def forward(self, points):
        points = points.to(self.projection.weight.dtype)
        x = (points.unsqueeze(-1) * self.frequencies.to(points.dtype)).view(*points.shape[:-1], -1)
        x = torch.cat((points, x.sin(), x.cos()), dim=-1)
        return self.projection(x)

class CADRecode(Qwen2ForCausalLM):
    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._tied_weights_keys = []
        torch.set_default_dtype(torch.float32)
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        torch.set_default_dtype(torch.bfloat16)

    @property
    def all_tied_weights_keys(self):
        return {}

    def tie_weights(self, **kwargs):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids=None, attention_mask=None, point_cloud=None,
                position_ids=None, past_key_values=None, inputs_embeds=None,
                labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, cache_position=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if past_key_values is None or past_key_values.get_seq_length() == 0:
            assert inputs_embeds is None
            inputs_embeds = self.model.embed_tokens(input_ids)
            point_embeds = self.point_encoder(point_cloud).bfloat16()
            inputs_embeds[attention_mask == -1] = point_embeds.reshape(-1, point_embeds.shape[2])
            attention_mask[attention_mask == -1] = 1
            input_ids = None
            position_ids = None
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, cache_position=cache_position)
        logits = self.lm_head(outputs[0]).float()
        return CausalLMOutputWithPast(loss=None, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs['point_cloud'] = kwargs['point_cloud']
        return model_inputs

# ── Point cloud ───────────────────────────────
def get_point_cloud(mesh, n=256):
    mesh = mesh.copy()
    mesh.apply_translation(-(mesh.bounds[0]+mesh.bounds[1])/2)
    mesh.apply_scale(2.0/max(mesh.extents))
    vertices, _ = trimesh.sample.sample_surface(mesh, 8192)
    return farthest_point_sampling(np.array(vertices, dtype=np.float32), n)

def chamfer_distance(pc1, pc2):
    p1 = torch.tensor(pc1).float().cuda()
    p2 = torch.tensor(pc2).float().cuda()
    diff1 = ((p1.unsqueeze(1) - p2.unsqueeze(0))**2).sum(-1).min(1).values.mean()
    diff2 = ((p2.unsqueeze(1) - p1.unsqueeze(0))**2).sum(-1).min(1).values.mean()
    return (diff1 + diff2).item()

# ── Main ──────────────────────────────────────
def main():
    stl_in        = sys.argv[1]
    stl_out       = sys.argv[2]
    py_out        = sys.argv[3]
    corrected_stl = sys.argv[4]
    corrected_py  = sys.argv[5]
    result_json   = sys.argv[6]

    print(f"Loading STL: {stl_in}", flush=True)
    input_mesh = trimesh.load_mesh(stl_in)

    print("Loading model...", flush=True)
    HF_REPO = os.environ.get('HF_REPO', '')
    if HF_REPO:
        hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        model_cache = os.path.join(hf_home, 'hub', 'models--' + HF_REPO.replace('/', '--'))
        refs_main = os.path.join(model_cache, 'refs', 'main')
        if os.path.exists(refs_main):
            with open(refs_main) as f:
                latest = f.read().strip()
            MODEL_PATH = os.path.join(model_cache, 'snapshots', latest)
            local_only = True
        else:
            MODEL_PATH = HF_REPO
            local_only = False
    else:
        MODEL_PATH = r'C:\Users\fou0\Downloads\promptcad_2\promptcad\model'
        local_only = True

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, pad_token='<|im_end|>', padding_side='left',
        local_files_only=local_only)
    model = CADRecode.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        local_files_only=local_only).eval()
    print("Model loaded.", flush=True)

    im_start = tokenizer('<|im_start|>')['input_ids'][0]
    ref_pc = get_point_cloud(input_mesh)
    candidates = []

    for i in range(10):
        pc = get_point_cloud(input_mesh)
        input_ids_list = [tokenizer.pad_token_id]*256 + [im_start]
        attn_mask = [-1]*256 + [1]

        with torch.no_grad():
            out = model.generate(
                input_ids=torch.tensor(input_ids_list).unsqueeze(0).cuda(),
                attention_mask=torch.tensor(attn_mask).unsqueeze(0).cuda(),
                point_cloud=torch.tensor(pc).unsqueeze(0).cuda().to(torch.bfloat16),
                max_new_tokens=768, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)

        result = tokenizer.batch_decode(out)[0]
        begin = result.rfind('<|im_start|>')
        if begin == -1: continue
        generated = result[begin + 12:]
        end = generated.find('<|endoftext|>')
        if end != -1: generated = generated[:end]
        generated = generated.strip()
        print(f"Sample {i+1}: {generated[:80]}", flush=True)
        candidates.append(generated)

    # Pick best by Chamfer distance
    best_code, best_cd = None, float('inf')
    for code in candidates:
        try:
            local = {}
            exec(code, {'cq': cq}, local)
            r = local.get('r')
            if r is None: continue
            verts, faces = r.val().tessellate(0.001, 0.1)
            verts = np.array([(v.x,v.y,v.z) for v in verts], dtype=np.float32)
            gen_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            gen_pc, _ = trimesh.sample.sample_surface(gen_mesh, 8192)
            cd = chamfer_distance(ref_pc, gen_pc)
            if cd < best_cd:
                best_cd = cd
                best_code = code
        except Exception as e:
            print(f"Candidate failed: {e}", flush=True)

    print(f"\n=== RESULT (CD={best_cd:.4f}) ===", flush=True)
    print(best_code, flush=True)

    # Save raw generated STL + code
    scale_info = {}
    if best_code:
        try:
            local = {}
            exec(best_code, {'cq': cq}, local)
            r = local.get('r')
            if r:
                exporters.export(r, stl_out)
                print(f"Saved generated STL: {stl_out}", flush=True)
            with open(py_out, 'w') as f:
                f.write(best_code)
        except Exception as e:
            print(f"Save failed: {e}", flush=True)

        # ── Smart Scaler ──────────────────────────────
        print("\nRunning SmartScaler...", flush=True)
        try:
            scale_info = smart_scale(
                gt_stl_path=stl_in,
                gen_stl_path=stl_out,
                gen_code=best_code,
                corrected_stl_path=corrected_stl,
                corrected_py_path=corrected_py,
            )
            print(f"Scale applied: x={scale_info['scale_x']:.3f} y={scale_info['scale_y']:.3f} z={scale_info['scale_z']:.3f}", flush=True)
        except Exception as e:
            print(f"SmartScaler failed: {e}", flush=True)
            scale_info = {'error': str(e)}

    # Write result JSON for app.py to read
    output = {
        'chamfer_distance': best_cd if best_cd != float('inf') else None,
        'generated_code': best_code,
        'scale_info': {k: v for k, v in scale_info.items() if k != 'corrected_code'},
        'corrected_code': scale_info.get('corrected_code', best_code),
    }
    with open(result_json, 'w') as f:
        json.dump(output, f)
    print(f"Result written to {result_json}", flush=True)

if __name__ == '__main__':
    main()
