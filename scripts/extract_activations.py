"""Extract per-layer residual-stream activations for every sample in probe_table.jsonl.

All models go through ONE code path, openpi's pi0.5 prefix forward (PyTorch port):
  [SigLIP tokens of each view] + [<bos> instruction "\n"] -> Gemma-2B backbone, full (bidirectional) attention over the prefix,
  no action expert. For PaliGemma the HF checkpoint is run through the same function, so inputs are identical across models.
Inputs (defaults): agentview + wrist renders (256 image tokens each, 224x224, pixels in [-1, 1]), instruction lowercased
(LIBERO prompts are lowercase), tokenized as encode(text, add_bos=True) + encode("\n") exactly like openpi's pi0.5 LIBERO policy
(discrete_state_input=False, so no robot state in the prompt; pi05_libero ignores state entirely).
Saved for each of the L+1 hidden states (embeddings + every block, pre final-norm):
  last_instr : last instruction token        mean_instr : mean over instruction tokens
  mean_img   : mean over all image tokens     last_tok   : final position (the "\n" token)

Needs the openpi environment with its patched transformers (see scripts/setup_pod.sh). Usage:
  python scripts/extract_activations.py --model paligemma --out cache/acts/paligemma.npz
  python scripts/extract_activations.py --model pi05_base --ckpt cache/ckpt/pi05_base_pytorch --out cache/acts/pi05_base.npz
  python scripts/extract_activations.py --model pi05_base --ckpt ... --check 5 --limit 5 --out cache/acts/smoke.npz   # compare vs openpi's own prefix path
  python scripts/extract_activations.py ... --no-text        # image-only control
Rows keep the order of probe_table.jsonl; the npz stores pair_id / label / split / group for scripts/train_probes.py.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_rows(table, variant, views, limit):
    rows = [json.loads(l) for l in open(table)]
    if limit: rows = rows[:limit]
    for r in rows:
        imgs = r["images"] if variant == "orig" else r["variants"][variant]
        r["image_paths"] = [str(ROOT / imgs[v]) for v in views]
    return rows


# ----------------------------------------------------------------------------- models
def load_backbone(model, ckpt, hf_id, device):
    """-> (embed_image, embed_tokens, language_model, pi0_model_or_None). language_model is the Gemma decoder of the VLM."""
    import torch
    if model == "paligemma":
        from transformers import PaliGemmaForConditionalGeneration
        pg = PaliGemmaForConditionalGeneration.from_pretrained(hf_id, torch_dtype=torch.bfloat16).to(device).eval()
        return pg.model.get_image_features, pg.model.language_model.embed_tokens, pg.model.language_model, None
    from openpi.training import config as _config
    cfg = _config.get_config("pi05_libero")  # same architecture for pi05_base / pi05_libero / pi05_libero_safety; only weights differ
    pi0 = cfg.model.load_pytorch(cfg, str(Path(ckpt) / "model.safetensors"))
    pi0.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    pi0 = pi0.to(device).eval()
    pwe = pi0.paligemma_with_expert
    return pwe.embed_image, pwe.embed_language_tokens, pwe.paligemma.language_model, pi0


def make_runner(model, ckpt, hf_id, device):
    import torch
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import image_tools
    embed_image, embed_tokens, lm, pi0 = load_backbone(model, ckpt, hf_id, device)
    sp = PaligemmaTokenizer(max_len=200)._tokenizer
    layers = lm.layers
    d = lm.config.hidden_size

    def to_pixels(img):
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)[None].to(device)  # [1,H,W,3] in [-1,1]
        if x.shape[1:3] != (224, 224):
            x = image_tools.resize_with_pad_torch(x, 224, 224)
            if x.dim() == 3: x = x[None]  # openpi squeezes a batch of 1
        return x.permute(0, 3, 1, 2)  # [1,3,224,224]

    def tokens(text):
        return sp.encode(text, add_bos=True) + sp.encode("\n")

    @torch.no_grad()
    def prefix(images, text):
        img = [embed_image(to_pixels(im)) for im in images]  # each [1,256,d]
        n_img = sum(e.shape[1] for e in img)
        embs, ids = list(img), []
        if text is not None:
            ids = tokens(text)
            t = torch.tensor([ids], device=device)
            embs.append(embed_tokens(t) * math.sqrt(d))
        x = torch.cat(embs, 1)
        return x, n_img, ids

    @torch.no_grad()
    def run(images, text):
        """-> dict of [L+1, d] float32 arrays; also the final-norm output at the last position (for --check)."""
        x, n_img, ids = prefix(images, text)
        T = x.shape[1]
        mask = torch.zeros(1, 1, T, T, device=device, dtype=x.dtype)  # full attention over the prefix, as in pi0.5 / PaliGemma prefix-LM
        pos = torch.arange(T, device=device)[None]
        hs = []
        h0 = layers[0].register_forward_pre_hook(lambda m, a, kw: hs.append(kw.get("hidden_states", a[0] if a else None)), with_kwargs=True)
        hk = [l.register_forward_hook(lambda m, a, o: hs.append(o[0] if isinstance(o, tuple) else o)) for l in layers]
        try:
            lm.config._attn_implementation = "eager"
            out = lm(inputs_embeds=x, attention_mask=mask, position_ids=pos, use_cache=False)
        finally:
            h0.remove(); [h.remove() for h in hk]
        H = torch.stack([h[0] for h in hs]).float()  # [L+1, T, d]
        assert H.shape[0] == len(layers) + 1, H.shape
        feats = {"mean_img": H[:, :n_img].mean(1), "last_tok": H[:, -1]}
        if text is not None:
            k = len(ids) - 2  # instruction tokens = ids minus <bos> and "\n"
            instr = slice(n_img + 1, n_img + 1 + k)
            feats["last_instr"] = H[:, instr.stop - 1]
            feats["mean_instr"] = H[:, instr].mean(1)
        final = out.last_hidden_state[0, -1].float()
        return {k: v.cpu().numpy() for k, v in feats.items()}, final.cpu().numpy(), ids

    def reference(images, text):
        """openpi's own prefix path (pi0 models only): 3 image slots (3rd masked), padded tokenizer, its own masks/positions.
        -> final-norm hidden state at the last real text token."""
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        tok, tmask = PaligemmaTokenizer(max_len=200).tokenize(text)
        px = [to_pixels(im) for im in images]
        px.append(torch.zeros_like(px[0]))
        masks = [torch.ones(1, dtype=torch.bool, device=device)] * len(images) + [torch.zeros(1, dtype=torch.bool, device=device)]
        lt = torch.tensor(tok[None], device=device); lm_ = torch.tensor(tmask[None], device=device)
        with torch.no_grad():
            e, pm, am = pi0.embed_prefix(px, masks, lt, lm_)
            m4 = pi0._prepare_attention_masks_4d(make_att_2d_masks(pm, am))
            pi0.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
            (o, _), _ = pi0.paligemma_with_expert.forward(attention_mask=m4, position_ids=torch.cumsum(pm, 1) - 1,
                                                          past_key_values=None, inputs_embeds=[e, None], use_cache=False)
        last = int(pm[0].nonzero().max())
        return o[0, last].float().cpu().numpy(), [int(t) for t, m in zip(tok, tmask) if m]

    return run, (reference if pi0 is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["paligemma", "pi05_base", "pi05_libero", "pi05_libero_safety"], required=True)
    ap.add_argument("--ckpt", help="converted PyTorch checkpoint dir (pi0.5 models)")
    ap.add_argument("--hf-id", default="google/paligemma-3b-pt-224", help="PaliGemma checkpoint")
    ap.add_argument("--table", default=str(ROOT / "results/dataset_review/probe_table.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="agentview,wrist", help="comma list of agentview, wrist, front")
    ap.add_argument("--variant", default="orig", help="orig, v1 or v2 (lighting/camera variants)")
    ap.add_argument("--no-text", action="store_true", help="image-only control: no text tokens")
    ap.add_argument("--keep-case", action="store_true", help="do not lowercase instructions")
    ap.add_argument("--check", type=int, default=0, help="compare the first N rows against openpi's own prefix forward (pi0 models)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.model != "paligemma" and not a.ckpt: ap.error("--ckpt is required for pi0.5 models")

    from PIL import Image
    views = a.views.split(",")
    rows = load_rows(a.table, a.variant, views, a.limit)
    run, reference = make_runner(a.model, a.ckpt, a.hf_id, a.device)

    acc, t0 = {}, time.time()
    for i, r in enumerate(rows):
        text = None if a.no_text else (r["instruction"] if a.keep_case else r["instruction"].lower())
        imgs = [Image.open(p).convert("RGB") for p in r["image_paths"]]
        f, final, ids = run(imgs, text)
        if i < a.check and reference is not None and text is not None:
            ref, ref_ids = reference(imgs, text)
            cos = float(final @ ref / (np.linalg.norm(final) * np.linalg.norm(ref)))
            rel = float(np.linalg.norm(final - ref) / np.linalg.norm(ref))
            print(f"check row {i}: tokens match={ids == ref_ids}  cos={cos:.5f}  rel_err={rel:.4f}  {text!r}", flush=True)
        for k, v in f.items():
            acc.setdefault(k, []).append(v.astype(np.float16))
        if i % 50 == 0: print(f"{i}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out, **{k: np.stack(v) for k, v in acc.items()},  # each [N, L+1, d]
        pair_id=np.array([r["pair_id"] for r in rows]), label=np.array([r["label"] for r in rows]),
        split=np.array([r["split"] for r in rows]), group=np.array([r["group"] for r in rows]),
        meta=json.dumps(dict(model=a.model, ckpt=a.ckpt, hf_id=a.hf_id, views=views, variant=a.variant, no_text=a.no_text,
                             lowercase=not a.keep_case, forward="openpi_prefix_full_attention")))
    print("saved", a.out, {k: np.stack(v).shape for k, v in acc.items()})


if __name__ == "__main__":
    sys.exit(main())
