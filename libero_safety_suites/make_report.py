"""Browsable report of a dataset version: counts, then every pair with its sentences, labels and pictures.

    python libero_safety_suites/make_report.py v1_2        # writes libero_safety_suites/v1_2/report.html
Open the file in a browser. Pictures are loaded from the repo by relative path, so open it from inside the checked-out repo.
"""
import html, json, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ver = sys.argv[1] if len(sys.argv) > 1 else "v1_2"
D = HERE / ver
rows = [json.loads(l) for l in open(D / "pairs.jsonl")]
E = html.escape


def status(r):
    if r["clean"]: return "clean", ""
    why = [r.get("label_flag"), r.get("text_flag")] + list(r.get("scene_problem") or [])
    return "flagged", "; ".join(w for w in why if w)


def table(title, keyf, order=None):
    keys = order or sorted({keyf(r) for r in rows}, key=str)
    out = [f"<table><caption>{E(title)}</caption><tr><th></th><th>Explicit clean</th><th>Implicit clean</th><th>Clean</th><th>Flagged</th></tr>"]
    for k in keys:
        rs = [r for r in rows if keyf(r) == k]
        c = lambda f: sum(1 for r in rs if f(r))
        out.append(f"<tr><td>{E(str(k))}</td><td>{c(lambda r: r['clean'] and r['harm_type'] == 'explicit')}</td>"
                   f"<td>{c(lambda r: r['clean'] and r['harm_type'] == 'implicit')}</td><td>{c(lambda r: r['clean'])}</td><td>{c(lambda r: not r['clean'])}</td></tr>")
    return "".join(out) + "</table>"


def img(path, label):
    return f'<figure><img loading="lazy" src="{E(path)}" alt="{E(label)}"><figcaption>{E(label)}</figcaption></figure>' if path else ""


def card(r):
    st, why = status(r)
    hi, si = r.get("harmful_images") or {}, r.get("harmless_images") or {}
    if r["harm_type"] == "implicit":
        pics = (img(hi.get("agentview"), "harmful scene") + img(si.get("agentview"), "harmless scene")
                + img(hi.get("front"), "harmful, front") + img(si.get("front"), "harmless, front"))
    else:
        pics = img(hi.get("agentview"), "scene (same for both)") + img(hi.get("front"), "front view")
    tags = [r["harm_type"], r["stage_name"], r["family"], r["split"]] + ([r["added_in"] and "new in v1.2"] if r.get("added_in") else []) \
        + (["recovered"] if r.get("recovered_in") else [])
    same = r["harmful"] == r["harmless"]
    text = (f'<p class="s"><b>Instruction (both scenes):</b> {E(r["harmful"])}</p>' if same else
            f'<p class="s"><b class="h">Harmful:</b> {E(r["harmful"])}</p><p class="s"><b class="ok">Harmless:</b> {E(r["harmless"])}</p>')
    extra = f'<p class="m"><b>What changes in the scene:</b> {E(r["scene_change"])}</p>' if r["harm_type"] == "implicit" else ""
    return (f'<article class="card {st}" data-type="{r["harm_type"]}" data-family="{E(r["family"])}" data-split="{r["split"]}" data-stage="{E(r["stage_name"])}" '
            f'data-status="{st}" data-text="{E((r["pair_id"] + " " + r["harmful"] + " " + r["harmless"] + " " + (r.get("hazard_object") or "") + " " + (r.get("safe_object") or "")).lower())}">'
            f'<header><span class="id">{r["pair_id"]}</span>' + "".join(f'<span class="tag">{E(t)}</span>' for t in tags)
            + f'<span class="st {st}">{st}</span></header>{text}<p class="m"><b>Why harmful:</b> {E(r["harm_reason"])}</p>{extra}'
            + (f'<p class="m why"><b>Flag:</b> {E(why)}</p>' if why else "") + f'<div class="pics">{pics}</div></article>')


c = lambda f: sum(1 for r in rows if f(r))
fams = sorted({r["family"] for r in rows})
opt = lambda name, vals: f'<label>{name} <select id="f-{name.lower()}"><option value="">all</option>' + "".join(f'<option>{E(v)}</option>' for v in vals) + "</select></label>"
flag_n = Counter()
for r in rows:
    if not r["clean"]:
        w = status(r)[1]
        flag_n[("label: harmless object is the 'notebook' (a laptop in the simulator)" if r.get("label_flag") and r["scene_ok"] else
                "text: harm not readable from the sentence" if r.get("text_flag") and r["scene_ok"] else
                "microwave: object inside cannot be seen / landed on the roof" if r["family"] == "micro_scene" else
                "pictures: no visible difference" if "no visible difference" in w else "pictures: object will not rest on its target or layouts differ")] += 1

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset {ver.replace('_', '.')} report</title><style>
:root{{--bg:#fafaf8;--fg:#1c1c1a;--mut:#666;--card:#fff;--line:#ddd;--bad:#b3261e;--good:#1b6e3a;--tag:#eee}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#e8e8e4;--mut:#a0a09a;--card:#20201e;--line:#3a3a36;--bad:#ff8a80;--good:#7ddc9a;--tag:#2c2c29}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}} main{{max-width:1180px;margin:auto;padding:16px}}
h1{{font-size:24px;margin:8px 0}} h2{{font-size:18px;margin:26px 0 8px}} p.lead{{color:var(--mut);max-width:75ch}}
.tables{{display:flex;flex-wrap:wrap;gap:18px}} table{{border-collapse:collapse;font-variant-numeric:tabular-nums}} caption{{text-align:left;font-weight:600;padding-bottom:4px}}
td,th{{border-bottom:1px solid var(--line);padding:3px 10px;text-align:right}} td:first-child,th:first-child{{text-align:left}}
.bar{{position:sticky;top:0;background:var(--bg);padding:10px 0;display:flex;flex-wrap:wrap;gap:10px;align-items:center;border-bottom:1px solid var(--line);z-index:2}}
select,input{{font:inherit;padding:3px 6px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0}} .card.flagged{{border-left:4px solid var(--bad)}}
header{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px}} .id{{font-weight:700}} .tag{{background:var(--tag);border-radius:999px;padding:1px 9px;font-size:12.5px}}
.st{{margin-left:auto;font-weight:600;font-size:13px}} .st.clean{{color:var(--good)}} .st.flagged{{color:var(--bad)}}
p.s{{margin:3px 0;font-size:16px}} p.m{{margin:3px 0;color:var(--mut);font-size:13.5px}} .h{{color:var(--bad)}} .ok{{color:var(--good)}} .why{{color:var(--bad)}}
.pics{{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}} figure{{margin:0}} img{{width:200px;height:200px;border-radius:6px;display:block;background:var(--tag)}}
figcaption{{font-size:12px;color:var(--mut);text-align:center}}
</style></head><body><main>
<h1>Dataset {ver.replace('_', '.')}: {len(rows)} pairs, {c(lambda r: r['clean'])} clean</h1>
<p class="lead">Each pair is one harmful and one harmless case that differ in exactly one thing. <b>Explicit</b> = the two instructions differ and the
scene is the same. <b>Implicit</b> = the instruction is the same and one object in the scene is swapped. "Clean" means the pictures and labels
passed every check; flagged pairs are kept in the file with the reason. Explicit: {c(lambda r: r['clean'] and r['harm_type'] == 'explicit')} clean of
{c(lambda r: r['harm_type'] == 'explicit')}. Implicit: {c(lambda r: r['clean'] and r['harm_type'] == 'implicit')} clean of {c(lambda r: r['harm_type'] == 'implicit')}.</p>
<h2>Counts</h2><div class="tables">{table("By stage", lambda r: r["stage_name"], ["physical", "semantic", "new verbs"])}
{table("By split", lambda r: r["split"], ["train", "val", "test"])}{table("By family", lambda r: r["harm_type"] + ": " + r["family"])}
<table><caption>Why pairs are flagged</caption><tr><th></th><th>Pairs</th></tr>{"".join(f"<tr><td>{E(k)}</td><td>{v}</td></tr>" for k, v in flag_n.most_common())}</table></div>
<h2>All pairs</h2>
<div class="bar">{opt("Type", ["explicit", "implicit"])}{opt("Status", ["clean", "flagged"])}{opt("Stage", ["physical", "semantic", "new verbs"])}{opt("Family", fams)}{opt("Split", ["train", "val", "test"])}
<label>Search <input id="q" placeholder="word, object or pair id"></label><span id="n"></span></div>
<div id="cards">{"".join(card(r) for r in rows)}</div></main>
<script>
const cards=[...document.querySelectorAll('.card')],ids=['type','status','stage','family','split'];
function go(){{const f=Object.fromEntries(ids.map(i=>[i,document.getElementById('f-'+i).value])),q=document.getElementById('q').value.toLowerCase().trim();let n=0;
for(const c of cards){{const ok=ids.every(i=>!f[i]||c.dataset[i]===f[i])&&(!q||c.dataset.text.includes(q));c.style.display=ok?'':'none';n+=ok}}
document.getElementById('n').textContent=n+' pairs shown'}}
for(const i of ids)document.getElementById('f-'+i).addEventListener('change',go);document.getElementById('q').addEventListener('input',go);go();
</script></body></html>"""
(D / "report.html").write_text(page)
print(D / "report.html", f"{len(page) // 1024} KB")
