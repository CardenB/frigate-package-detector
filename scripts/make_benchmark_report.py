#!/usr/bin/env python3
"""Side-by-side benchmark report from report/benchmarks/<schema>/<run>/.

Reads every run's manifest.json + per_class.json and emits:
  report/benchmark_report.md    — overall + per-class table (git/terminal)
  report/benchmark_report.html  — pick metric, click-sort, per-class deltas vs a
                                  baseline, and links into each run's gallery to
                                  see top failures/positives for a class.

Models are columns, classes are rows. Baseline = run whose name contains
'stock' (else earliest); override with --baseline. Runs across schema versions
are all loaded (each self-describes via schema_version).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bench
from common import REPO

REPORT_DIR = REPO / "report"
METRICS = ["map50_95", "map50", "precision", "recall"]


def load_runs() -> list[dict]:
    runs = []
    for manifest in sorted(bench.bench_root().glob("*/*/manifest.json")):
        run = manifest.parent
        m = json.loads(manifest.read_text())
        pc_path = run / "per_class.json"
        m["classes"] = json.loads(pc_path.read_text()) if pc_path.exists() else []
        gallery = run / "gallery.html"
        m["_gallery"] = (str(gallery.relative_to(REPORT_DIR))
                         if gallery.exists() else None)
        cm = m.get("confusion_matrix")
        cm_png = run / cm["png_normalized"] if cm else None
        m["_cm"] = (str(cm_png.relative_to(REPORT_DIR))
                    if cm_png and cm_png.exists() else None)
        m["_run_id"] = run.name
        runs.append(m)
    return runs


def pick_baseline(runs, override):
    if override:
        return override
    for r in runs:
        if "stock" in r["name"].lower():
            return r["_run_id"]
    return min(runs, key=lambda r: r.get("timestamp", ""))["_run_id"]


def build_payload(runs, baseline_id):
    base = next((r for r in runs if r["_run_id"] == baseline_id), runs[0])
    base_support = {c["name"]: c.get("support", 0) for c in base["classes"]}
    names = set()
    for r in runs:
        names.update(c["name"] for c in r["classes"])
    order = sorted(names, key=lambda n: (-base_support.get(n, 0), n))

    by_run_class = {r["_run_id"]: {c["name"]: c for c in r["classes"]} for r in runs}

    classes = []
    for name in order:
        e = {"name": name, "support": base_support.get(name, 0), "byModel": {}}
        for r in runs:
            c = by_run_class[r["_run_id"]].get(name)
            if c:
                cell = {m: c.get(m) for m in METRICS}
                cell["ex"] = bool(c.get("examples") and any(
                    c["examples"].get(cat) for cat in bench.CATEGORIES))
                e["byModel"][r["_run_id"]] = cell
        classes.append(e)

    models = [{
        "id": r["_run_id"], "name": r["name"], "model": r.get("model", ""),
        "params": r.get("model_params", 0), "imgsz": r.get("imgsz"),
        "instances": r.get("num_instances", 0), "timestamp": r.get("timestamp", ""),
        "schema": r.get("schema_version", "?"), "gallery": r["_gallery"],
        "cm": r["_cm"], "split": r.get("split", "val"),
        "is_baseline": r["_run_id"] == baseline_id,
        **{m: r["overall"].get(m) for m in METRICS},
    } for r in runs]
    return {"models": models, "baseline": baseline_id, "classes": classes,
            "metrics": METRICS}


def fmt(v):
    return "–" if v is None else f"{v:.3f}"


def write_markdown(payload):
    models = payload["models"]
    base = next(m for m in models if m["id"] == payload["baseline"])
    L = ["# Benchmark report", "",
         f"Baseline: **{base['name']}** (`{base['id']}`) · per-class metric: **mAP50-95**", "",
         "## Overall", "",
         "| model | run | params | imgsz | mAP50-95 | mAP50 | P | R | gallery |",
         "|" + "---|" * 9]
    for m in models:
        star = " ⭐" if m["is_baseline"] else ""
        g = f"[gallery]({m['gallery']})" if m["gallery"] else "–"
        L.append(f"| {m['name']}{star} | `{m['id']}` | {m['params']/1e6:.1f}M | {m['imgsz']} "
                 f"| {fmt(m['map50_95'])} | {fmt(m['map50'])} | {fmt(m['precision'])} "
                 f"| {fmt(m['recall'])} | {g} |")
    L += ["", "## Per-class mAP50-95 (Δ vs baseline)", ""]
    mids = [m["id"] for m in models]
    mnames = [m["name"] for m in models]
    L.append("| class | support | " + " | ".join(mnames) + " |")
    L.append("|" + "---|" * (2 + len(mids)))
    for c in payload["classes"]:
        cells = []
        bv = c["byModel"].get(base["id"], {}).get("map50_95")
        for mid in mids:
            cell = c["byModel"].get(mid)
            if not cell or cell.get("map50_95") is None:
                cells.append("–"); continue
            v = cell["map50_95"]
            if mid == base["id"] or bv is None:
                cells.append(f"{v:.3f}")
            else:
                d = v - bv
                cells.append(f"{v:.3f} ({'+' if d>=0 else ''}{d:.3f})")
        L.append(f"| {c['name']} | {c['support']} | " + " | ".join(cells) + " |")
    if any(m.get("cm") for m in models):
        L += ["", "## Confusion matrices (normalized, columns = true class)", ""]
        for m in models:
            if m.get("cm"):
                L.append(f"- **{m['name']}** (`{m['split']}` split): "
                         f"[{m['cm']}]({m['cm']})")
    out = REPORT_DIR / "benchmark_report.md"
    out.write_text("\n".join(L) + "\n")
    return out


HTML_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>Benchmark report</title>
<style>
 body{font:14px system-ui,sans-serif;margin:24px;color:#1a1a1a}
 h1,h2{margin:.4em 0}
 table{border-collapse:collapse;margin:8px 0 24px}
 th,td{border:1px solid #ddd;padding:4px 8px;text-align:right}
 th{background:#f4f4f4;cursor:pointer;user-select:none;position:sticky;top:0}
 td.name,th.name{text-align:left}
 tr:nth-child(even){background:#fafafa}
 .pos{color:#0a7d28}.neg{color:#c0271a}
 .base{font-weight:600}
 .controls{margin:12px 0}
 .bar{display:inline-block;height:9px;background:#bcdffb;vertical-align:middle;margin-left:6px;border-radius:2px}
 .muted{color:#888}a{color:#1564c0;text-decoration:none}a:hover{text-decoration:underline}
 .cm{display:inline-block;vertical-align:top;margin:8px 14px 8px 0;text-align:center}
 .cm img{width:400px;border:1px solid #ccc;display:block}
</style></head><body>
<h1>Benchmark report</h1>
<div class=muted id=meta></div>
<h2>Overall</h2><div id=summary></div>
<h2>Confusion matrices <span class=muted style="font-size:12px">(normalized; columns = true class — click to enlarge)</span></h2>
<div id=confmats></div>
<h2>Per-class drill-down</h2>
<div class=controls>Metric:
 <select id=metric></select>
 &nbsp; Baseline: <b id=baselabel></b>
 &nbsp;<span class=muted>(click a header to sort; deltas vs baseline; 🔍 links to that model's failures/positives for the class)</span>
</div>
<div id=perclass></div>
<script>
const DATA = __DATA__;
const ML = {map50_95:"mAP50-95", map50:"mAP50", precision:"Precision", recall:"Recall"};
const safe = s => s.replace(/[ \\/]/g,"_");
let metric="map50_95", sortCol="support", sortDir=-1;
const el=(t,c,h)=>{const e=document.createElement(t);if(c)e.className=c;if(h!=null)e.innerHTML=h;return e;};
const galleryFor = id => (DATA.models.find(m=>m.id===id)||{}).gallery;

function renderSummary(){
 const t=el("table"),head=["model","run","params","imgsz",...DATA.metrics.map(x=>ML[x]),"gallery"];
 const tr=el("tr");head.forEach((h,i)=>tr.appendChild(el("th",i===0?"name":null,h)));t.appendChild(tr);
 DATA.models.forEach(r=>{const row=el("tr");
  row.appendChild(el("td","name"+(r.is_baseline?" base":""),r.name+(r.is_baseline?" ⭐":"")));
  row.appendChild(el("td","muted",r.id));
  row.appendChild(el("td",null,(r.params/1e6).toFixed(1)+"M"));
  row.appendChild(el("td",null,r.imgsz));
  DATA.metrics.forEach(x=>row.appendChild(el("td",null,r[x]==null?"–":r[x].toFixed(4))));
  row.appendChild(el("td",null,r.gallery?`<a href="${r.gallery}">open</a>`:"–"));
  t.appendChild(row);});
 const d=document.getElementById("summary");d.innerHTML="";d.appendChild(t);
}
function renderConfmats(){
 const d=document.getElementById("confmats");d.innerHTML="";
 const withCm=DATA.models.filter(m=>m.cm);
 if(!withCm.length){d.innerHTML="<span class=muted>No confusion matrices yet — re-run eval to generate them.</span>";return;}
 withCm.forEach(m=>{const fig=el("div","cm");
  fig.innerHTML=`<a href="${m.cm}" target=_blank><img src="${m.cm}"></a>`+
   `<div>${m.name}${m.is_baseline?" ⭐":""} <span class=muted>(${m.split})</span></div>`;
  d.appendChild(fig);});
}
function renderPerClass(){
 const models=DATA.models.map(m=>m.id);
 const maxv=Math.max(...DATA.classes.flatMap(c=>models.map(m=>(c.byModel[m]||{})[metric]||0)),0.001);
 const rows=[...DATA.classes];
 rows.sort((a,b)=>{
   if(sortCol==="name")return sortDir*(a.name<b.name?-1:a.name>b.name?1:0);
   let av,bv;
   if(sortCol==="support"){av=a.support;bv=b.support;}
   else{av=(a.byModel[sortCol]||{})[metric]??-1;bv=(b.byModel[sortCol]||{})[metric]??-1;}
   return sortDir*(av-bv);
 });
 const t=el("table"),tr=el("tr");
 const cols=[["name","class"],["support","support"],...DATA.models.map(m=>[m.id,m.name])];
 cols.forEach(([key,lbl],i)=>{const th=el("th",i===0?"name":null,lbl+(sortCol===key?(sortDir<0?" ▾":" ▴"):""));
   th.onclick=()=>{sortDir=(sortCol===key)?-sortDir:(key==="name"?1:-1);sortCol=key;renderPerClass();};tr.appendChild(th);});
 t.appendChild(tr);
 rows.forEach(c=>{const row=el("tr");
  row.appendChild(el("td","name",c.name));
  row.appendChild(el("td",null,c.support));
  models.forEach(mid=>{const cell=c.byModel[mid]||{};const v=cell[metric];
   const td=el("td",mid===DATA.baseline?"base":null);
   let html=(v==null?"–":v.toFixed(3));
   if(v!=null){
    if(mid!==DATA.baseline){const bv=(c.byModel[DATA.baseline]||{})[metric];
      if(bv!=null){const d=v-bv;html+=` <span class=${d>=0?"pos":"neg"}>${d>=0?"+":""}${d.toFixed(3)}</span>`;}}
    const g=galleryFor(mid);
    if(cell.ex&&g)html+=` <a title="failures/positives" href="${g}#${safe(c.name)}">🔍</a>`;
    html+=`<span class=bar style="width:${Math.round(40*v/maxv)}px"></span>`;
   }
   td.innerHTML=html;row.appendChild(td);});
  t.appendChild(row);});
 const d=document.getElementById("perclass");d.innerHTML="";d.appendChild(t);
}
function init(){
 document.getElementById("meta").textContent=DATA.models.length+" run(s) · "+DATA.classes.length+" classes";
 document.getElementById("baselabel").textContent=DATA.baseline;
 const sel=document.getElementById("metric");
 DATA.metrics.forEach(x=>{const o=el("option");o.value=x;o.textContent=ML[x];sel.appendChild(o);});
 sel.onchange=()=>{metric=sel.value;renderPerClass();};
 renderSummary();renderConfmats();renderPerClass();
}
init();
</script></body></html>"""


def write_html(payload):
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload))
    out = REPORT_DIR / "benchmark_report.html"
    out.write_text(html)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=None, help="run id (folder name) to diff against")
    args = ap.parse_args()
    runs = load_runs()
    if not runs:
        raise SystemExit(f"No runs under {bench.bench_root()}. Run eval_baseline.py first.")
    baseline = pick_baseline(runs, args.baseline)
    payload = build_payload(runs, baseline)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md = write_markdown(payload)
    html = write_html(payload)
    print(f"[report] {len(runs)} run(s), baseline={baseline}")
    print(f"[report] markdown -> {md}")
    print(f"[report] html     -> {html}")


if __name__ == "__main__":
    main()
