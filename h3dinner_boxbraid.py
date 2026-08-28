#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,sys,time,urllib.request
from datetime import datetime
from pathlib import Path

COMFY_URL="http://127.0.0.1:8188"
WIDTH,HEIGHT=608,352
DURATION_SECONDS=5.0
STEPS=20
UNET="minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE="minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE="minimax_h3_audio_vae_fp32.safetensors"
OUTPUT_ROOT="video/h3boxbraid"
MIN_FREE_VRAM_MIB=4500
POLL_SECONDS=10
JOB_TIMEOUT_SECONDS=7200

COMMON="""The recurring object is a simple rectangular industrial box, roughly suitcase-sized, with a bold black triangle symbol centered on one side. The triangle remains obvious. The box may change minor surface details but remains recognizably the same kind of object. Dead-serious classified institutional procedure. Native H3 audio is essential. Speech, when present, is rhythmic expressive synthetic gibberish and intentionally unintelligible. Machine sounds correspond to visible actions. No readable generated text, subtitles, logos, or watermark."""

JOBS=[
{"name":"01_box_blank_intake","stage":"INTAKE","seed":5101,"prompt":"""1950s atomic-age classified intake laboratory. A pristine rectangular industrial box with a bold black triangle symbol sits open on a metal table. It has NO inspection stamps or stickers yet. A serious female technician inserts a blank rectangular white card or blank photo sheet into the box, closes the lid, and slides the box toward a large analog intake machine. Vacuum tubes, gauges, relays and cyan input lamps surround the station. One clear machine chirp sounds as the box is accepted into the process. The technician gives a short formal burst of synthetic gibberish."""+COMMON},
{"name":"02_box_one_stamp_routing","stage":"ROUTING","seed":5102,"prompt":"""1970s classified routing department. A rectangular industrial box with a bold black triangle symbol is clearly visible. It now has EXACTLY ONE large mismatched inspection stamp or sticker. A stern operator places the box into a physical routing cradle surrounded by patch panels, oscilloscopes, rotary switches and blinking lamps. Cyan signal paths enter the cradle, then amber routing lamps illuminate as the operator flips one large switch. A crisp relay clack and short electronic chirp synchronize with the visible route change."""+COMMON},
{"name":"03_box_two_stamps_generation","stage":"GENERATION","seed":5103,"prompt":"""Near-future generation chamber built partly from glossy modern equipment and partly from bulky 1970s machinery. A rectangular industrial box with a bold black triangle symbol sits inside the central generation bay. It now has TWO visibly different accumulated inspection stamps or stickers. Amber processing lights sequence around the box. Cyan scope traces pulse. Heavy relays clunk in rhythm. Near the end, several green validation lamps illuminate one by one. A strange synthetic presenter calmly watches and delivers authoritative unintelligible gibberish."""+COMMON},
{"name":"04_box_three_stamps_output","stage":"OUTPUT","seed":5104,"prompt":"""1980s retro-futurist classified output room. A rectangular industrial box with a bold black triangle symbol emerges from a large absurdly overbuilt mechanical output machine. The box now carries THREE clearly visible mismatched accumulated inspection stamps or stickers. The triangle remains visible. The machine makes one enormous synchronized mechanical STOMP as the box moves forward, then pauses. A dead-serious presenter waits beside the machine, briefly checks a wristwatch with restrained impatience, then picks up the stamped box and faces camera. End with the presenter holding the box clearly enough to see the triangle and accumulated stamps."""+COMMON},
]

REQ={"VAELoader","VAEDecodeAudio","VAEDecode","KSamplerSelect","BasicScheduler","SamplerCustomAdvanced","BasicGuider","UNETLoader","CLIPLoader","RandomNoise","CreateVideo","MiniMaxH3ImageToVideo","SaveVideo"}

def http_json(path,method="GET",data=None,timeout=30):
    body=None; headers={}
    if data is not None:
        body=json.dumps(data).encode(); headers["Content-Type"]="application/json"
    req=urllib.request.Request(COMFY_URL.rstrip("/")+path,data=body,headers=headers,method=method)
    with urllib.request.urlopen(req,timeout=timeout) as r:
        return json.loads(r.read().decode())

def h3_length(seconds):
    base=max(5,round(seconds*24)); return base+(5-(base%17))%17

def graph(job,i):
    return {
"119":{"class_type":"VAELoader","inputs":{"vae_name":VIDEO_VAE}},
"120":{"class_type":"VAELoader","inputs":{"vae_name":AUDIO_VAE}},
"121":{"class_type":"VAEDecodeAudio","inputs":{"samples":["125",0],"vae":["120",0]}},
"122":{"class_type":"VAEDecode","inputs":{"samples":["125",0],"vae":["119",0]}},
"123":{"class_type":"KSamplerSelect","inputs":{"sampler_name":"res_multistep"}},
"124":{"class_type":"BasicScheduler","inputs":{"model":["127",0],"scheduler":"simple","steps":STEPS,"denoise":1.0}},
"125":{"class_type":"SamplerCustomAdvanced","inputs":{"noise":["129",0],"guider":["126",0],"sampler":["123",0],"sigmas":["124",0],"latent_image":["131",1]}},
"126":{"class_type":"BasicGuider","inputs":{"model":["127",0],"conditioning":["131",0]}},
"127":{"class_type":"UNETLoader","inputs":{"unet_name":UNET,"weight_dtype":"default"}},
"128":{"class_type":"CLIPLoader","inputs":{"clip_name":CLIP,"type":"minimax","device":"default"}},
"129":{"class_type":"RandomNoise","inputs":{"noise_seed":job["seed"]}},
"130":{"class_type":"CreateVideo","inputs":{"images":["122",0],"audio":["121",0],"fps":24,"bit_depth":8}},
"131":{"class_type":"MiniMaxH3ImageToVideo","inputs":{"clip":["128",0],"vae":["119",0],"prompt":job["prompt"],"width":WIDTH,"height":HEIGHT,"length":h3_length(DURATION_SECONDS)}},
"92":{"class_type":"SaveVideo","inputs":{"video":["130",0],"filename_prefix":f"{OUTPUT_ROOT}/{i:02d}_{job['name']}","format":"auto","codec":"auto"}}}

def gpu():
    out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.total,memory.used,memory.free","--format=csv,noheader,nounits"],text=True).strip().splitlines()[0]
    return tuple(map(lambda x:int(x.strip()),out.split(",")))

def llama():
    try:
        out=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True).strip()
    except subprocess.CalledProcessError:
        return []
    return [x for x in out.splitlines() if "llama" in x.lower()]

def preflight(force=False):
    print("H3 triangle-box braid preflight")
    http_json("/system_stats"); print("[OK] ComfyUI reachable")
    info=http_json("/object_info")
    missing=sorted(REQ-set(info))
    if missing: raise RuntimeError("Missing nodes: "+", ".join(missing))
    if llama() and not force: raise RuntimeError("llama is still using GPU")
    total,used,free=gpu(); print(f"[INFO] VRAM {used}/{total} MiB used; {free} free")
    if free<MIN_FREE_VRAM_MIB and not force: raise RuntimeError(f"Only {free} MiB VRAM free")
    print(f"[INFO] {len(JOBS)} jobs, expected ~60-70 minutes cold")
    print("[PASS]")

def submit(g):
    r=http_json("/prompt","POST",{"prompt":g,"client_id":"h3boxbraid"},60)
    if r.get("node_errors"): raise RuntimeError(json.dumps(r["node_errors"],indent=2))
    return r["prompt_id"]

def wait(pid):
    end=time.time()+JOB_TIMEOUT_SECONDS
    while time.time()<end:
        h=http_json(f"/history/{pid}")
        e=h.get(pid)
        if e:
            s=e.get("status") or {}; st=str(s.get("status_str","")).lower()
            if s.get("completed") or st=="success": return True,e,""
            if st in {"error","failed"}: return False,e,json.dumps(s.get("messages") or [])
        time.sleep(POLL_SECONDS)
    raise TimeoutError(pid)

def log(rec):
    with Path(__file__).with_name("h3boxbraid.log.jsonl").open("a") as f:
        f.write(json.dumps(rec)+"\n")

def run(start=1,force=False):
    preflight(force)
    print("\nStarting triangle-box braid test")
    for i,j in enumerate(JOBS,1):
        if i<start: continue
        st=datetime.now().astimezone()
        print(f"\n[{i}/{len(JOBS)}] {j['name']}")
        pid=submit(graph(j,i)); print("  prompt_id="+pid)
        ok,h,err=wait(pid)
        en=datetime.now().astimezone(); rt=(en-st).total_seconds()
        log({"index":i,"name":j["name"],"stage":j["stage"],"seed":j["seed"],"prompt_id":pid,"started":st.isoformat(),"ended":en.isoformat(),"runtime_seconds":rt,"status":"success" if ok else "failed","outputs":h.get("outputs") if ok else None,"error":err if not ok else None})
        print(f"  {'SUCCESS' if ok else 'FAILED'} in {rt/60:.1f} min")
        if not ok and ("out of memory" in err.lower() or "cuda" in err.lower()): return 2
    print("\nTriangle-box braid test complete.")
    return 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--check",action="store_true"); ap.add_argument("--from",dest="start",type=int,default=1); ap.add_argument("--force",action="store_true")
    a=ap.parse_args()
    try:
        if a.check: preflight(a.force); return 0
        return run(max(1,a.start),a.force)
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
