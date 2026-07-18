"""Full IAM transcription with kraken McCATMuS_nfd_nofix_V1.

Method: pre-segmented Teklia/IAM-line images (128px height); each line is
pasted onto a white canvas with a 24px margin (the legacy polygon extractor
rejects boundaries touching image edges) and recognized against a synthetic
horizontal baseline at 75% of line height with a full-rectangle boundary.
kraken's blla neural segmenter is not usable at this scale on this machine
(minutes per line on these extreme-aspect strips).

Output: Andy-schema JSONL (index, ground_truth, predicted_text, cer, wer,
inference_time, error, run_date) — plugs into evaluation/rescore_iam.py.
"""

import json, time, unicodedata
from pathlib import Path
from PIL import Image
from kraken.lib import models
from kraken import rpred
from kraken.containers import Segmentation, BaselineLine
import rapidfuzz.distance.Levenshtein as Lev

MODEL = "/Users/wouter/Library/Application Support/htrmopo/78d7d26e-50b3-55f8-b839-c7a882ea2053/McCATMuS_nfd_nofix_V1.mlmodel"
SRC = Path("/private/tmp/claude-501/-Users-wouter-gitrepos-not-flawless/3055962f-aa6b-4a0e-acd2-b326608d209b/scratchpad/iam_all_lines")
OUT = Path("/Users/wouter/gitrepos/misnomer/evaluation/kraken-mccatmus_results.jsonl")
M = 24

net = models.load_any(MODEL)
gt_map = json.loads((SRC / "gt.json").read_text())

def recognize(im):
    w, h = im.width, im.height
    canvas = Image.new("L", (w + 2 * M, h + 2 * M), 255)
    canvas.paste(im, (M, M))
    y = M + int(h * 0.75)
    seg = Segmentation(type="baselines", imagename="line", text_direction="horizontal-lr",
                       script_detection=False,
                       lines=[BaselineLine(id="l1", baseline=[[M, y], [M + w, y]],
                                           boundary=[[M, M], [M + w, M], [M + w, M + h], [M, M + h]])])
    return unicodedata.normalize("NFC", " ".join(
        r.prediction.strip() for r in rpred.rpred(net, canvas, seg)).strip())

def wer(pred, gt):
    p, g = pred.split(), gt.split()
    return Lev.distance(p, g) / max(1, len(g)) * 100

start = time.monotonic()
with OUT.open("w", encoding="utf-8") as fh:
    for i in range(10373):
        gt = gt_map[str(i)]
        t0 = time.monotonic()
        err = None
        try:
            pred = recognize(Image.open(SRC / f"{i}.png").convert("L"))
        except Exception as exc:
            pred, err = "", f"{type(exc).__name__}: {exc}"
        dt = time.monotonic() - t0
        fh.write(json.dumps({
            "index": i,
            "ground_truth": gt,
            "predicted_text": pred,
            "cer": Lev.distance(pred, gt) / max(1, len(gt)) * 100,
            "wer": wer(pred, gt),
            "inference_time": dt,
            "error": err,
            "run_date": "2026-07-18",
        }, ensure_ascii=False) + "\n")
        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.monotonic() - start)
            print(f"{i+1}/10373 ({rate:.1f}/s, ~{(10373-i-1)/rate/60:.0f} min left)", flush=True)
print(f"done in {(time.monotonic()-start)/60:.1f} min -> {OUT}", flush=True)
