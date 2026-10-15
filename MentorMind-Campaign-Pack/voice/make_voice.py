"""Make one day's narration with Kokoro and save its exact timings.

Usage:  python3 make_voice.py spec.json

spec.json example:
{
  "out": "day10", "voice": "af_nicole", "lang": "en-us", "lead": 0.5, "tail": 0.6,
  "min_speed": 0.88, "max_speed": 1.08,
  "lines": [
    {"text": "First line.", "target": 2.5, "pitch": 1.0, "vol": 1.0, "gap": 0.8},
    {"text": "Second line.", "target": 2.0, "pitch": 0.98, "vol": 0.95, "gap": 0.0}
  ]
}
target = how long the spoken line should last (seconds). gap = silence after it.
Writes /home/claude/voice/<out>_voice_raw.wav (dry), <out>_voice.json (starts and lengths) and <out>_voice.mp3 (preview).
Needs kokoro-v1.0.onnx and voices-v1.0.bin in /home/claude/voice (see setup_restore.sh).
Lessons: do not slow a line below about 0.88 speed (words blur). Use gaps for pacing instead.
The speech recogniser in asr_check.py is biased toward American speech and cannot judge "AI mentors"; trust human ears.
"""
import sys, json, subprocess
import numpy as np, soundfile as sf
sys.path.insert(0, "/home/claude/voice")
from kokoro_onnx import Kokoro
try:
    from asr_check import transcribe, score
except Exception:
    transcribe = score = None

spec = json.load(open(sys.argv[1]))
k = Kokoro("/home/claude/voice/kokoro-v1.0.onnx", "/home/claude/voice/voices-v1.0.bin")
AF = ("highpass=f=70,equalizer=f=170:t=q:w=1:g=3,equalizer=f=3800:t=q:w=1:g=-1.5,"
      "acompressor=threshold=-20dB:ratio=2.5:attack=15:release=180:makeup=2,alimiter=limit=0.95")


def tighten(a, sr, max_pause=0.16):
    env = np.convolve(np.abs(a), np.ones(int(.01 * sr)) / int(.01 * sr), mode="same")
    v = env > 0.025 * env.max()
    idx = np.where(v)[0]
    lo, hi = max(idx[0] - int(sr * .03), 0), idx[-1] + int(sr * .06)
    a, v = a[lo:hi], v[lo:hi]
    out, i, mp = [], 0, int(max_pause * sr)
    while i < len(a):
        j = i
        while j < len(a) and v[j] == v[i]:
            j += 1
        seg = a[i:j]
        if not v[i] and len(seg) > mp:
            seg = seg[:mp] * np.hanning(mp) ** 0.2
        out.append(seg)
        i = j
    return np.concatenate(out)


def pitch(a, sr, ratio):
    if abs(ratio - 1) < 1e-3:
        return a
    sf.write("_p.wav", a.astype(np.float32), sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "_p.wav", "-af", f"asetrate={sr}*{ratio},atempo={1/ratio:.5f},aresample={sr}", "_p2.wav"], check=True)
    b, _ = sf.read("_p2.wav")
    return b.astype(np.float32)


clips, sr = [], 24000
for ln in spec["lines"]:
    a, sr = k.create(ln["text"], voice=spec["voice"], speed=1.0, lang=spec["lang"], trim=False)
    sp = float(np.clip((len(tighten(a, sr)) / sr) / ln["target"], spec.get("min_speed", 0.88), spec.get("max_speed", 1.08)))
    a, sr = k.create(ln["text"], voice=spec["voice"], speed=sp, lang=spec["lang"], trim=False)
    a = pitch(tighten(a, sr), sr, ln.get("pitch", 1.0))
    a = a / np.max(np.abs(a)) * 0.85 * ln.get("vol", 1.0)
    clips.append(a)
    heard = transcribe(a, sr) if transcribe else ""
    print(f"{ln['text'][:50]!r:54} {len(a)/sr:.2f}s speed {sp:.2f} heard {heard[:60]!r}")
d = [len(c) / sr for c in clips]
t0, starts = spec.get("lead", 0.5), []
for i, c in enumerate(clips):
    starts.append(round(t0, 2))
    t0 += d[i] + spec["lines"][i].get("gap", 0.0)
total = round(t0 + spec.get("tail", 0.6), 2)
track = np.zeros(int(total * sr), np.float32)
for c, s in zip(clips, starts):
    track[int(s * sr):int(s * sr) + len(c)] += c
out = spec["out"]
sf.write(f"/home/claude/voice/{out}_voice_raw.wav", track, sr)
json.dump(dict(starts=starts, durs=[round(x, 2) for x in d], total=total, sr=sr, lines=[l["text"] for l in spec["lines"]], voice=spec["voice"]),
          open(f"/home/claude/voice/{out}_voice.json", "w"))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"/home/claude/voice/{out}_voice_raw.wav", "-af", AF, "-codec:a", "libmp3lame", "-qscale:a", "3", f"/home/claude/voice/{out}_voice.mp3"], check=True)
print("starts", starts, "lengths", [round(x, 2) for x in d], "total", total)
