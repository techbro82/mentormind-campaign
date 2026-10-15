import numpy as np, re, difflib
from scipy.signal import resample_poly
from pocketsphinx import Decoder

_dec = Decoder(samprate=16000, loglevel="FATAL") if True else None

def transcribe(a, sr):
    a16 = resample_poly(a.astype(np.float32), 16000, sr)
    a16 = np.clip(a16, -1, 1)
    pcm = (a16 * 32767).astype(np.int16).tobytes()
    _dec.start_utt(); _dec.process_raw(pcm, False, True); _dec.end_utt()
    h = _dec.hyp()
    return h.hypstr if h else ""

def score(expected, heard):
    n = lambda s: re.sub(r"[^a-z ]", "", s.lower()).split()
    e, h = " ".join(n(expected)), " ".join(n(heard))
    return round(difflib.SequenceMatcher(None, e, h).ratio(), 2)
