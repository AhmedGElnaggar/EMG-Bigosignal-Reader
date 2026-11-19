import math
import random
import threading
import time
from collections import deque

import numpy as np
from flask import Flask, jsonify, render_template
from scipy.signal import butter, sosfilt

app = Flask(__name__)

FS = 1000              # sampling rate (Hz)
BUFFER_SECONDS = 4
BUFFER_SIZE = FS * BUFFER_SECONDS

raw_buffer = deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
lock = threading.Lock()

# 20-450 Hz bandpass (500 Hz is Nyquist at FS=1000, so stay under it)
SOS = butter(4, [20, 450], btype="bandpass", fs=FS, output="sos")

def simulated_emg():
    """Fake muscle signal: noise + 60Hz hum, with a flex burst every few seconds."""
    t = 0.0
    next_flex = time.time() + 2
    flex_end = 0.0
    while True:
        now = time.time()
        if now >= next_flex:
            flex_end = now + random.uniform(0.8, 1.6)
            next_flex = flex_end + random.uniform(1.5, 3.5)

        amp = 0.05
        if now < flex_end:
            progress = 1 - (flex_end - now) / 1.2
            amp = 0.05 + 0.9 * math.sin(max(0, min(progress, 1)) * math.pi)

        sample = amp * random.gauss(0, 1) + 0.03 * math.sin(2 * math.pi * 60 * t)
        with lock:
            raw_buffer.append(sample)

        t += 1 / FS
        time.sleep(1 / FS)


def analyze(window):
    x = np.array(window)
    filtered = sosfilt(SOS, x)
    spectrum = np.abs(np.fft.rfft(filtered * np.hanning(len(filtered))))
    freqs = np.fft.rfftfreq(len(filtered), 1 / FS)
    rms = float(np.sqrt(np.mean(filtered[-200:] ** 2)))  # last 200 ms
    return filtered, freqs, spectrum, rms


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/signal")
def signal():
    with lock:
        window = list(raw_buffer)
    filtered, freqs, spectrum, rms = analyze(window)
    step = 5  # downsample for the browser
    return jsonify({
        "raw": window[::step],
        "filtered": filtered[::step].tolist(),
        "freqs": freqs[:500].tolist(),
        "spectrum": spectrum[:500].tolist(),
        "rms": rms,
        "fs": FS // step,
    })


if __name__ == "__main__":
    threading.Thread(target=simulated_emg, daemon=True).start()
    app.run(debug=False, port=5002)