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