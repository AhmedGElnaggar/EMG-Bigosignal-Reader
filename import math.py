import math
import random
import threading
import time
from collections import deque

import numpy as np
from flask import Flask, jsonify, render_template
from scipy.signal import butter, sosfilt