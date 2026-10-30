from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Two frozen v79 queues; complete inclusive timing and eligible events.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import sys, time, json, traceback, hashlib, argparse
from pathlib import Path
from unittest.mock import patch
B = Path(__file__).resolve().parent
R = B.parent
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, psutil
from observed_anchor import ObservedAnchor
import moment_tail
from moment_tail import MomentSearch
from metric_basis import build_proposal
CAP = 10.0
TAU = 2 ** (-16)
METHODS = ('gradient', 'metric_pod', 'metric_greedy', 'full')

def classify(l, u):
    return 'STOP' if u <= TAU else 'VIOLATING' if l > TAU else 'UNRESOLVED'

def endpoint(events, horizon):
    valid = [e for e in events if e['seconds'] <= horizon]
    end = valid[-1] if valid else None
    first = next((e for e in valid if classify(e['lower'], e['upper']) != 'UNRESOLVED'), None)
    stop = next((e for e in valid if e['upper'] <= TAU), None)
    return dict(horizon_seconds=horizon, decision=classify(end['lower'], end['upper']) if end else 'UNRESOLVED', lower_K2=end['lower'] * 128 ** 2 if end else None, upper_K2=end['upper'] * 128 ** 2 if end else None, stage=end['stage'] if end else 'setup_not_ready', first_decision_seconds=first['seconds'] if first else None, first_stop_seconds=stop['seconds'] if stop else None, decision_event_observed=bool(first), stop_event_observed=bool(stop), restricted_decision_seconds=first['seconds'] if first else horizon, restricted_stop_seconds=stop['seconds'] if stop else horizon)

def fingerprint(args):
    h = hashlib.sha256()
    for v in args:
        h.update(np.asarray(v, dtype=np.float64).tobytes())
    return h.hexdigest()
