#!/usr/bin/env python3
"""Summarize existing STORM phase counters; timings overlap and are not additive."""
import argparse
import json
from pathlib import Path
import re
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('runs', type=Path, help='runs.json written by compare.py')
args = parser.parse_args()
patterns = {
    'setup': r'MC step max-rank time: setup=(\S+) s',
    'generation': r'MC step max-rank time:.*generation=(\S+) s',
    'loop': r'MC step max-rank time:.*loop=(\S+) s',
    'census': r'MC step max-rank time:.*census=(\S+) s',
    'communication': r'MC loop split max-rank: rma=(\S+) s',
    'handle': r'MC loop split max-rank:.*?handle=(\S+) s',
}
by_case = {}
for run in json.loads(args.runs.read_text()):
    if run['phase'] != 'measured':
        continue
    text = Path(run['log']).read_text()
    totals = {key: sum(map(float, re.findall(pattern, text)))
              for key, pattern in patterns.items()}
    by_case.setdefault(run['case'], []).append(totals)
print(json.dumps({case: {key: statistics.median(row[key] for row in rows)
                        for key in patterns}
                  for case, rows in by_case.items()}, indent=2))
