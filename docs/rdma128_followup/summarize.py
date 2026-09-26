import json, re, statistics, sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[]
for label in ['baseline1','candidate1','candidate2','default1','baseline2','baseline3','candidate3']:
    output=root/f'{label}.out'; elapsed=root/f'{label}.elapsed'
    if not output.exists() or not elapsed.exists() or not elapsed.read_text().strip(): continue
    text=output.read_text()
    finish=re.search(r'Finished at t=(\S+) s after (\d+) cycles in (\S+) s',text)
    if not finish: continue
    loops=[float(v) for v in re.findall(r'^Loop time: (\S+) seconds',text,re.M)]
    rows.append(dict(label=label,total_seconds=float(elapsed.read_text()),cycle_seconds=float(finish[3]),
        cycles=int(finish[2]),final_time=float(finish[1]),transport_seconds=sum(loops)))
result={'runs':rows}
for metric in ['total_seconds','cycle_seconds','transport_seconds']:
    b=[r[metric] for r in rows if r['label'].startswith('baseline')]
    c=[r[metric] for r in rows if r['label'].startswith('candidate')]
    if b and c:
        result[metric]={'baseline_median':statistics.median(b),'candidate_median':statistics.median(c),
           'time_reduction_percent':100*(1-statistics.median(c)/statistics.median(b)),
           'baseline_range':[min(b),max(b)],'candidate_range':[min(c),max(c)]}
print(json.dumps(result,indent=2))
