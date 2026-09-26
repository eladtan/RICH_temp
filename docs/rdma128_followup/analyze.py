from pathlib import Path
import collections,struct,subprocess,sys,json
root=Path(sys.argv[1]); byfile=collections.defaultdict(set); ranks={}
for f in sorted(root.glob('rank_*.bin')):
 maps=[]
 for l in f.with_suffix('.maps').read_text().splitlines():
  p=l.split()
  if len(p)<6 or not p[5].startswith('/') or not Path(p[5]).exists():continue
  a,b=(int(x,16) for x in p[0].split('-'));maps.append((a,b,int(p[2],16),p[5]))
 counts=collections.Counter(struct.unpack('<'+'Q'*(f.stat().st_size//8),f.read_bytes())); mapped=collections.Counter()
 for pc,n in counts.items():
  for a,b,off,path in maps:
   if a<=pc<b:
    addr=pc-a+off
    if path.endswith(('baseline','candidate_final')):addr=pc
    mapped[path,addr]+=n;byfile[path].add(addr);break
 ranks[int(f.stem.split('_')[1])]=mapped
symbols={}
for path,addresses in byfile.items():
 addrs=sorted(addresses)
 for start in range(0,len(addrs),2000):
  batch=addrs[start:start+2000]
  out=subprocess.check_output(['addr2line','-f','-C','-e',str(Path('/home/maorm/RICH/build/rdma128_followup/node_libs')/Path(path).name) if path in ('/usr/lib64/libm.so.6','/usr/lib64/libc.so.6') else path,*[hex(a) for a in batch]],text=True).splitlines()
  for i,a in enumerate(batch):symbols[path,a]=(out[2*i],out[2*i+1],path)
agg=collections.Counter(); results={}
for rank,counts in ranks.items():
 funcs=collections.Counter()
 for key,n in counts.items():funcs[symbols[key]]+=n
 agg.update(funcs)
 total=sum(funcs.values())
 app=sum(n for (fn,loc,path),n in funcs.items() if path.endswith(('baseline','candidate_final')))
 physics=sum(n for (fn,loc,path),n in funcs.items() if '/radiation/' in loc or 'FindIntersection' in fn or 'AdvanceOne' in fn or 'CounterRNG' in loc)
 results[rank]={'samples':total,'app':app,'physics':physics,'top':[(n,*key) for key,n in funcs.most_common(35)]}
total=sum(agg.values())
print('ALL RANKS',len(ranks),'SAMPLES',total)
for key,n in agg.most_common(45):print(round(100*n/total,2),n,*key[:2])
print('RANKS SORTED BY PHYSICS SAMPLES (10 ms CPU sampling, includes startup)')
for rank,r in sorted(results.items(),key=lambda kv:kv[1]['physics'],reverse=True)[:20]:print(rank,r['samples'],r['app'],r['physics'])
for rank in [0,*sorted(results,key=lambda r:results[r]['physics'],reverse=True)[:3]]:
 print('RANK',rank)
 for n,fn,loc,path in results[rank]['top'][:20]:print(round(100*n/results[rank]['samples'],2),n,fn,loc)
(root/'analysis.json').write_text(json.dumps({'ranks':results,'aggregate':[(n,*key) for key,n in agg.most_common()]},indent=2))
