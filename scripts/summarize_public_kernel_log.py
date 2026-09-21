import json,re,sys
for p in sys.argv[1:]:
    try: rows=json.loads(open(p,encoding='utf-8',errors='replace').read())
    except Exception as e: print(p,'ERR',e); continue
    txt=''.join(r['data'] for r in rows if r.get('stream_name')=='stdout')
    sc=[float(m.group(1)) for m in re.finditer(r'\[finished\].*?score=([\d.]+)',txt)]
    ac=[int(m.group(1)) for m in re.finditer(r'\[finished\].*?actions=(\d+)',txt)]
    lv=[int(a) for a,b in re.findall(r'\[finished\].*?level=(\d+)/(\d+)',txt)]
    fl={}
    m=re.search(r"avo_agent=(True|False)",txt); fl['avo']=m.group(1) if m else '-'
    m=re.search(r"animation_awareness=(True|False)",txt); fl['anim']=m.group(1) if m else '-'
    m=re.search(r"concurrency=(\d+)",txt); fl['conc']=m.group(1) if m else '-'
    print(f"{p.split('/')[-1][:45]:47s} n={len(sc):2d} mean={sum(sc)/max(1,len(sc)):6.2f} actions={sum(ac):5d} levels={sum(lv):3d} scoring={sum(1 for s in sc if s>0):2d} {fl}")
