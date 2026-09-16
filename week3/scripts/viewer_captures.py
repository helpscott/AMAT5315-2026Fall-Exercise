#!/usr/bin/env python3
import json, pathlib, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]; E=ROOT/'evidence'
frames=[]
with open(ROOT/'spins.jsonl') as f:
 for line in f:
  if line.strip(): frames.append(json.loads(line))
for T in (1.8,2.3,3.0):
 cand=min(frames,key=lambda r:abs(r['T']-T)); L=cand['L']; a=np.array(cand['spins']).reshape(L,L)
 plt.figure(figsize=(5,5));plt.imshow(a,cmap='coolwarm',vmin=-1,vmax=1,interpolation='nearest');plt.title(f'Ising viewer capture, T={cand["T"]:.2f}, sweep={cand["sweep"]}');plt.axis('off');plt.tight_layout();plt.savefig(E/f'viewer-T{T:.1f}.png',dpi=150);plt.close()
