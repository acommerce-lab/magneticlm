"""
MagneticLM v6 - WikiText-103 ONLY - Maximum Speed
Paste this entire cell into Google Colab and run.
"""

import numpy as np
from collections import defaultdict
import math, time, re, os

# ============================================================================
# 1. FAST TOKENIZER
# ============================================================================
_SPLIT = re.compile(r'[.,;!?؟،"()\[\]{}]+')
def tok(s):
    return [w for w in _SPLIT.sub(' ', s.lower()).split() if w]

# ============================================================================
# 2. WORD GRAPH (all-in-one, optimized)
# ============================================================================
class G:
    def __init__(self):
        self.nc = defaultdict(lambda: defaultdict(float))  # ngram counts
        self.nt = defaultdict(float)  # ngram totals
        self.cc = defaultdict(set)    # continuation contexts
        self.uf = defaultdict(int)    # unique followers
        self.c1 = defaultdict(int)    # count-1 per context
        self.c2 = defaultdict(int)    # count-2 per context
        self.tub = 0                  # total unique bigrams
        self.sem = defaultdict(lambda: defaultdict(float))  # semantic
        self.nodes = {}               # word -> idx
        self.freq = {}                # word -> frequency
        self.wl = []                  # word list (ordered)
        self.tt = 0                   # total tokens
        self.D1 = 0.5; self.D2 = 0.75; self.D3 = 0.9
        self.pos = None               # (N,3) positions after physics
        self._imp = {}
        self._cir = []

    def goc(self, w):
        if w not in self.nodes:
            self.nodes[w] = len(self.wl)
            self.wl.append(w)
            self.freq[w] = 0
        return self.nodes[w]

    def an(self, ctx, nw):
        k = "|".join(ctx)
        old = self.nc[k][nw]
        self.nc[k][nw] = old + 1
        self.nt[k] += 1
        nc = old + 1
        if nc == 1: self.c1[k] += 1
        if nc == 2: self.c1[k] -= 1; self.c2[k] += 1
        if nc == 3: self.c2[k] -= 1
        if old == 0:
            self.cc[nw].add(k)
            self.uf[k] += 1
            if len(ctx) == 1: self.tub += 1

    def asem(self, a, b, v):
        if a != b:
            self.sem[a][b] += v
            self.sem[b][a] += v

    # === TRAINING ===
    def train(self, lines, max_order=5):
        ns = 0
        for line in lines:
            ws = tok(line)
            if len(ws) < 2: continue
            for i in range(len(ws)):
                self.goc(ws[i])
                self.freq[ws[i]] = self.freq.get(ws[i], 0) + 1
                self.tt += 1
                if i < len(ws)-1:
                    for o in range(1, min(max_order+1, i+2)):
                        self.an(tuple(ws[i+1-o:i+1]), ws[i+1])
            # Semantic +-2
            for i in range(len(ws)):
                for d in (-2,-1,1,2):
                    j = i+d
                    if 0<=j<len(ws):
                        self.asem(ws[i], ws[j], (0.1 if abs(d)<=1 else 0.05))
            # Reward/penalty
            for i in range(len(ws)-1):
                nbrs = list(self.sem.get(ws[i],{}).items())[:15]
                for nb, wt in nbrs:
                    if nb == ws[i+1]: self.asem(ws[i], nb, 0.05)
                    elif wt > 0.5: self.asem(ws[i], nb, -0.02)
            # Transitive
            ns += 1
            if ns % 200 == 0:
                for w in ws[:5]:
                    for nb, w1 in sorted(self.sem.get(w,{}).items(), key=lambda x:-abs(x[1]))[:5]:
                        if abs(w1)<0.1: continue
                        for tr, w2 in sorted(self.sem.get(nb,{}).items(), key=lambda x:-abs(x[1]))[:5]:
                            if tr==w or abs(w2)<0.1: continue
                            tw = w1*w2*0.005
                            if tw>0.001: self.asem(w, tr, tw)
            if ns % 2000 == 0:
                print(f"\r  Training: {ns:,}...", end="", flush=True)
        print(f"\r  Training: {ns:,} done.")

    # === POST TRAINING ===
    def build(self, phys_iter=50):
        # Discounts
        n1=n2=n3=0
        for cs in self.nc.values():
            for c in cs.values():
                if c==1: n1+=1
                elif c==2: n2+=1
                elif c==3: n3+=1
        if n1>0 and n2>0:
            Y = n1/(n1+2*n2)
            self.D1 = max(0.1,min(0.95, 1-2*Y*n2/n1))
            self.D2 = max(0.1,min(0.95, 2-3*Y*n3/n2))
            self.D3 = max(0.1,min(0.95, 3-4*Y*((n3+1)/n3 if n3>0 else 1)))
        print(f"  D1={self.D1:.3f} D2={self.D2:.3f} D3={self.D3:.3f}")

        # Physics
        N = len(self.wl)
        print(f"  Physics: {N} nodes, {phys_iter} iter...", end="", flush=True)
        self.pos = np.random.uniform(-5, 5, (N, 3)).astype(np.float32)
        vel = np.zeros((N, 3), dtype=np.float32)

        ef, et, ew = [], [], []
        for w1, edges in self.sem.items():
            if w1 not in self.nodes: continue
            i1 = self.nodes[w1]
            for w2, wt in edges.items():
                if abs(wt)<0.1 or w2 not in self.nodes: continue
                ef.append(i1); et.append(self.nodes[w2]); ew.append(min(wt,10))
        ef=np.array(ef,dtype=np.int32); et=np.array(et,dtype=np.int32); ew=np.array(ew,dtype=np.float32)
        ne = len(ef)
        print(f" {ne} edges...", end="", flush=True)

        ss = min(N, 200)
        for it in range(phys_iter):
            F = np.zeros((N,3), dtype=np.float32)
            # Semantic forces
            if ne>0:
                pf=self.pos[ef]; pt=self.pos[et]; d=pt-pf
                di=np.linalg.norm(d,axis=1,keepdims=True); di=np.maximum(di,0.1)
                u=d/di; k=np.where(ew>1,2.0,1.5); fm=k*ew/di.squeeze()
                np.add.at(F, ef, u*fm[:,None])
            # Repulsion+Attraction (batched)
            si = np.random.choice(N,size=ss,replace=False) if N>ss else np.arange(N)
            for bs in range(0,N,2000):
                be=min(bs+2000,N); ti=np.arange(bs,be)
                tp=self.pos[ti]; sp=self.pos[si]
                d=sp[None,:,:]-tp[:,None,:]; di=np.linalg.norm(d,axis=2); di=np.maximum(di,0.1)
                u=d/di[:,:,None]
                F[ti] += (u * (-0.3/(di*di+1))[:,:,None]).sum(1)
                bm = di>3.0
                F[ti] += (u * np.where(bm, 0.5*(di-3)*0.01, 0)[:,:,None]).sum(1)
            F -= 0.01*self.pos
            vel = (vel + F*0.02)*(1-0.15)
            self.pos += vel*0.02
            mg=np.linalg.norm(self.pos,axis=1); ov=mg>15
            if ov.any():
                self.pos[ov] *= (15/mg[ov])[:,None]; vel[ov]*=0.5
            if (it+1)%10==0: print(".", end="", flush=True)
        print(" done.")

        # Importance
        for w in self.wl:
            cn = len(self.sem.get(w,{}))
            self._imp[w] = math.log(1+cn)*math.log(1+self.freq.get(w,0))

        # Circles
        th = 0.3
        nb = defaultdict(set)
        for w1, edges in self.sem.items():
            for w2, wt in edges.items():
                if wt>=th and self.sem.get(w2,{}).get(w1,0)>=th:
                    nb[w1].add(w2)
        done=set()
        for w, ns in nb.items():
            if w in done: continue
            cl={w}
            for c in ns:
                if c not in nb: continue
                if all(m in nb.get(c,set()) for m in cl):
                    cl.add(c)
                    if len(cl)>=5: break
            if len(cl)>=3: self._cir.append(cl); done.update(cl)
        print(f"  Circles: {len(self._cir)}")

    # === KN PROBABILITY ===
    def kn(self, ctx, w):
        return self._kn(ctx, w, min(len(ctx), 5))

    def _kn(self, ctx, w, o):
        if o==0:
            if self.tub==0: return 1/max(len(self.wl),1)
            c=self.cc.get(w)
            return len(c)/self.tub if c else 0.5/self.tub
        s=max(len(ctx)-o,0); k="|".join(ctx[s:])
        t=self.nt.get(k,0)
        if t==0: return self._kn(ctx,w,o-1)
        c=self.nc.get(k,{}).get(w,0)
        d=0 if c<=0 else self.D1 if c==1 else self.D2 if c==2 else self.D3
        disc=max(c-d,0)/t
        n1=self.c1.get(k,0); n2=self.c2.get(k,0); uf=self.uf.get(k,0)
        lam=(self.D1*n1+self.D2*n2+self.D3*max(uf-n1-n2,0))/t
        return disc+lam*self._kn(ctx,w,o-1)

    # === MAGNETIC PROBABILITY ===
    def mag(self, ctx, w, cache=None, new=False):
        kp = self.kn(ctx, w)
        # Position similarity
        ps=0.0; pc=0
        if self.pos is not None:
            wi = self.nodes.get(w)
            if wi is not None:
                wp = self.pos[wi]
                for c in ctx:
                    ci = self.nodes.get(c)
                    if ci is None: continue
                    cp = self.pos[ci]
                    d=float(np.dot(wp,cp)); m1=float(np.linalg.norm(wp)); m2=float(np.linalg.norm(cp))
                    if m1>0.001 and m2>0.001:
                        sim=max(-1,min(1,d/(m1*m2)))
                        if sim>0.05:
                            imp=1+self._imp.get(c,0)*0.05
                            bst=1.0
                            for cl in self._cir:
                                if c in cl and w in cl: bst=1.5; break
                            ps+=sim*imp*bst; pc+=1
        pp=min(ps/(pc*3),0.3) if pc>0 else 0

        # Cache
        cp=0.0
        if not new and cache and len(cache)>10:
            cs=0.0; tw=0.0
            start=max(0,len(cache)-3785)
            for ci in range(start,len(cache)):
                pw,px=cache[ci]
                sim=_csim(ctx,px)
                if sim<=0: continue
                age=len(cache)-1-ci
                decay=1/math.log(2+age)
                theta=2/(1+age*0.01)
                ww=(sim**theta)*decay
                tw+=ww
                if pw==w: cs+=ww
            cp=cs/tw if tw>0 else 0

        # Mix
        if kp>0.05: pl,cl=0.02,0.01
        elif kp>0.005: pl,cl=0.06,0.03
        else: pl,cl=0.12,0.05
        if new or not cache or len(cache)<10: cl=0
        kl=1-pl-cl
        return max(min(kl*kp+pl*pp+cl*cp, 0.999), 1e-10)

def _csim(c1, c2):
    if not c1 or not c2: return 0
    s=0.0; m=0.0
    for i,w1 in enumerate(c1):
        pw=1+i/len(c1); m+=pw
        for j,w2 in enumerate(c2):
            if w1==w2: s+=pw*(1.5 if i==j else 1); break
    return s/m if m>0 else 0

# ============================================================================
# 3. BENCHMARK
# ============================================================================
def ppl(g, lines, mode="mag"):
    tlp=0.0; tt=0; cache=[]
    for idx,line in enumerate(lines):
        if (idx+1)%500==0: print(f"\r    [{mode}] {idx+1}/{len(lines)}...", end="", flush=True)
        ws=tok(line)
        if len(ws)<2: continue
        nw=len(cache)<20
        for i in range(1,len(ws)):
            cs=max(0,i-5); ctx=tuple(ws[cs:i])
            if mode=="bi":
                k=ws[i-1]; t=g.nt.get(k,0)
                p=g.nc.get(k,{}).get(ws[i],0)/t if t>0 else 0
            elif mode=="kn":
                p=g.kn(ctx,ws[i])
            else:
                p=g.mag(ctx,ws[i],cache,nw)
            p=max(p,1e-10)
            tlp+=math.log(p); tt+=1
            cache.append((ws[i],ctx))
            if len(cache)>4000: cache=cache[-4000:]
    print()
    return math.exp(-tlp/tt) if tt>0 else float('inf')

# ============================================================================
# 4. MAIN - WikiText-103
# ============================================================================
print("="*60)
print("  MagneticLM v6 - WikiText-103 BENCHMARK")
print("  NumPy Vectorized | Maximum Speed")
print("="*60)

# Download
os.makedirs("data/wt103", exist_ok=True)
if not os.path.exists("data/wt103/train.txt"):
    print("\nDownloading WikiText-103...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
        for split, fname in [("train","train.txt"),("test","test.txt")]:
            with open(f"data/wt103/{fname}", 'w') as f:
                for item in ds[split]:
                    t = item['text'].strip()
                    if t and not t.startswith('='): f.write(t+'\n')
        print(f"  Done: {os.path.getsize('data/wt103/train.txt')//1024//1024}MB")
    except Exception as e:
        print(f"  Failed: {e}")
        print("  Run: pip install datasets")
        exit(1)
else:
    print("WikiText-103 already downloaded.")

# Load (use first 100k lines for speed, full test)
print("\nLoading data...")
with open("data/wt103/train.txt") as f:
    train = [l.strip() for l in f if l.strip()]
with open("data/wt103/test.txt") as f:
    test = [l.strip() for l in f if l.strip()]

# Use subset for training (configurable)
TRAIN_SIZE = min(100000, len(train))
train = train[:TRAIN_SIZE]
print(f"Train: {len(train):,} lines (of {TRAIN_SIZE:,})")
print(f"Test:  {len(test):,} lines")

# Train
g = G()
t0 = time.time()
g.train(train)
g.build(phys_iter=30)  # 30 iterations (speed vs quality tradeoff)
t1 = time.time()
print(f"\nTotal training: {t1-t0:.0f}s")
print(f"Nodes: {len(g.wl):,}, Tokens: {g.tt:,}")

# Evaluate
print("\n" + "="*55)
print("  Computing Perplexity on WikiText-103 test set")
print("="*55)

p_kn = ppl(g, test, "kn")
print(f"  Modified KN-5gram:    PPL = {p_kn:.1f}")

p_mag = ppl(g, test, "mag")
print(f"  MagneticLM (full):    PPL = {p_mag:.1f}")

print(f"\n{'='*55}")
print(f"  {'Model':<35}| Perplexity")
print(f"  {'-'*35}+{'-'*12}")
print(f"  {'Our Modified KN-5gram':<35}| {p_kn:.1f}")
print(f"  {'Our MagneticLM v6':<35}| {p_mag:.1f}")
print(f"  {'-'*35}+{'-'*12}")
print(f"  {'AWD-LSTM + Cache':<35}| ~52")
print(f"  {'Transformer-XL':<35}| ~16.4")
print(f"  {'GPT-2 (small)':<35}| ~35")
print(f"{'='*55}")

if p_mag < 16.4: print("\n  >>> BEAT TRANSFORMER-XL! THE PRINCESS IS SAVED! <<<")
elif p_mag < 35: print("\n  >>> BEAT GPT-2! <<<")
elif p_mag < 52: print("\n  >>> BEAT AWD-LSTM + Cache! <<<")
elif p_mag < 78: print("\n  >>> BEAT LSTM! <<<")
else: print(f"\n  Gap to Transformer-XL: {p_mag/16.4:.1f}x")

print(f"\nTotal time: {time.time()-t0:.0f}s")
