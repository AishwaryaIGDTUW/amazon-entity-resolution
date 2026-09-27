# ============================== CONFIG ==============================
import os

# ---- PATHS (edit these for SageMaker) -----------------------------------------------------------------
# Folder that contains dataset/ and utils/ (the unzipped student_resource folder).
BASE_DIR  = "/home/ubuntu/student_resource"
# BASE_DIR = "/home/ec2-user/SageMaker/student_resource"            # <-- SageMaker notebook instance example
# BASE_DIR = "/home/sagemaker-user/student_resource"                # <-- SageMaker Studio example

TRAIN_DIR = f"{BASE_DIR}/dataset/train/"   # train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
TEST_DIR  = f"{BASE_DIR}/dataset/test/"    # test_source1.tsv,  test_source2.tsv,  test_source3.tsv
OUT_DIR   = "/home/ubuntu/output"
# OUT_DIR = "/home/ec2-user/SageMaker/output"                       # <-- SageMaker example
VALIDATOR = f"{BASE_DIR}/utils/validate_submission.py"
CHECK_IDS = True     # also verify every matched/candidate ID exists in the test files (uses more memory; set False if RAM is tight)

# ---- SWITCHES ---------------------------------------------------------------------------------------------
DEV_MODE = False

if DEV_MODE:
    N_PER         = 700     # >>> DEV ONLY: training S1 entities per (country, state) stratum (700 -> ~43k S1, ~10-20 min)
    DEV_TEST_S1   = 3000    # >>> DEV ONLY: test S1 entities per country actually processed (others get empty rows)
    DEV_TEST_POOL = 80000   # >>> DEV ONLY: test S2/S3 records per country kept as the candidate pool
else:
    N_PER         = 2500    # FULL run: bigger training sample = closer to test-time density (2500 -> ~190k S1). Raise if RAM/time allow.
    DEV_TEST_S1   = None
    DEV_TEST_POOL = None

N_JOBS       = os.cpu_count()
# N_JOBS     = os.cpu_count()          # <-- uncomment on SageMaker (Linux/fork). Keep 1 on Windows.
FUZZ_WORKERS = -1                      # threads used by rapidfuzz pairwise scoring (-1 = all cores)

# ---- MODEL / BLOCKING HYPER-PARAMETERS (validated on the training data) --------------------------------
SEED      = 0
K1, K2    = 10, 5          # FINAL candidates per S1: top-K1 by combined score + top-K2 by address alone
M1, M2    = 40, 20         # RETRIEVAL size per S1 before the full-vocabulary re-rank (bigger = higher recall, slower)
CAP_FRAC  = 0.01           # retrieval ignores tokens present in more than CAP_FRAC of the pool ...
CAP_MIN   = 2500           # ... but never below CAP_MIN records.  Validated (dev sample): 0.01/2500 -> blocking recall 0.988, F0.5 0.991;
#     0.003/800 -> recall 0.983 (F0.5 0.989) and roughly 2x faster: use it if the full run is too slow.
Q_CHUNK   = 100            # S1 rows per blocking task (bounds memory of the sparse product)
PAIR_CHUNK= 400_000        # pairs per feature chunk
N_TREES   = 300
RESUME         = True
ONLY_COUNTRIES = None     # None = all countries; e.g. ["France"] to run only some (other countries then get empty rows - a trial, not a submission)
THRESH    = None           # None = tune on the held-out fold; or force a value e.g. 0.8
# ---- SCORE TUNING NOTES (macro F0.5) -------------------------------------------------------------------
# Code-level speed/memory fixes below do NOT change the score - only these do, roughly in order of effect:
#  1. Blocking recall is the hard ceiling: if the true match never survives into (qi, pj), no downstream
#     model can find it. Raise M1/M2 (retrieval pool) and/or K1/K2 (final candidates kept) and re-check the
#     "blocking recall = X.XXXX" log line per country - that number upper-bounds everything else. Cost: more
#     candidate pairs -> more pairwise() / feature time, roughly linear in K1+K2.
#  2. N_PER (training sample size) and N_TREES: more training rows / trees usually helps until it plateaus;
#     watch the held-out F0.5 printed at "CHOSEN threshold=...", not just the final number.
#  3. THRESH / per-country thresholding: the grid search already picks the single best threshold on the
#     held-out fold; UNSEEN_MIN_THRESH and TIGHTEN_UNSEEN protect precision on countries never seen in
#     training (e.g. France) - loosening them trades precision for recall on those countries specifically.
#  4. New features (FEATS / S2_NAMES) - e.g. geocoding distance, phone/email exact-match, industry-code
#     match - typically move macro F0.5 more than any hyperparameter once blocking recall is already high.
# None of this is a guarantee: a jump from ~0.90 to ~0.98 macro F0.5 depends on how separable your true vs.
# false pairs actually are in the data, not just on tuning - validate every change against the held-out
# fold (evq) before trusting it, and change ONE knob at a time (CKPT_TAG changes automatically when you
# touch N_PER/K1/K2/M1/M2, so old checkpoints won't silently get reused).
UNSEEN_MIN_THRESH = 0.85   # countries not in the training data (France) never use a threshold below this: the dense-city stress test showed the curve is
                           # flat for ordinary data but the best threshold rises to ~0.9 when many entities share a city, as in France
TIGHTEN_UNSEEN = True      # for a country NOT in the training data (France): if the model predicts clearly more matches per S1 than
TIGHTEN_RATIO  = 1.4       # the training data ever shows (x TIGHTEN_RATIO), raise that country's threshold until it does not (protects precision)
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(f"{OUT_DIR}/models", exist_ok=True)
print("DEV_MODE =",DEV_MODE,"| N_PER =",N_PER,"| N_JOBS =",N_JOBS)


import sys, re, gc, time, zlib, unicodedata, subprocess, warnings, json, multiprocessing as mp
from functools import lru_cache
warnings.filterwarnings("ignore")
import numpy as np, polars as pl, scipy.sparse as sp
from rapidfuzz import fuzz, process
import lightgbm as lgb
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

T0=time.time()
def log(*a):
    try:
        import resource
        mem = f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB peak"
    except Exception:
        mem = "-"
    print(f"[{(time.time()-T0)/60:6.1f} min | {mem}]", *a, flush=True)
rng=np.random.default_rng(SEED)

def read_tsv(path,cols=None):
    """tab-separated, no quote handling (addresses/ID lists contain commas and quotes); nulls -> ''"""
    return pl.read_csv(path,separator="\t",quote_char=None,infer_schema=False,columns=cols).with_columns(pl.all().fill_null(""))

def pmap(fn,items,n_jobs=None):
    """fn over items, in order. Serial when n_jobs<=1 or fork is unavailable (Windows).
    Robust: if a worker dies (e.g. out of memory) only the unfinished items are re-run on a smaller pool; progress is logged."""
    n_jobs=N_JOBS if n_jobs is None else n_jobs
    items=list(items)
    if n_jobs<=1 or os.name!="posix": return [fn(x) for x in items]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    ctx=mp.get_context("fork"); n=len(items)
    res=[None]*n; done=[False]*n; nj=n_jobs; nxt=0.1; big=n>=200
    while not all(done):
        todo=[i for i in range(n) if not done[i]]
        try:
            with ProcessPoolExecutor(max_workers=nj,mp_context=ctx) as ex:
                futs={ex.submit(fn,items[i]):i for i in todo}
                for f in as_completed(futs):
                    i=futs[f]; res[i]=f.result(); done[i]=True; futs[f]=None
                    if big and sum(done)/n>=nxt:
                        log(f"      pmap {sum(done)}/{n} tasks done"); nxt+=0.1
        except BrokenProcessPool:
            nj=max(1,nj//2)
            log(f"      a worker died (likely out of memory) - re-running {n-sum(done)} unfinished tasks with {nj} workers")
    return res

def split_chunks(seq,n_jobs=None):
    n_jobs=N_JOBS if n_jobs is None else n_jobs
    k=max(1,n_jobs*4) if n_jobs>1 else 1
    step=-(-len(seq)//k); return [seq[i:i+step] for i in range(0,len(seq),step)] or [[]]


# ==================== TEXT NORMALISATION (all scripts, US / India / France) ====================
ABBR={"st":"street","rd":"road","ave":"avenue","dr":"drive","ln":"lane","blvd":"boulevard","ct":"court","pvt":"private","ltd":"limited","corp":"corporation","inc":"incorporated","co":"company","r":"rue","ft":"fort","mt":"mount","hwy":"highway","apt":"apartment","flr":"floor","fl":"floor"}
US={"al":"alabama","ak":"alaska","az":"arizona","ar":"arkansas","ca":"california","co":"colorado","ct":"connecticut","de":"delaware","fl":"florida","ga":"georgia","hi":"hawaii","id":"idaho","il":"illinois","in":"indiana","ia":"iowa","ks":"kansas","ky":"kentucky","la":"louisiana","me":"maine","md":"maryland","ma":"massachusetts","mi":"michigan","mn":"minnesota","ms":"mississippi","mo":"missouri","mt":"montana","ne":"nebraska","nv":"nevada","nh":"new hampshire","nj":"new jersey","nm":"new mexico","ny":"new york","nc":"north carolina","nd":"north dakota","oh":"ohio","ok":"oklahoma","or":"oregon","pa":"pennsylvania","ri":"rhode island","sc":"south carolina","sd":"south dakota","tn":"tennessee","tx":"texas","ut":"utah","vt":"vermont","va":"virginia","wa":"washington","wv":"west virginia","wi":"wisconsin","wy":"wyoming","dc":"district of columbia"}
IN={"ap":"andhra pradesh","ar":"arunachal pradesh","as":"assam","br":"bihar","ct":"chhattisgarh","cg":"chhattisgarh","ga":"goa","gj":"gujarat","hr":"haryana","hp":"himachal pradesh","jh":"jharkhand","ka":"karnataka","kl":"kerala","mp":"madhya pradesh","mh":"maharashtra","mn":"manipur","ml":"meghalaya","mz":"mizoram","nl":"nagaland","od":"odisha","or":"odisha","pb":"punjab","rj":"rajasthan","sk":"sikkim","tn":"tamil nadu","ts":"telangana","tg":"telangana","tr":"tripura","up":"uttar pradesh","uk":"uttarakhand","ut":"uttarakhand","wb":"west bengal","dl":"delhi","jk":"jammu and kashmir"}
STATE={**IN,**US}  # last-token 2-letter codes only; ambiguous codes (ct, ar, tn...) resolved by per-country map below
STATE_BY_C={"US":US,"India":IN}
# France (seen in test inputs only, no labels): street-type abbreviations used by S2/S3, and department -> region (S1 uses regions, S2/S3 often departments)
FR_ABBR={"r":"rue","av":"avenue","ave":"avenue","bd":"boulevard","bvd":"boulevard","all":"allee","imp":"impasse","pl":"place","rte":"route","chem":"chemin","qu":"quai","fg":"faubourg","sq":"square","res":"residence","st":"saint","ste":"sainte"}
FR_DEPT={"gironde":"nouvelle aquitaine","nord":"hauts de france","pas de calais":"hauts de france","loire atlantique":"pays de la loire"}
# Indic script -> Latin: all Indic blocks share Devanagari's layout, so shift into 0x0900 and transliterate once.
CONS=dict(zip("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ",["k","kh","g","gh","n","ch","chh","j","jh","n","t","th","d","dh","n","t","th","d","dh","n","p","f","b","bh","m","y","r","l","v","sh","sh","s","h","l"]))
VOW=dict(zip("अआइईउऊऋएऐओऔ",["a","a","i","i","u","u","ri","e","ai","o","au"]))
MAT=dict(zip("ािीुूृेैोौ",["a","i","i","u","u","ri","e","ai","o","au"]))
CONS.update({"ऴ":"l","ऱ":"r","ऩ":"n","ड़":"r","ढ़":"rh","ज़":"j","फ़":"f","य़":"y","क़":"k","ख़":"kh","ग़":"g","ऺ":""})
VOW.update({"ऎ":"e","ऒ":"o","ॠ":"ri","ऌ":"li","ऍ":"e","ऑ":"o"})
MAT.update({"ॆ":"e","ॊ":"o","ॅ":"e","ॉ":"o","ॄ":"ri"})
CHILLU={"ൺ":"n","ൻ":"n","ർ":"r","ൽ":"l","ൾ":"l","ൿ":"k"}   # Malayalam chillu letters (shifting would land outside the Devanagari tables)
SPECIAL=str.maketrans({"œ":"oe","æ":"ae","ß":"ss","ø":"o","ł":"l","đ":"d","ð":"d","þ":"th","ı":"i"})
def indic(s):
    if not re.search(r"[\u0900-\u0d7f]",s): return s
    s="".join(CHILLU.get(c) or (chr(ord(c)-((ord(c)-0x900)//0x80)*0x80) if 0x980<=ord(c)<0xd80 else c) for c in s)
    out=[];i=0
    while i<len(s):
        c=s[i]
        if c in CONS:
            out.append(CONS[c]); n=s[i+1] if i+1<len(s) else ""
            if n=="\u093c": i+=1; n=s[i+1] if i+1<len(s) else ""
            if n=="\u094d": i+=1
            elif n in MAT: out.append(MAT[n]); i+=1
            elif n in CONS or n in ("ं","ँ","ः"): out.append("a")  # inherent vowel mid-word
        elif c in VOW: out.append(VOW[c])
        elif c in "ंँ":
            n=s[i+1] if i+1<len(s) else ""
            out.append("m" if n in "पफबभम" and n else "n")   # anusvara assimilates before labials
        elif c=="ः": out.append("h")
        elif c in "\u093c\u094d": pass
        else: out.append(c)
        i+=1
    return "".join(out)
def norm(s,abbr=None):
    s=indic(s or "")
    s="".join(str(unicodedata.digit(c,c)) if (not c.isascii() and c.isdigit()) else c for c in s)
    s=s.replace("\u00ba","").replace("\u00b0","")                      # no. / n° markers
    s=unicodedata.normalize("NFKD",s.lower().translate(SPECIAL))
    s=re.sub(r"[\u0300-\u036f]","",s).replace("&"," and ")
    s=re.sub(r"(?<!\w)((?:\w\.){2,}\w?)(?!\w)",lambda m:m.group(1).replace(".",""),s)   # S.A.R.L -> sarl, L.L.C. -> llc
    s=re.sub(r"[^\w\s]"," ",s)
    s=re.sub(r"\b\d{7,}\b"," ",s)                                        # appended phone / id numbers
    s=re.sub(r"\b(?:no|n|numero)\b\s*(?=\d)"," ",s)                      # 'No 5' / 'N 5' -> '5'
    s=re.sub(r"\b0+(\d)",r"\1",s)
    ab=ABBR if abbr is None else abbr
    return " ".join(ab.get(w,w) for w in s.split())
_DG=[("chh","c"),("ch","c"),("sh","s"),("kh","k"),("gh","k"),("th","t"),("dh","t"),("bh","p"),("ph","p"),("jh","j"),("ck","k"),("ee","i"),("oo","u")]
_SK=str.maketrans({"g":"k","q":"k","c":"k","d":"t","b":"p","f":"p","v":"p","w":"p","z":"j","y":"i","x":"k","h":""})
def skel(s):
    """phonetic skeleton: collapses voicing/aspiration, keeps first vowel of each word, drops others"""
    out=[]
    for w in s.split():
        for a,b in _DG: w=w.replace(a,b)
        w=w.translate(_SK)
        w=w[:1]+re.sub(r"[aeiou]","",w[1:])
        w=re.sub(r"(.)\1+",r"\1",w)
        out.append(w)
    return " ".join(out)
FR_AB={**ABBR,**FR_ABBR}
def norm_addr(s,c):
    if c=="France":
        parts=[norm(x,FR_AB) for x in (s or "").split(",")]
        parts=[FR_DEPT.get(p,p) for p in parts if p!="france"]     # drop a bare ', France' component, then department -> region
        return " ".join(" ".join(parts).split())
    w=norm(s).split(); m=STATE_BY_C.get(c,{})
    if w and w[-1] in m: w[-1]=m[w[-1]]      # trailing state code -> full name (country-specific table; unknown countries untouched)
    return " ".join(w)


def region(addr,c):
    """state/region of an address: first comma-part (scanning from the end) that equals a known state name or 2-letter code"""
    m=STATE_BY_C.get(c,{}); names=set(m.values())
    for part in reversed([x.strip() for x in re.sub(r"[^a-z ,]"," ",(addr or "").lower()).split(",")]):
        if part in names: return part
        if part in m: return m[part]
    return "other"


# ======================= SCALABLE BLOCKING (per country) =======================
B = 1 << 20   # hashed feature space


def _hash_rows(rows_of_tokens):
    ind, ptr, dat = [], [0], []
    for toks in rows_of_tokens:
        cnt = {}
        for g in toks:
            h = zlib.crc32(g.encode()) % B
            cnt[h] = cnt.get(h, 0) + 1
        ind += cnt.keys()
        dat += cnt.values()
        ptr.append(len(ind))
    return sp.csr_matrix((np.array(dat, np.float32), np.array(ind, np.int64), np.array(ptr, np.int64)),
                         shape=(len(rows_of_tokens), B))


def _char_grams(s):
    out = []
    for w in s.split():
        w = f" {w} "
        for n in (3, 4):
            out += [w[i:i + n] for i in range(max(1, len(w) - n + 1))]
    return out


def hash_char(texts):
    return _hash_rows([_char_grams(s) for s in texts])


def hash_word(texts):
    return _hash_rows([s.split() for s in texts])


def hashmat(texts, fn):
    return sp.vstack(pmap(fn, split_chunks(texts)), format="csr")


def tfidf_prune(Xq, Xp, cap):
    """1+log tf x idf. Features present in more than `cap` records are dropped: they are useless for blocking and
    make the sparse product explode. Rows are L2-normalised so a dot product is a cosine."""
    df = np.bincount(np.concatenate([Xq.indices, Xp.indices]), minlength=B)
    n = Xq.shape[0] + Xp.shape[0]
    w = np.where(df <= cap, np.log((1 + n) / (1 + df)) + 1, 0).astype(np.float32)
    out = []
    for X in (Xq, Xp):
        X.data = ((1 + np.log(X.data)) * w[X.indices]).astype(np.float32)
        X.eliminate_zeros()
        rows = np.repeat(np.arange(X.shape[0], dtype=np.int32), np.diff(X.indptr))
        rn = (np.sqrt(np.bincount(rows, weights=(X.data * X.data), minlength=X.shape[0])) + 1e-9).astype(np.float32)
        X.data = (X.data / rn[rows]).astype(np.float32)
        del rows
        out.append(X.tocsr())
    return out


BQ = {}
BPT = {}      # globals shared with forked workers


def _topk(idx, val, k):
    return idx if len(idx) <= k else idx[np.argpartition(-val, k - 1)[:k]]


def block_task(rg):
    lo, hi = rg
    Sn = BQ["n"][lo:hi] @ BPT["n"]
    Sk = BQ["k"][lo:hi] @ BPT["k"]
    Sa = BQ["a"][lo:hi] @ BPT["a"]
    Sc = (0.6 * Sn.maximum(0.9 * Sk) + 0.4 * Sa).tocsr()      # name score = max(char-ngram cos, 0.9 * skeleton cos)
    Sa = Sa.tocsr()
    out = []
    for r in range(hi - lo):
        a, b = Sc.indptr[r], Sc.indptr[r + 1]
        c, d = Sa.indptr[r], Sa.indptr[r + 1]
        t1 = _topk(Sc.indices[a:b], Sc.data[a:b], M1)
        t2 = _topk(Sa.indices[c:d], Sa.data[c:d], M2)          # address-only pass: catches garbled names
        cand = np.union1d(t1, t2)
        if len(cand):
            out.append(np.column_stack([np.full(len(cand), lo + r, np.int32), cand.astype(np.int32)]))
    return np.vstack(out) if out else np.zeros((0, 2), np.int32)


def topk_mask(qi, score, k):
    """True for the k best-scoring rows of every group qi"""
    order = np.lexsort((-score, qi))
    qs = qi[order]
    st = np.r_[0, np.flatnonzero(np.diff(qs)) + 1]
    ln = np.diff(np.r_[st, len(qs)])
    pos = np.arange(len(qs)) - np.repeat(st, ln)
    m = np.zeros(len(qi), bool)
    m[order] = pos < k
    return m


def block_country(qn, qa, qk, pn, pa, pk):
    """Two steps.  (1) RETRIEVE: sparse TF-IDF product on token-pruned matrices (tokens present in > cap records are ignored,
    which keeps the product cheap) -> M1 best by combined score + M2 best by address alone per S1 entity.
    (2) RE-RANK the retrieved pairs with the FULL (unpruned) cosines and keep top-K1 by combined score + top-K2 by address.
    Returns (qi, pj) int32 pairs and the full-vocabulary matrices (reused for the pair features)."""
    global BQ, BPT
    cap = max(CAP_MIN, int(CAP_FRAC * len(pn)))
    raw = {}
    for key, (qt, pt, fn) in {"n": (qn, pn, hash_char), "k": (qk, pk, hash_char), "a": (qa, pa, hash_word)}.items():
        raw[key] = (hashmat(qt, fn), hashmat(pt, fn))
    full = {}
    for key, (Xq, Xp) in raw.items():                       # full-vocabulary weights (cap = infinity) on copies
        full[key] = tfidf_prune(Xq.copy(), Xp.copy(), 10 ** 12)
    pruned = {key: tfidf_prune(Xq, Xp, cap) for key, (Xq, Xp) in raw.items()}
    del raw
    BQ = {k_: v[0] for k_, v in pruned.items()}
    BPT = {k_: v[1].T.tocsr() for k_, v in pruned.items()}
    del pruned
    log(f"   matrices built (retrieval cap={cap}); retrieving {len(qn)} x {len(pn)} ...")
    tasks = [(lo, min(lo + Q_CHUNK, len(qn))) for lo in range(0, len(qn), Q_CHUNK)]
    pairs = np.vstack(pmap(block_task, tasks))
    BQ = {}
    BPT = {}
    gc.collect()
    qi, pj = pairs[:, 0], pairs[:, 1]
    del pairs
    cn = pair_cos(full["n"][0], full["n"][1], qi, pj)
    ck = pair_cos(full["k"][0], full["k"][1], qi, pj)
    ca = pair_cos(full["a"][0], full["a"][1], qi, pj)
    sc = 0.6 * np.maximum(cn, 0.9 * ck) + 0.4 * ca
    keep = topk_mask(qi, sc, K1) | topk_mask(qi, ca, K2)
    log(f"   re-ranked {len(qi)} retrieved pairs -> kept {int(keep.sum())}")
    qi, pj = qi[keep], pj[keep]
    o = np.lexsort((pj, qi))
    M = dict(Qn=full["n"][0], Pn=full["n"][1], Qk=full["k"][0], Pk=full["k"][1], Qa=full["a"][0], Pa=full["a"][1])
    return qi[o], pj[o], M


# ============================ PAIR FEATURES ============================
FEATS = ["ncos", "acos", "skcos", "sk_ratio", "sk_tset", "n_ratio", "n_tsort", "n_tset", "n_pr", "a_ratio", "a_tset",
         "a_empty", "len_n", "src3", "nonlatin", "num_jac", "num_first", "a_cratio", "nb_in_a", "na_in_b", "rk",
         "comp_uq", "comp_up", "comp_swap"]


def pair_cos(Q, P, qi, pj, bs=200_000):
    out = np.empty(len(qi), np.float32)
    for a in range(0, len(qi), bs):
        out[a:a + bs] = np.asarray(Q[qi[a:a + bs]].multiply(P[pj[a:a + bs]]).sum(1)).ravel()
    return out


def pairwise(scorer, sa, ia, sb, ib):
    """scorer(sa[ia[k]], sb[ib[k]]) for every k, multithreaded in C++ (rapidfuzz), chunked to bound memory"""
    out = np.empty(len(ia), np.float32)
    for a in range(0, len(ia), PAIR_CHUNK):
        A = [sa[i] for i in ia[a:a + PAIR_CHUNK]]
        Bm = [sb[i] for i in ib[a:a + PAIR_CHUNK]]
        out[a:a + PAIR_CHUNK] = process.cpdist(A, Bm, scorer=scorer, workers=FUZZ_WORKERS, dtype=np.float32)
    return out


_num = re.compile(r"\d+")


def _num_info(strings):
    """Parse the digit-runs of every row ONCE. qa/pa are per-ROW string lists (one entry per S1 / pool
    record), but the old code re-ran _num.findall inside the pair loop -- since every row appears in
    K1+K2 (~15) pairs on the query side, and in many pairs on the pool side (shared across queries),
    that repeated the same regex + set() work 10-20x. Precomputing per row removes that redundancy."""
    lists = [_num.findall(s) for s in strings]
    sets = [set(x) for x in lists]
    firsts = [x[0] if x else None for x in lists]
    lens = np.fromiter((len(x) for x in lists), np.int32, count=len(lists))
    return lists, sets, firsts, lens


def num_feats(qa, pa, qi, pj):
    n = len(qi)
    jac = np.full(n, -1, np.float32)
    first = np.zeros(n, np.float32)
    bina = np.full(n, -1, np.float32)
    ainb = np.full(n, -1, np.float32)
    qlists, qsets, qfirsts, qlens = _num_info(qa)
    plists, psets, pfirsts, plens = _num_info(pa)
    for k in range(n):
        i, j = qi[k], pj[k]
        la, lb = qlens[i], plens[j]
        if la == 0 and lb == 0:
            continue
        sa_, sb_ = qsets[i], psets[j]
        if la or lb:
            jac[k] = len(sa_ & sb_) / len(sa_ | sb_)
        if la and lb:
            first[k] = float(qfirsts[i] == pfirsts[j])
        if lb:
            bina[k] = (sum(x in sa_ for x in plists[j]) / lb) if la else 0.0
        if la:
            ainb[k] = (sum(x in sb_ for x in qlists[i]) / la) if lb else 0.0
    return jac, first, bina, ainb


def addr_comps(raw, c):
    """comma-separated address components WITHOUT digits (city / region / landmark), normalised, state codes and French
    departments expanded. A city mismatch leaves unmatched components on BOTH sides; a merely truncated address on one side only."""
    m = STATE_BY_C.get(c, {})
    ab = FR_AB if c == "France" else None
    out = []
    for part in (raw or "").split(","):
        t = norm(part, ab)
        if not t or re.search(r"\d", t) or t == "france":
            continue
        out.append(FR_DEPT.get(t, t) if c == "France" else m.get(t, t))
    return out


@lru_cache(maxsize=500_000)   # city/state/region tokens repeat a lot within a country; cache bounds the double loop below
def _same_comp(x, y):
    return x == y or fuzz.ratio(x, y) >= 80 or fuzz.token_set_ratio(x, y) >= 85


def comp_feats(qc, pc, qi, pj):
    n = len(qi)
    uq = np.zeros(n, np.float32)
    up = np.zeros(n, np.float32)
    for k in range(n):
        a = qc[qi[k]]
        b = pc[pj[k]]
        if not a and not b:
            continue
        uq[k] = sum(1 for x in a if not any(_same_comp(x, y) for y in b))
        up[k] = sum(1 for y in b if not any(_same_comp(x, y) for x in a))
    return uq, up, np.minimum(uq, up)


def make_features(qi, pj, S, M):
    """S = string arrays for the country; M = pruned TF-IDF matrices from blocking; returns float32 matrix, columns = FEATS"""
    cn = pair_cos(M["Qn"], M["Pn"], qi, pj)
    ck = pair_cos(M["Qk"], M["Pk"], qi, pj)
    ac = pair_cos(M["Qa"], M["Pa"], qi, pj)
    ncos = np.maximum(cn, 0.9 * ck)
    F = {"ncos": ncos, "acos": ac, "skcos": ck}
    F["sk_ratio"] = pairwise(fuzz.ratio, S["qk"], qi, S["pk"], pj)
    F["sk_tset"] = pairwise(fuzz.token_set_ratio, S["qk"], qi, S["pk"], pj)
    F["n_ratio"] = pairwise(fuzz.ratio, S["qn"], qi, S["pn"], pj)
    F["n_tsort"] = pairwise(fuzz.token_sort_ratio, S["qn"], qi, S["pn"], pj)
    F["n_tset"] = pairwise(fuzz.token_set_ratio, S["qn"], qi, S["pn"], pj)
    F["n_pr"] = pairwise(fuzz.partial_ratio, S["qn"], qi, S["pn"], pj)
    F["a_ratio"] = pairwise(fuzz.ratio, S["qa"], qi, S["pa"], pj)
    F["a_tset"] = pairwise(fuzz.token_set_ratio, S["qa"], qi, S["pa"], pj)
    F["a_cratio"] = pairwise(fuzz.ratio, S["qc"], qi, S["pc"], pj)      # punctuation/space-free address (G-006 vs G-0-06)
    F["a_empty"] = S["p_aempty"][pj].astype(np.float32)
    F["len_n"] = S["p_lenn"][pj].astype(np.float32)
    F["src3"] = S["p_src3"][pj].astype(np.float32)
    F["nonlatin"] = S["p_nonlatin"][pj].astype(np.float32)
    F["num_jac"], F["num_first"], F["nb_in_a"], F["na_in_b"] = num_feats(S["qa"], S["pa"], qi, pj)
    F["comp_uq"], F["comp_up"], F["comp_swap"] = comp_feats(S["qcomp"], S["pcomp"], qi, pj)
    s = 0.6 * ncos + 0.4 * ac                                            # rank of this candidate among the entity's candidates
    order = np.lexsort((-s, qi))
    qs = qi[order]
    st = np.r_[0, np.flatnonzero(np.diff(qs)) + 1]
    ln = np.diff(np.r_[st, len(qs)])
    rk = np.empty(len(qi), np.float32)
    rk[order] = np.arange(len(qs)) - np.repeat(st, ln)
    F["rk"] = rk
    return np.column_stack([F[f] for f in FEATS]).astype(np.float32)


def norm_names(chunk):
    return [norm(x) for x in chunk]


CUR_C = ""


def norm_addrs(chunk):
    return [norm_addr(x, CUR_C) for x in chunk]


def build_country(s1df, pooldf, country):
    """normalise -> block -> features for one country.
    s1df / pooldf: polars frames with entity_id, business_name, business_address (pool ids carry the S2-/S3- prefix)"""
    global CUR_C
    CUR_C = country
    t = time.time()
    qraw = s1df["business_name"].to_list()
    praw = pooldf["business_name"].to_list()
    S = {}
    S["qn"] = sum(pmap(norm_names, split_chunks(qraw)), [])
    S["pn"] = sum(pmap(norm_names, split_chunks(praw)), [])
    S["qa"] = sum(pmap(norm_addrs, split_chunks(s1df["business_address"].to_list())), [])
    S["pa"] = sum(pmap(norm_addrs, split_chunks(pooldf["business_address"].to_list())), [])
    S["qcomp"] = [addr_comps(x, country) for x in s1df["business_address"].to_list()]
    S["pcomp"] = [addr_comps(x, country) for x in pooldf["business_address"].to_list()]
    S["qk"] = [skel(x) for x in S["qn"]]
    S["pk"] = [skel(x) for x in S["pn"]]
    cm = lambda t_: re.sub(r"[^a-z0-9]", "", t_)
    S["qc"] = [cm(x) for x in S["qa"]]
    S["pc"] = [cm(x) for x in S["pa"]]
    S["p_aempty"] = np.array([x == "" for x in S["pa"]])
    S["p_lenn"] = np.array([len(x) for x in S["pn"]], np.int32)
    S["p_src3"] = (pooldf["entity_id"].str.slice(0, 3) == "S3-").to_numpy()
    S["p_nonlatin"] = pooldf["business_name"].str.contains(r"[^\x00-\x7f]").to_numpy()
    log(f"   {country}: normalised {len(qraw)} S1 + {len(praw)} pool records in {time.time() - t:.0f}s")
    qi, pj, M = block_country(S["qn"], S["qa"], S["qk"], S["pn"], S["pa"], S["pk"])
    log(f"   {country}: {len(qi)} candidate pairs ({len(qi) / max(1, len(qraw)):.1f} per S1)")
    X = make_features(qi, pj, S, M)
    del M
    gc.collect()
    log(f"   {country}: features done {X.shape}")
    return dict(qi=qi, pj=pj, X=X, S=S)


# ==================== STAGE-2 (COMPETITION) FEATURES + METRIC ====================
S2_NAMES = ["p1", "rk_p_s1", "gap_s1", "n_s1", "rk_p_cand", "gap_cand", "other_cand", "sib_a", "sib_n", "sib_p"]
ALL_NAMES = FEATS + S2_NAMES


def grp_stats(g, v):
    """per row: rank of v inside its group g (0 = best), best v among the OTHER rows of the group, group size,
    and the row index of that best other row (-1 if the row is alone)"""
    o = np.lexsort((-v, g))
    gs = g[o]
    vs = v[o]
    st = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
    ln = np.diff(np.r_[st, len(gs)])
    pos = np.arange(len(gs)) - np.repeat(st, ln)
    lnr = np.repeat(ln, ln)
    str_ = np.repeat(st, ln)
    top1 = vs[str_]
    second_idx = np.minimum(str_ + 1, len(gs) - 1)
    top2 = np.where(lnr > 1, vs[second_idx], 0.0)
    other = np.where(pos == 0, top2, top1)
    oidx = np.where(pos == 0, np.where(lnr > 1, o[second_idx], -1), o[str_])
    rank = np.empty(len(g))
    oth = np.empty(len(g))
    sz = np.empty(len(g))
    oi = np.empty(len(g), np.int64)
    rank[o] = pos
    oth[o] = other
    sz[o] = lnr
    oi[o] = oidx
    return rank, oth, sz, oi


def stage2_feats(pp, qi, pj, pa, pk, p_aempty):
    """Competition features. A Source-2/3 record has exactly one true owner, so
      * how does this pair compare with the entity's other candidates?           (rk_p_s1, gap_s1)
      * how does it compare with other entities that also list this record?      (rk_p_cand, gap_cand, other_cand)
      * does the candidate agree with the entity's best OTHER candidate (sibling) - a garbled name still shares its address"""
    r1, o1, s1n, oi = grp_stats(qi, pp)
    r2, o2, _, _ = grp_stats(pj, pp)
    ok = oi >= 0
    j = pj[np.maximum(oi, 0)]
    sib_a = pairwise(fuzz.token_set_ratio, pa, pj, pa, j)
    sib_a = np.where(ok & ~p_aempty[pj] & ~p_aempty[j], sib_a, -1)
    sib_n = np.where(ok, pairwise(fuzz.ratio, pk, pj, pk, j), -1)
    sib_p = np.where(ok, pp[np.maximum(oi, 0)], 0)
    return np.column_stack([pp, r1, pp - o1, s1n, r2, pp - o2, o2, sib_a, sib_n, sib_p]).astype(np.float32)


def macro_f05(sel, qg, y, ntrue, nq, evalmask):
    """macro F0.5 over S1 entities (singletons count 1.0 when nothing is predicted). sel = predicted pairs (bool)"""
    npred = np.bincount(qg[sel], minlength=nq).astype(float)
    tp = np.bincount(qg[sel & y], minlength=nq).astype(float)
    P = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
    R = np.where(ntrue > 0, tp / np.maximum(ntrue, 1), 0.0)
    f = np.where(P + R > 0, 1.25 * P * R / (0.25 * P + R + 1e-12), 0.0)
    f = np.where(ntrue == 0, (npred == 0).astype(float), f)
    return float(f[evalmask].mean()), f


def one_owner(sel, pj, p2):
    """optional hard rule: each candidate record can only belong to its highest-scoring entity"""
    r, _, _, _ = grp_stats(pj, p2)
    return sel & (r == 0)


# ---- CHECKPOINT: reuse a model trained earlier with the SAME settings (dev/full, sample size, blocking parameters, feature set)
CKPT_TAG = f"{'dev' if DEV_MODE else 'full'}_N{N_PER}_f{len(ALL_NAMES)}_K{K1}-{K2}_M{M1}-{M2}_cap{CAP_FRAC}"
CKPT_DIR = f"{OUT_DIR}/checkpoints/{CKPT_TAG}"
os.makedirs(CKPT_DIR, exist_ok=True)
CK_TRAIN_OK = RESUME and all(os.path.exists(f"{CKPT_DIR}/{f}") for f in ["stage1.txt", "stage2.txt", "config.json"])
if CK_TRAIN_OK:
    mAB = lgb.Booster(model_file=f"{CKPT_DIR}/stage1.txt")
    mm = lgb.Booster(model_file=f"{CKPT_DIR}/stage2.txt")
    _c = json.load(open(f"{CKPT_DIR}/config.json"))
    best_t, ONE_OWNER = _c["threshold"], _c["one_owner"]
    EXPECTED_PPS, TRAIN_COUNTRIES = _c["expected_pps"], _c["train_countries"]
    if THRESH is not None:
        best_t = THRESH
    log(f"TRAINING SKIPPED - loaded checkpoint {CKPT_TAG}: threshold={best_t} one_owner={ONE_OWNER} (held-out F0.5 was {_c['heldout_f05']:.4f})")
else:
    # ============ TRAINING: closed-world, region-stratified sample of the TRAIN data ============
    # Sample N_PER Source-1 entities from every (country, state) stratum. The pool holds EVERY Source-2/3 record that truly
    # belongs to a sampled entity plus a proportional sample of the records that belong to nobody (so every pool record's
    # true owner is either present in the sample or does not exist - like at test time).
    t0 = time.time()
    gt = read_tsv(TRAIN_DIR + "train_ground_truth.tsv")
    s1_tr = read_tsv(TRAIN_DIR + "train_source1.tsv")
    g = gt.join(s1_tr, left_on="source1_entity_id", right_on="entity_id", how="left")
    keys = np.array([c + "|" + region(a, c) for a, c in zip(g["business_address"].to_list(), g["country"].to_list())])
    uk, inv = np.unique(keys, return_inverse=True)
    idx = np.concatenate([rng.choice(np.where(inv == i)[0], min(N_PER, int((inv == i).sum())), replace=False) for i in range(len(uk))])
    samp = g[idx.tolist()]
    rkeys = keys[idx]
    log(f"train strata={len(uk)}  sampled S1={len(samp)}")

    all_matched = gt.select(pl.col("matched_entity_ids").str.split(",").alias("eid")).explode("eid").filter(pl.col("eid") != "").unique()
    need = set(e for x in samp["matched_entity_ids"].to_list() for e in x.split(",") if e)
    parts = []
    for f in ["train_source2.tsv", "train_source3.tsv"]:
        df = read_tsv(TRAIN_DIR + f)
        unm = df.join(all_matched, left_on="entity_id", right_on="eid", how="anti")
        n_unm = int(len(unm) * len(samp) / len(gt))
        parts += [df.filter(pl.col("entity_id").is_in(list(need))), unm.sample(n_unm, seed=SEED)]
        del df, unm
    pool_tr = pl.concat(parts)
    del parts, g, s1_tr, gt
    gc.collect()
    log(f"train pool={len(pool_tr)} records")

    TR = []          # one entry per country
    off_q = off_p = 0
    for c in sorted(samp["country"].unique().to_list()):
        S1c = samp.filter(pl.col("country") == c)
        Pc = pool_tr.filter(pl.col("country") == c)
        own = {}
        for i, ml in enumerate(S1c["matched_entity_ids"].to_list()):
            for e in ml.split(","):
                if e:
                    own[e] = i
        owner = np.array([own.get(e, -1) for e in Pc["entity_id"].to_list()])
        R = build_country(S1c, Pc, c)
        R["y"] = owner[R["pj"]] == R["qi"]
        R["ntrue"] = np.bincount(owner[owner >= 0], minlength=len(S1c))
        R["nq"] = len(S1c)
        R["off_q"] = off_q
        R["off_p"] = off_p
        off_q += len(S1c)
        off_p += len(Pc)
        TR.append(R)
        log(f"{c}: blocking recall = {R['y'].sum() / max(1, R['ntrue'].sum()):.4f}  ({len(R['qi']) / len(S1c):.1f} cands / S1)")

    qg = np.concatenate([R["qi"] + R["off_q"] for R in TR])         # global S1 index per pair
    pg = np.concatenate([R["pj"] + R["off_p"] for R in TR])         # global pool index per pair
    X1 = np.vstack([R["X"] for R in TR])
    y = np.concatenate([R["y"] for R in TR])
    ntrue = np.concatenate([R["ntrue"] for R in TR])
    NQ = off_q
    EXPECTED_PPS = float(ntrue.mean())                              # true matches per S1 entity in the training data
    TRAIN_COUNTRIES = sorted(samp['country'].unique().to_list())
    log(f'training truth: {EXPECTED_PPS:.2f} matches per S1 entity, singletons {(ntrue == 0).mean():.3f}')
    fold = rng.choice(3, NQ, p=[.35, .35, .30])                     # 0 = A, 1 = B (train), 2 = C (held-out evaluation)
    rowfold = fold[qg]
    log(f"training pairs={len(y)}  positives={int(y.sum())}  overall blocking recall={y.sum() / ntrue.sum():.4f}")

    PRM = {"objective": "binary", "learning_rate": .08, "num_leaves": 63, "verbose": -1, "num_threads": max(1, os.cpu_count() or 1)}
    fit = lambda Xm, ym: lgb.train(PRM, lgb.Dataset(Xm, ym.astype(int)), N_TREES)
    # stage 1, cross-fitted (A->B, B->A) so stage 2 never sees in-sample scores; C is scored by the model fit on A+B
    p1 = np.zeros(len(y))
    mA = fit(X1[rowfold == 0], y[rowfold == 0])
    mB = fit(X1[rowfold == 1], y[rowfold == 1])
    mAB = fit(X1[rowfold < 2], y[rowfold < 2])
    p1[rowfold == 1] = mA.predict(X1[rowfold == 1])
    p1[rowfold == 0] = mB.predict(X1[rowfold == 0])
    p1[rowfold == 2] = mAB.predict(X1[rowfold == 2])
    log("stage 1 fitted")

    # stage 2 features are computed per country (competition only exists inside a country), then stacked
    X2p = []
    pos = 0
    for R in TR:
        n = len(R["qi"])
        X2p.append(stage2_feats(p1[pos:pos + n], R["qi"], R["pj"], R["S"]["pa"], R["S"]["pk"], R["S"]["p_aempty"]))
        pos += n
    X2 = np.hstack([X1, np.vstack(X2p)])
    mm = fit(X2[rowfold < 2], y[rowfold < 2])
    p2 = mm.predict(X2)
    log("stage 2 fitted")

    evq = fold == 2
    grid = [.3, .4, .5, .6, .7, .75, .8, .85, .9, .95]
    res = {}
    for t in grid:
        s = p2 >= t
        res[(t, False)] = macro_f05(s, qg, y, ntrue, NQ, evq)[0]
        res[(t, True)] = macro_f05(one_owner(s, pg, p2), qg, y, ntrue, NQ, evq)[0]
    res1 = {t: macro_f05(p1 >= t, qg, y, ntrue, NQ, evq)[0] for t in grid}
    print("stage-1 only :", {t: round(v, 4) for t, v in res1.items()})
    print("stage-2      :", {t: round(res[(t, False)], 4) for t in grid})
    print("stage-2 + one-owner rule:", {t: round(res[(t, True)], 4) for t in grid})
    (best_t, ONE_OWNER) = max(res, key=res.get)
    if THRESH is not None:
        best_t = THRESH
    log(f"CHOSEN threshold={best_t}  one_owner_rule={ONE_OWNER}  held-out macro F0.5={res[(best_t, ONE_OWNER)]:.4f}  (stage-1 best {max(res1.values()):.4f})")

    sel = p2 >= best_t
    if ONE_OWNER:
        sel = one_owner(sel, pg, p2)
    ev_pairs = evq[qg]
    nt_ev = ntrue[evq].sum()
    print(f"held-out: pair precision {y[sel & ev_pairs].mean():.4f} | pair recall {y[sel & ev_pairs].sum() / nt_ev:.4f} | blocking recall {y[ev_pairs].sum() / nt_ev:.4f}")
    _, f_ent = macro_f05(sel, qg, y, ntrue, NQ, evq)
    for R in TR:
        m = np.zeros(NQ, bool)
        m[R["off_q"]:R["off_q"] + R["nq"]] = True
        m &= evq
        print(f"  country score (held-out)  n={m.sum()}  F0.5={f_ent[m].mean():.4f}")
    imp = sorted(zip(ALL_NAMES, mm.feature_importance("gain")), key=lambda z: -z[1])
    print("top features:", [(n, int(v)) for n, v in imp[:8]])
    mAB.save_model(f"{CKPT_DIR}/stage1.txt")
    mm.save_model(f"{CKPT_DIR}/stage2.txt")
    json.dump({"threshold": float(best_t), "one_owner": bool(ONE_OWNER), "heldout_f05": float(res[(best_t, ONE_OWNER)]),
               "expected_pps": float(EXPECTED_PPS), "train_countries": TRAIN_COUNTRIES,
               "features": ALL_NAMES, "N_PER": N_PER, "K1": K1, "K2": K2, "DEV_MODE": DEV_MODE}, open(f"{CKPT_DIR}/config.json", "w"), indent=1)   # written LAST: marks the checkpoint complete
    log(f"checkpoint saved to {CKPT_DIR}")
    del TR, X1, X2, X2p
    gc.collect()


# ================= TEST SET: read -> block -> features -> predict (one country at a time, checkpointed) =================
t0 = time.time()
s1_te = read_tsv(TEST_DIR + "test_source1.tsv")
s2_te = read_tsv(TEST_DIR + "test_source2.tsv")
s3_te = read_tsv(TEST_DIR + "test_source3.tsv")
pool_te = pl.concat([s2_te, s3_te])
del s2_te, s3_te
ALL_S1_IDS = s1_te["entity_id"].to_list()            # EVERY test S1 id must appear in both output files
log(f"test: S1={len(s1_te)}  pool(S2+S3)={len(pool_te)}  countries={sorted(s1_te['country'].unique().to_list())}")


def _ck(c, kind):
    return f"{CKPT_DIR}/test_{c}_{kind}.parquet"


def _save_atomic(df, path):
    df.write_parquet(path + ".tmp")
    os.replace(path + ".tmp", path)                  # a half-written file can never look like a finished checkpoint


cand_s1, cand_id, match_s1, match_id = [], [], [], []
for c in sorted(s1_te["country"].unique().to_list()):           # country is an open set: France (unseen in training) is handled like any other label
    if ONLY_COUNTRIES and c not in ONLY_COUNTRIES:              # optional: process only some countries (e.g. a France-only trial)
        continue
    if RESUME and os.path.exists(_ck(c, "cand")) and os.path.exists(_ck(c, "match")):
        a = pl.read_parquet(_ck(c, "cand"))
        b = pl.read_parquet(_ck(c, "match"))
        cand_s1.append(a["s1"].to_numpy())
        cand_id.append(a["id"].to_numpy())
        match_s1.append(b["s1"].to_numpy())
        match_id.append(b["id"].to_numpy())
        log(f"--- test country {c}: LOADED FROM CHECKPOINT ({len(a)} candidate pairs, {len(b)} matched pairs) - skipped")
        continue
    S1c = s1_te.filter(pl.col("country") == c)
    Pc = pool_te.filter(pl.col("country") == c)
    if DEV_MODE:                                                # >>> DEV ONLY (remove for the full run): keep the run small
        S1c = S1c.head(DEV_TEST_S1)                             # >>> DEV ONLY
        Pc = Pc.sample(min(DEV_TEST_POOL, len(Pc)), seed=SEED)  # >>> DEV ONLY
    log(f"--- test country {c}: S1={len(S1c)} pool={len(Pc)}")
    R = build_country(S1c, Pc, c)
    qi, pj, S = R["qi"], R["pj"], R["S"]
    p1t = mAB.predict(R["X"])
    X2t = np.hstack([R["X"], stage2_feats(p1t, qi, pj, S["pa"], S["pk"], S["p_aempty"])])
    p2t = mm.predict(X2t)
    unseen = c not in TRAIN_COUNTRIES
    t_use = max(best_t, UNSEEN_MIN_THRESH) if unseen else best_t
    sel = p2t >= t_use
    if ONE_OWNER:
        sel = one_owner(sel, pj, p2t)
    pps = sel.sum() / len(S1c)
    log(f"   {c}: threshold={t_use}, predicted matches per S1 = {pps:.2f}  (training truth: {EXPECTED_PPS:.2f}){'  <-- country unseen in training' if unseen else ''}")
    if unseen and TIGHTEN_UNSEEN and not DEV_MODE:                  # DEV subsamples the pool, so these rates are only meaningful in a full run
        while pps > TIGHTEN_RATIO * EXPECTED_PPS and t_use < 0.99:
            t_use = round(t_use + 0.01, 2)
            sel = p2t >= t_use
            if ONE_OWNER:
                sel = one_owner(sel, pj, p2t)
            pps = sel.sum() / len(S1c)
        log(f"   {c}: threshold used = {t_use}  -> predicted matches per S1 = {pps:.2f}")
    s1ids = np.array(S1c["entity_id"].to_list(), dtype=object)
    pids = np.array(Pc["entity_id"].to_list(), dtype=object)
    # every pair with a real chance, with its final score: thresholds / rules can be re-applied later WITHOUT re-running
    keep = p2t >= 0.3
    pl.DataFrame({"s1": s1ids[qi[keep]].astype(str), "cand": pids[pj[keep]].astype(str),
                  "p2": p2t[keep].astype(np.float32)}).write_parquet(f"{OUT_DIR}/scores_{c}.parquet")
    cs1, cid, ms1, mid = s1ids[qi], pids[pj], s1ids[qi[sel]], pids[pj[sel]]
    cand_s1.append(cs1)
    cand_id.append(cid)
    match_s1.append(ms1)
    match_id.append(mid)
    _save_atomic(pl.DataFrame({"s1": cs1.astype(str), "id": cid.astype(str)}), _ck(c, "cand"))     # checkpoint: this country is done
    _save_atomic(pl.DataFrame({"s1": ms1.astype(str), "id": mid.astype(str)}), _ck(c, "match"))
    log(f"   {c}: candidates={len(qi)}  matched pairs={int(sel.sum())}  S1 with >=1 match={len(np.unique(qi[sel]))}/{len(S1c)}  [checkpoint saved]")
    del R, S, X2t, p1t, p2t
    gc.collect()


# ==================== WRITE OUTPUTS + RUN THE OFFICIAL VALIDATOR ====================
def write_lists(s1_arr, id_arr, col, path):
    """one row per test S1 entity (in file order); comma-separated ID list; empty when none; tab separated, no quoting"""
    base = pl.DataFrame({"source1_entity_id": ALL_S1_IDS})
    if len(s1_arr):
        pairs = pl.DataFrame({"source1_entity_id": np.concatenate(s1_arr).astype(str), "v": np.concatenate(id_arr).astype(str)}).unique()
        grouped = pairs.group_by("source1_entity_id").agg(pl.col("v").sort().str.join(",").alias(col))
        out = base.join(grouped, on="source1_entity_id", how="left")
    else:
        out = base.with_columns(pl.lit(None, dtype=pl.Utf8).alias(col))
    out = out.with_columns(pl.col(col).fill_null(""))
    out.write_csv(path, separator="\t", quote_style="never")
    return out


m_df = write_lists(match_s1, match_id, "matched_entity_ids", f"{OUT_DIR}/matching_results.tsv")
c_df = write_lists(cand_s1, cand_id, "candidate_entity_ids", f"{OUT_DIR}/candidate_pairs.tsv")
n_c = c_df["candidate_entity_ids"].str.count_matches(",").sum() + (c_df["candidate_entity_ids"] != "").sum()
n_m = m_df["matched_entity_ids"].str.count_matches(",").sum() + (m_df["matched_entity_ids"] != "").sum()
log(f"wrote {len(m_df)} rows to matching_results.tsv ({int(n_m)} matched pairs, {(m_df['matched_entity_ids'] == '').sum()} entities with no match)")
log(f"wrote {len(c_df)} rows to candidate_pairs.tsv ({int(n_c)} candidate pairs, {n_c / max(1, int((c_df['candidate_entity_ids'] != '').sum())):.1f} per S1 entity that has candidates)")

if os.path.exists(VALIDATOR):
    r = subprocess.run([sys.executable, VALIDATOR, "--matching", f"{OUT_DIR}/matching_results.tsv",
                        "--candidate", f"{OUT_DIR}/candidate_pairs.tsv", "--test-dir", TEST_DIR.rstrip("/")] + (["--check-ids"] if CHECK_IDS else []),
                       capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-1500:])
    print("OFFICIAL VALIDATOR:", "PASS" if (r.returncode == 0 and "PASS" in r.stdout) else "FAILED - read the messages above, fix, and re-run")
else:
    print("validator not found at", VALIDATOR, "- fix the VALIDATOR path in the config cell")
log("ALL DONE")
