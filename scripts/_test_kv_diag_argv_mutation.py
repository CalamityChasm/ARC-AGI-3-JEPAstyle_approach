import json, re, sys
src = open("kaggle_submission_duck_nvfp4/notebook_kv_diag/kv_benchmark_cell.py", encoding="utf-8").read()
# extract just the pure functions we want to test
ns = {}
for name in ("_drop_flag", "_set_flag", "_mutate"):
    m = re.search(r"^def %s\(.*?(?=^\S|\Z)" % name, src, re.S | re.M)
    assert m, name
    exec(m.group(0), ns)
_mutate = ns["_mutate"]

BASE = json.load(open(r"C:\Users\desktop-06\AppData\Local\Temp\nvfp4_out\vllm-server-identity.json"))["argv"]
GB = 1024**3
CONFIGS = [
    ("seqs40", {"set": [("--max-num-seqs", 40)]}),
    ("kvfp8-seqs40", {"set": [("--kv-cache-dtype", "fp8"), ("--max-num-seqs", 40)]}),
    ("kv8-seqs40", {"set": [("--kv-cache-memory-bytes", 8*GB), ("--max-num-seqs", 40)]}),
    ("kv8-fp8-seqs40", {"set": [("--kv-cache-memory-bytes", 8*GB), ("--kv-cache-dtype","fp8"), ("--max-num-seqs", 40)]}),
    ("seqs16", {"set": [("--max-num-seqs", 16)]}),
    ("prefix-seqs40", {"drop": [("--no-enable-prefix-caching", False)],
                       "set": [("--enable-prefix-caching", None), ("--max-num-seqs", 40)]}),
    ("kv12-seqs40", {"set": [("--kv-cache-memory-bytes", 12*GB), ("--max-num-seqs", 40)]}),
]
def val(a, f):
    return a[a.index(f)+1] if f in a else None

fail = 0
print("baseline: seqs=%s kvbytes=%s kvdtype=%s no-prefix=%s prefix=%s len=%d" % (
    val(BASE,"--max-num-seqs"), val(BASE,"--kv-cache-memory-bytes"),
    val(BASE,"--kv-cache-dtype"), "--no-enable-prefix-caching" in BASE,
    "--enable-prefix-caching" in BASE, len(BASE)))
for name, spec in CONFIGS:
    a = _mutate(BASE, spec)
    # invariants: every flag appears at most once; model path preserved; all
    # untouched baseline tokens survive
    dupes = [f for f in set(t for t in a if t.startswith("--")) if a.count(f) > 1]
    ok = not dupes and a[4] == BASE[4]
    touched = set()
    for f, _ in (spec.get("set") or []): touched.add(f)
    for f, _ in (spec.get("drop") or []): touched.add(f)
    missing = [t for i, t in enumerate(BASE)
               if t.startswith("--") and t not in touched and t not in a]
    ok = ok and not missing
    print("%-16s seqs=%-4s kvbytes=%-12s kvdtype=%-5s prefix_on=%-5s no_prefix=%-5s len=%d  %s%s%s" % (
        name, val(a,"--max-num-seqs"), val(a,"--kv-cache-memory-bytes"),
        val(a,"--kv-cache-dtype"), "--enable-prefix-caching" in a,
        "--no-enable-prefix-caching" in a, len(a),
        "OK" if ok else "FAIL", (" dupes=%s"%dupes) if dupes else "",
        (" lost=%s"%missing) if missing else ""))
    fail += 0 if ok else 1
# explicit assertions on the two that matter most
a = _mutate(BASE, dict(CONFIGS[5][1]))
assert "--enable-prefix-caching" in a and "--no-enable-prefix-caching" not in a, "prefix flip broken"
a = _mutate(BASE, dict(CONFIGS[0][1]))
assert val(a, "--max-num-seqs") == "40", "seqs not set"
assert val(a, "--kv-cache-memory-bytes") == "5368709120", "seqs40 must not disturb kv"
print("\nasserts passed; failures:", fail)
sys.exit(1 if fail else 0)
