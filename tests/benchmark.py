"""
Semantic-Shift benchmark: produces REAL numbers for the video.
Run the server first (uvicorn backend.main:app or python app.py), with an EMPTY cache (delete cache_store.db), then:
    python tests/benchmark.py --url http://127.0.0.1:8000 --out tests/results_run3_online_strict.json --label "Run 3" --note "online deployment, STRICT"
Phases (each query is sent once, in this order):
  A warm      canonical question per intent          expect MISS (LLM call)
  B paraphase 3 paraphrases per intent               expect HIT  (same intent)
  C trap      same topic, DIFFERENT intent           expect MISS (a HIT = wrong answer served)
  D repeat    canonical again                        expect HIT  (exact match)
Reports: paraphrase hit rate, trap false-positive rate, LLM calls avoided, latency by path.
Tell viewers the dataset size (printed at the end) and the machine you ran it on.
"""
import argparse, json, statistics, time, platform, datetime
import httpx

INTENTS = [
    ("How do I reset my password?",
     ["What are the steps to recover my password?", "I forgot my password, how can I get back in?", "Can you help me change my password if I can't log in?"],
     "How do I delete my account?"),
    ("How do I cancel my subscription?",
     ["What is the process to end my subscription?", "I want to stop my monthly plan, how?", "Steps for unsubscribing from the paid plan?"],
     "How do I upgrade my subscription?"),
    ("What is machine learning?",
     ["Can you explain what machine learning means?", "Define machine learning in simple words.", "Explain machine learning to a beginner."],
     "What is the history of machine learning?"),
    ("How do I install Python on Windows?",
     ["What are the steps to set up Python on a Windows PC?", "Guide me through installing Python on Windows.", "How can I get Python running on Windows 11?"],
     "How do I uninstall Python on Windows?"),
    ("What is the difference between a list and a tuple in Python?",
     ["How do lists and tuples differ in Python?", "Python list vs tuple, what changes?", "Explain how a tuple is different from a list in Python."],
     "What is the difference between a list and a set in Python?"),
    ("How does HTTPS keep data secure?",
     ["Why is HTTPS secure?", "Explain how HTTPS protects my data.", "What makes an HTTPS connection safe?"],
     "How do I get an HTTPS certificate for my site?"),
    ("How do I center a div in CSS?",
     ["What is the way to horizontally and vertically center a div using CSS?", "CSS: how to put a div in the middle of the page?", "Best method to center a div in CSS?"],
     "How do I make a div responsive in CSS?"),
    ("What is an API?",
     ["Can you explain what an API is?", "Explain the meaning of API.", "What does API stand for and what does it do?"],
     "How do I build an API?"),
    ("How do I undo my last git commit?",
     ["What is the command to revert my most recent git commit?", "I committed by mistake, how can I undo it in git?", "How can I roll back the last commit in git?"],
     "How do I squash my last git commits?"),
    ("What is the capital of France?",
     ["Which city is France's capital?", "Tell me the capital city of France.", "France's capital is which city?"],
     "What is the population of France?"),
    ("How do I lose weight safely?",
     ["What is a safe way to reduce body weight?", "Healthy ways to lose weight?", "How can I slim down without harming my health?"],
     "How do I gain muscle safely?"),
    ("How do I write a good resume?",
     ["What makes a strong resume and how do I write one?", "Tips for writing an effective resume?", "How should I structure my resume?"],
     "How do I write a good cover letter?"),
    ("What is Docker used for?",
     ["Why do developers use Docker?", "Explain what Docker is for.", "What problem does Docker solve?"],
     "How do I install Docker on Ubuntu?"),
    ("How does a binary search work?",
     ["Explain the binary search algorithm.", "How does binary search find an item?", "Walk me through binary search."],
     "How does a linear search work?"),
    ("What is the best way to learn programming?",
     ["How should a beginner start learning to code?", "What is the most effective way to learn programming?", "Advice for learning programming from scratch?"],
     "What is the best programming language to learn first?"),
]


def ask(client, url, q, max_retries=3):
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        r = client.post(f"{url}/v1/chat/completions", json={"messages": [{"role": "user", "content": q}]}, timeout=60)
        wall = (time.perf_counter() - t0) * 1000
        if r.status_code == 502 and attempt < max_retries - 1:
            print(f"\n[Groq 429/502 rate limit] Backing off 12s before retry {attempt+1}/{max_retries} for query: {q[:30]}...")
            time.sleep(12)
            continue
        r.raise_for_status()
        d = r.json()
        dbg = d.get("debug", {})
        cached = bool(d.get("cached"))
        if not cached:
            time.sleep(2.0)  # pace LLM calls to avoid Groq rate-limiting inflation
        sim = dbg.get("similarity")
        if sim is None:
            sim = dbg.get("best_score")
        return {"q": q, "cached": cached, "latency_ms": d.get("latency_ms", wall),
                "wall_ms": wall, "method": dbg.get("method") or dbg.get("reason"),
                "similarity": sim, "ce": dbg.get("crossencoder_score"), "matched": dbg.get("matched_query")}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else None


def stats(rows):
    # Use server-side latency_ms (internal pipeline time), fallback to wall_ms
    L = [r.get("latency_ms", r["wall_ms"]) for r in rows]
    return {"n": len(L), "median_ms": round(statistics.median(L), 2) if L else None,
            "p95_ms": round(pct(L, 0.95), 2) if L else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--label", default="", help="Short run label (e.g. 'Run 3')")
    ap.add_argument("--note", default="", help="Free-text note about settings/environment")
    a = ap.parse_args()
    res = {"A_warm": [], "B_paraphrase": [], "C_trap": [], "D_repeat": []}
    with httpx.Client() as c:
        # Capture backend config at start so the results file records exact settings used.
        try:
            cfg_resp = c.get(f"{a.url}/config", timeout=10)
            backend_config = cfg_resp.json() if cfg_resp.is_success else {"error": f"HTTP {cfg_resp.status_code}"}
        except Exception as exc:
            backend_config = {"error": str(exc)}

        print(f"Connecting to {a.url}...")
        print("Starting Phase A (warm canonical queries)...")
        for i, (canon, _, _) in enumerate(INTENTS, 1):
            print(f"  [A {i}/{len(INTENTS)}] {canon[:40]}...", end=" ", flush=True)
            res_item = ask(c, a.url, canon)
            print("HIT" if res_item["cached"] else "MISS")
            res["A_warm"].append(res_item)

        print("\nStarting Phase B (paraphrase queries)...")
        total_paras = sum(len(p) for _, p, _ in INTENTS)
        idx = 1
        for _, paras, _ in INTENTS:
            for p in paras:
                print(f"  [B {idx}/{total_paras}] {p[:40]}...", end=" ", flush=True)
                res_item = ask(c, a.url, p)
                print("HIT" if res_item["cached"] else "MISS")
                res["B_paraphrase"].append(res_item)
                idx += 1

        print("\nStarting Phase C (adversarial trap queries)...")
        for i, (_, _, trap) in enumerate(INTENTS, 1):
            print(f"  [C {i}/{len(INTENTS)}] {trap[:40]}...", end=" ", flush=True)
            res_item = ask(c, a.url, trap)
            print("HIT (trap)" if res_item["cached"] else "MISS (rejected)")
            res["C_trap"].append(res_item)

        print("\nStarting Phase D (exact repeat queries)...")
        for i, (canon, _, _) in enumerate(INTENTS, 1):
            print(f"  [D {i}/{len(INTENTS)}] {canon[:40]}...", end=" ", flush=True)
            res_item = ask(c, a.url, canon)
            print("HIT" if res_item["cached"] else "MISS")
            res["D_repeat"].append(res_item)

    A, B, C, D = (res[k] for k in ("A_warm", "B_paraphrase", "C_trap", "D_repeat"))
    total = sum(len(v) for v in res.values())
    hits = sum(r["cached"] for v in res.values() for r in v)
    para_hits = sum(r["cached"] for r in B)
    trap_hits = [r for r in C if r["cached"]]
    exact = [r for r in D if r["cached"]]
    summary = {
        "label": a.label,
        "note": a.note,
        "url": a.url,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "backend_config": backend_config,
        "machine": platform.platform(), "processor": platform.processor(),
        "total_requests": total,
        "paraphrases": len(B), "paraphrase_hit_rate_pct": round(100 * para_hits / len(B), 1),
        "traps": len(C), "trap_false_positives": len(trap_hits),
        "trap_false_positive_rate_pct": round(100 * len(trap_hits) / len(C), 1),
        "exact_repeat_hit_rate_pct": round(100 * len(exact) / len(D), 1),
        "llm_calls_made": total - hits, "llm_calls_avoided": hits,
        "llm_calls_avoided_pct": round(100 * hits / total, 1),
        "latency_miss": stats([r for v in res.values() for r in v if not r["cached"]]),
        "latency_paraphrase_hit": stats([r for r in B if r["cached"]]),
        "latency_exact_hit": stats(exact),
        "false_positive_examples": [{"asked": r["q"], "matched": r["matched"], "similarity": r["similarity"], "ce": r.get("ce")} for r in trap_hits],
        "missed_paraphrases": [r["q"] for r in B if not r["cached"]],
    }
    json.dump({"summary": summary, "rows": res}, open(a.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("\nDataset: %d intents, %d requests. Say this on screen. Convert calls avoided to $ using your real per-call cost from the Groq console." % (len(INTENTS), total))


if __name__ == "__main__":
    main()
