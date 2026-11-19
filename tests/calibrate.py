"""
Offline threshold calibration for Semantic-Shift.

Scores every benchmark pair with the SAME two models the server uses
(all-MiniLM-L6-v2 + quora-distilroberta-base). No server, no Groq key, no LLM calls.

Run from the project folder:
    python tests/calibrate.py

It prints (1) the real cosine + cross-encoder score of every paraphrase and trap pair,
(2) a grid of (cosine threshold, cross-encoder threshold) -> paraphrase hit rate / trap false positives.
Choose settings from this table, apply them, then re-run tests/benchmark.py to confirm.

Approximation: each query is compared with its own canonical question only. The live server compares
against the whole cache, so always confirm with a fresh benchmark run.
"""

HIGH = 0.88  # server skips the cross-encoder at or above this cosine (backend/verification.py)


def is_hit(cos, ce, t_cos, t_ce):
    return cos >= t_cos and (cos >= HIGH or ce >= t_ce)


def sweep(pairs, cos_grid, ce_grid):
    paras = [p for p in pairs if p["kind"] == "para"]
    traps = [p for p in pairs if p["kind"] == "trap"]
    rows = []
    for t_cos in cos_grid:
        for t_ce in ce_grid:
            hits = sum(is_hit(p["cos"], p["ce"], t_cos, t_ce) for p in paras)
            fps = sum(is_hit(p["cos"], p["ce"], t_cos, t_ce) for p in traps)
            rows.append({"t_cos": t_cos, "t_ce": t_ce,
                         "hit_pct": round(100 * hits / len(paras), 1), "fp": fps, "traps": len(traps)})
    return rows


def main():
    from sentence_transformers import CrossEncoder, SentenceTransformer
    try:
        from tests.benchmark import INTENTS
    except ImportError:
        from benchmark import INTENTS

    emb = SentenceTransformer("all-MiniLM-L6-v2")
    ce = CrossEncoder("cross-encoder/quora-distilroberta-base")

    pairs = []
    for canon, paras, trap in INTENTS:
        for kind, q in [("para", p) for p in paras] + [("trap", trap)]:
            a, b = emb.encode([canon, q], normalize_embeddings=True)
            pairs.append({"kind": kind, "canon": canon, "q": q,
                          "cos": float(a @ b), "ce": float(ce.predict([(q, canon)])[0])})

    print("\n=== TRAPS (should be rejected) ===")
    for p in sorted((p for p in pairs if p["kind"] == "trap"), key=lambda x: -x["ce"]):
        print(f"cos={p['cos']:.3f}  ce={p['ce']:.3f}  {p['q']}  <>  {p['canon']}")
    print("\n=== PARAPHRASES (should be accepted), lowest CE first ===")
    for p in sorted((p for p in pairs if p["kind"] == "para"), key=lambda x: x["ce"])[:15]:
        print(f"cos={p['cos']:.3f}  ce={p['ce']:.3f}  {p['q']}  <>  {p['canon']}")

    cos_grid = [0.72, 0.74, 0.76, 0.78, 0.80]
    ce_grid = [round(0.25 + 0.05 * i, 2) for i in range(15)]  # 0.25 .. 0.95
    rows = sweep(pairs, cos_grid, ce_grid)

    print("\n=== GRID: cosine thr, CE thr -> paraphrase hit %, trap false positives ===")
    for r in rows:
        print(f"cos>={r['t_cos']:.2f}  ce>={r['t_ce']:.2f}  hits={r['hit_pct']:5.1f}%  false_pos={r['fp']}/{r['traps']}")

    safe = [r for r in rows if r["fp"] == 0]
    if safe:
        best = max(safe, key=lambda r: (r["hit_pct"], -r["t_cos"]))
        print(f"\nBEST with 0 false positives: cosine>={best['t_cos']}, ce>={best['t_ce']} -> {best['hit_pct']}% paraphrase hits")
    else:
        print("\nNo (cosine, CE) setting reaches 0 false positives: the cross-encoder cannot separate all traps.")
        best = min(rows, key=lambda r: (r["fp"], -r["hit_pct"]))
        print(f"Lowest false positives: cosine>={best['t_cos']}, ce>={best['t_ce']} -> {best['fp']}/{best['traps']} FPs, {best['hit_pct']}% hits")


if __name__ == "__main__":
    main()
