import time
import numpy as np
import faiss
import psutil
import os
from tabulate import tabulate


def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def generate_data(n_db=1_000_000, n_query=1000, dim=768, seed=42):
    rng = np.random.default_rng(seed)
    xb = rng.random((n_db, dim), dtype=np.float32)
    xq = rng.random((n_query, dim), dtype=np.float32)
    return xb, xq


def compute_recall(gt, result, k=10):
    recalls = []
    for i in range(len(gt)):
        hit = len(set(gt[i]) & set(result[i]))
        recalls.append(hit / k)
    return np.mean(recalls)


def bench_flat(xb, xq, k=10):
    mem_before = get_memory_mb()
    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)
    mem_after = get_memory_mb()

    start = time.perf_counter()
    D, I = index.search(xq, k)
    elapsed = (time.perf_counter() - start) / len(xq) * 1000

    return {
        "index": "Flat",
        "params": "-",
        "recall@10": 1.0,
        "latency_ms": round(elapsed, 2),
        "memory_mb": round(mem_after - mem_before, 1),
    }, I


def bench_ivf(xb, xq, gt, k=10, nlist=1024, nprobe_list=None):
    if nprobe_list is None:
        nprobe_list = [1, 8, 32, 64, 128]

    quantizer = faiss.IndexFlatL2(xb.shape[1])
    index = faiss.IndexIVFFlat(quantizer, xb.shape[1], nlist)
    index.train(xb)

    mem_before = get_memory_mb()
    index.add(xb)
    mem_after = get_memory_mb()

    results = []
    for nprobe in nprobe_list:
        index.nprobe = nprobe
        start = time.perf_counter()
        D, I = index.search(xq, k)
        elapsed = (time.perf_counter() - start) / len(xq) * 1000
        recall = compute_recall(gt, I, k)

        results.append({
            "index": "IVF",
            "params": f"nlist={nlist}, nprobe={nprobe}",
            "recall@10": round(recall, 4),
            "latency_ms": round(elapsed, 2),
            "memory_mb": round(mem_after - mem_before, 1),
        })
    return results


def bench_hnsw(xb, xq, gt, k=10, M=32, ef_construction=200, ef_search_list=None):
    if ef_search_list is None:
        ef_search_list = [16, 32, 64, 128, 256]

    index = faiss.IndexHNSWFlat(xb.shape[1], M)
    index.hnsw.efConstruction = ef_construction

    mem_before = get_memory_mb()
    index.add(xb)
    mem_after = get_memory_mb()

    results = []
    for efs in ef_search_list:
        index.hnsw.efSearch = efs
        start = time.perf_counter()
        D, I = index.search(xq, k)
        elapsed = (time.perf_counter() - start) / len(xq) * 1000
        recall = compute_recall(gt, I, k)

        results.append({
            "index": "HNSW",
            "params": f"M={M}, efSearch={efs}",
            "recall@10": round(recall, 4),
            "latency_ms": round(elapsed, 2),
            "memory_mb": round(mem_after - mem_before, 1),
        })
    return results


def main():
    print("Generating 1M vectors (dim=768)...")
    xb, xq = generate_data()

    print("Building ground truth (Flat)...")
    flat_result, gt = bench_flat(xb, xq)

    print("Benchmarking IVF...")
    ivf_results = bench_ivf(xb, xq, gt)

    print("Benchmarking HNSW...")
    hnsw_results = bench_hnsw(xb, xq, gt)

    all_results = [flat_result] + ivf_results + hnsw_results
    headers = ["index", "params", "recall@10", "latency_ms", "memory_mb"]
    table = [[r[h] for h in headers] for r in all_results]

    print("\n" + tabulate(table, headers=headers, tablefmt="github"))

    with open("results.md", "w") as f:
        f.write("# FAISS Index Benchmark\n\n")
        f.write(f"- **Vectors**: 1,000,000 × 768\n")
        f.write(f"- **Queries**: 1,000\n")
        f.write(f"- **k**: 10\n\n")
        f.write(tabulate(table, headers=headers, tablefmt="github"))
        f.write("\n")

    print("\nResults saved to results.md")


if __name__ == "__main__":
    main()
