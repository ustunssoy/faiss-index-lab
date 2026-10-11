# FAISS Index Lab

Flat, IVF ve HNSW indekslerini 1M vektör (768 boyut) üzerinde karşılaştırır.

## Metrikler
- Recall@10
- Sorgu gecikmesi (ms)
- Bellek kullanımı (MB)

## Çalıştırma

```bash
pip install -r requirements.txt
python benchmark.py
```

### Docker

```bash
docker build -t faiss-index-lab .
docker run faiss-index-lab
```

## Parametre Analizi

| Parametre | İndeks | Etki |
|-----------|--------|------|
| `nlist` | IVF | Küme sayısı. ↑ → daha hızlı ama recall düşer |
| `nprobe` | IVF | Taranan küme sayısı. ↑ → daha yavaş ama recall artar |
| `efSearch` | HNSW | Arama genişliği. ↑ → daha yavaş ama recall artar |
