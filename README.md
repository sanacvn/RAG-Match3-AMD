# RAG Match3 AMD

RAG Match3 AMD adalah aplikasi tanya jawab dokumen yang dibuat untuk AMD AI League
Match 3. Pengguna cukup memberikan satu file atau sebuah folder, lalu mengajukan
pertanyaan dengan bahasa sehari-hari. Sistem mencari bagian dokumen yang paling
relevan, menyusun jawaban dari bagian tersebut, dan mengembalikan nama file sumbernya.

Versi saat ini memakai Chroma Cloud untuk embedding dan pencarian, serta OpenAI untuk
membaca gambar dan menyusun jawaban. Model PyTorch lokal masih tersedia sebagai jalur
eksperimen, tetapi bukan konfigurasi utama yang dijelaskan di README ini.

## Cara kerja

```text
Dokumen
   │
   ├─ parsing teks, tabel, halaman, dan gambar
   ├─ chunking + metadata sumber
   ▼
Chroma Cloud
   ├─ Qwen dense embedding
   ├─ SPLADE sparse embedding
   └─ hybrid search dengan RRF dan GroupBy
   ▼
Potongan dokumen yang relevan
   ▼
OpenAI GPT-4.1 Mini
   └─ jawaban + citation ke file asli
```

Saat indexing, teks dipecah menjadi chunk dan dikirim ke Chroma Cloud. Chroma membuat
dense embedding dengan Qwen dan sparse embedding dengan SPLADE. Hasil keduanya
digabungkan memakai Reciprocal Rank Fusion (RRF). `GroupBy` membatasi hasil dari
dokumen yang sama agar konteks tidak dipenuhi oleh potongan yang berulang.

Jika input berupa gambar atau halaman PDF scan, OpenAI vision dipakai untuk membaca
isinya saat indexing. Ketika pengguna bertanya, hanya konteks hasil pencarian yang
diteruskan ke OpenAI untuk menyusun jawaban. Citation diperiksa kembali sebelum hasil
ditulis.

## Fitur

- Hybrid search: dense Qwen + sparse SPLADE + RRF.
- Mendukung PDF, DOCX, XLSX, CSV, TXT, LOG, Python, PNG, JPG, dan JPEG.
- Membaca paragraf, tabel, seluruh sheet spreadsheet, gambar, dan halaman PDF scan.
- Menyimpan provenance seperti file, halaman, sheet, baris, blok, dan indeks chunk.
- Memisahkan collection berdasarkan scope, corpus, dan versi schema.
- Memakai ulang indeks jika dokumen belum berubah.
- Menolak citation atau kutipan yang tidak didukung oleh evidence.
- Menyediakan migrasi snapshot dari indeks lokal ke Chroma Cloud.

## Teknologi

| Komponen | Pilihan saat ini |
|---|---|
| Vector database | Chroma Cloud |
| Dense embedding | Chroma Cloud Qwen |
| Sparse embedding | Chroma Cloud SPLADE |
| Penggabungan hasil | Reciprocal Rank Fusion, bobot 70/30 |
| Deduplication | GroupBy berdasarkan dokumen |
| LLM jawaban | OpenAI `gpt-4.1-mini` |
| Vision | OpenAI `gpt-4.1-mini` |
| Runtime | Python 3.10+ |

## Persiapan

Yang dibutuhkan:

- Python 3.10 atau lebih baru.
- Chroma Cloud API key dan database aktif.
- OpenAI API key dari project yang memiliki kredit API.
- Koneksi ke `api.trychroma.com`, `embed.trychroma.com`, dan `api.openai.com`.

Clone repository dan buat virtual environment:

```bash
git clone https://github.com/sanacvn/RAG-Match3-AMD.git
cd RAG-Match3-AMD

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cloud.txt
python -m pip check
```

Salin konfigurasi contoh:

```bash
cp .env.example .env
```

Isi `.env` dengan credential milik sendiri:

```dotenv
RAG_BACKEND=openai
RAG_SEARCH_BACKEND=chroma
RAG_CONFIG=configs/cloud.json

OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
OPENAI_VISION_MODEL=gpt-4.1-mini

CHROMA_HOST=api.trychroma.com
CHROMA_TENANT=11afbef7-38c1-4590-93de-3dc5dc9851f9
CHROMA_DATABASE=RAG-MATCH3-AMD
CHROMA_SCOPE=aliaatikah12
CHROMA_API_KEY=

RAG_DEVICE=cpu
RAG_REQUIRE_ROCM=false
```

`.env` tidak masuk Git. Jangan menaruh API key di source code, commit, screenshot,
atau output terminal yang dibagikan.

## Menjalankan project

Untuk memastikan koneksi Chroma bekerja:

```bash
./.venv/bin/python scripts/check_chroma.py
```

Script tersebut membuat collection sementara, menguji upload dan search, lalu
menghapus collection yang dibuatnya.

Index dokumen dan ajukan pertanyaan pertama:

```bash
./.venv/bin/python ask.py --reindex \
  '/path/AI_Eng.docx' \
  'apa materi di bulan 1 hari ke 8?'
```

Untuk pertanyaan berikutnya pada dokumen yang sama, `--reindex` tidak diperlukan:

```bash
./.venv/bin/python ask.py \
  '/path/AI_Eng.docx' \
  'apa tugas yang harus dikerjakan setelah materi tersebut?'
```

`ask.py` menampilkan jawaban, sumber, dan lokasi file JSON. File asli tidak diubah.
Jika input berupa satu file, project membuat salinan kerja di dalam `state/ask-inputs`.

Contoh output:

```text
Jawaban: Materi bulan pertama hari ke-8 membahas ...
Sumber:
- /path/AI_Eng.docx
JSON: output/ask_<id>_output.json
```

## Indexing dan collection

Chroma Cloud membatasi ukuran satu dokumen sebesar 16 KiB. Project memakai batas
12 KiB UTF-8 dan memotong teks berdasarkan baris sebelum mengunggahnya. Setiap chunk
membawa `document_id`, `chunk_index`, `chunk_id`, path sumber, dan lokasi asal.

Nama collection dibentuk dari:

- `CHROMA_SCOPE`, untuk memisahkan pengguna atau organisasi;
- identitas corpus, berdasarkan lokasi corpus atau `RAG_CORPUS_ID`;
- versi schema embedding.

Setiap versi dokumen mendapat revision tersendiri. Search selalu difilter ke revision
aktif. Indexing ulang dengan isi yang sama tidak menggandakan record, sedangkan
perubahan dokumen menghasilkan revision baru.

## Konfigurasi penting

| Variabel | Keterangan |
|---|---|
| `OPENAI_API_KEY` | Key OpenAI untuk jawaban dan vision |
| `OPENAI_MODEL` | Model pembuat jawaban, default `gpt-4.1-mini` |
| `OPENAI_VISION_MODEL` | Model pembaca gambar, default `gpt-4.1-mini` |
| `CHROMA_API_KEY` | Key Chroma Cloud untuk storage, embedding, dan search |
| `CHROMA_SCOPE` | Pemisah collection antar pengguna/organisasi |
| `RAG_CORPUS_ID` | Identitas corpus tetap, opsional |
| `CHROMA_BATCH_SIZE` | Jumlah record per batch, default 64 |
| `CHROMA_CHUNK_BYTES` | Batas chunk Cloud, default 12288 byte |
| `CHROMA_CHUNKS_PER_DOCUMENT` | Maksimum chunk per dokumen setelah GroupBy, default 3 |
| `RAG_STATE_DIR` | Snapshot dan salinan kerja, default `state` |
| `RAG_OUTPUT_DIR` | Hasil jawaban JSON, default `output` |
| `RAG_QUERY_TIMEOUT` | Batas waktu query, default 26 detik |
| `RAG_STARTUP_TIMEOUT` | Batas startup dan indexing, default 570 detik |

Nilai environment yang sudah diekspor di terminal memiliki prioritas lebih tinggi
daripada `.env`. Restart worker setelah mengganti backend, model, endpoint, atau API
key agar konfigurasi baru benar-benar digunakan.

## Migrasi indeks lama

Snapshot lokal dapat dipindahkan ke Chroma tanpa menyalin vektor lama. Script migrasi
membaca evidence, membuat chunk Cloud, lalu menghasilkan embedding Qwen dan SPLADE.

```bash
./.venv/bin/python scripts/migrate_chroma.py \
  --snapshot state/index.json \
  --dry-run

./.venv/bin/python scripts/migrate_chroma.py \
  --snapshot state/index.json \
  --output state/chroma/index.json
```

File snapshot asal tidak ditimpa.

## Struktur repository

```text
ask.py                  CLI untuk satu file atau folder
app.py                  client worker dan kontrak index/query
worker.py               proses persisten untuk pipeline aktif
rag/chroma_search.py    schema, upload, hybrid search, dan GroupBy
rag/openai_models.py    jawaban dan vision melalui OpenAI API
rag/parsers.py          parser dokumen, chunking, dan provenance
rag/evidence.py         validasi jawaban dan citation
rag/pipeline.py         indexing, restore, fingerprint, dan query
configs/cloud.json      konfigurasi retrieval Cloud
scripts/                check, migrasi, evaluasi, dan provisioning
tests/                  unit dan integration tests
```

## Pengujian

Jalankan seluruh test suite dari root repository:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

Saat ini terdapat 37 test untuk parser, lifecycle worker, Chroma SDK, schema
dense/sparse, RRF, GroupBy, revision isolation, migrasi snapshot, OpenAI response,
vision payload, timeout, serta validasi citation. Test integrasi memakai boundary HTTP
lokal sehingga tidak memakai kredit Chroma atau OpenAI.

## Masalah yang sering muncul

### Worker configuration changed

Worker masih memakai konfigurasi lama. Cari PID worker milik project lalu hentikan:

```bash
pgrep -fl 'RAG-Match3-AMD/worker.py'
kill -TERM <PID>
```

Perintah `ask.py` berikutnya akan menjalankan worker baru.

### OpenAI `insufficient_quota`

API key dikenali, tetapi project OpenAI yang menerbitkan key tidak memiliki kredit.
Tambahkan saldo di OpenAI Platform atau gunakan key dari project yang memiliki saldo.
Langganan ChatGPT tidak termasuk kredit OpenAI API.

### Index gagal

Lihat log worker:

```bash
tail -n 80 state/worker.log
```

Log biasanya menunjukkan apakah masalah berasal dari credential, quota, parsing,
atau koneksi ke layanan Cloud.

## Catatan penggunaan data dan biaya

Teks chunk beserta metadata disimpan dan diproses oleh Chroma Cloud. Gambar dokumen
dan konteks hasil retrieval diproses oleh OpenAI. Pastikan dokumen yang digunakan
memang boleh dikirim ke kedua layanan tersebut.

OpenAI dikenai biaya ketika vision membaca gambar dan ketika model menyusun jawaban.
Embedding Qwen/SPLADE dijalankan oleh Chroma Cloud dan mengikuti quota atau paket
Chroma. Setelah indeks pertama selesai, hindari `--reindex` jika isi dokumen tidak
berubah.

## Referensi Chroma

- [Chroma documentation index](https://docs.trychroma.com/llms.txt)
- [Hybrid search](https://docs.trychroma.com/cloud/search-api/hybrid-search.md)
- [Sparse vector search](https://docs.trychroma.com/cloud/schema/sparse-vector-search.md)
- [Chroma Cloud Qwen](https://docs.trychroma.com/integrations/embedding-models/chroma-cloud-qwen.md)
- [Chroma Cloud SPLADE](https://docs.trychroma.com/integrations/embedding-models/chroma-cloud-splade.md)
- [GroupBy](https://docs.trychroma.com/cloud/search-api/group-by.md)
- [Chunking guide](https://docs.trychroma.com/guides/build/chunking.md)
