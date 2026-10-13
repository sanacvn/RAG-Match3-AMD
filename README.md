# AMD AI League Match 3 RAG

Pipeline RAG dengan pencarian hybrid Chroma Cloud, corpus dinamis, citation berbukti,
dan worker persisten. Mode pencarian lokal/offline tetap tersedia.
Implementasi model nyata memakai Transformers lokal; belum diverifikasi pada GPU AMD.
Container submission **belum siap dinyatakan lulus**: base image wajib, aturan umum,
starter kit, aturan bobot/network, dan hardware evaluator belum tersedia.

## Chroma Cloud: pencarian dan embedding

`.env.example` menyiapkan pencarian Cloud: **Chroma Cloud Qwen** untuk dense embedding,
**Chroma Cloud SPLADE** untuk sparse/lexical embedding, lalu **RRF 70/30** untuk
menggabungkan keduanya. Generator jawaban dan vision tetap model PyTorch lokal;
`RAG_GENERATOR_PATH` dan `RAG_VISION_PATH` tetap menunjuk folder bobot. Embedding BGE
lokal tidak dimuat dalam mode Cloud. Chroma tidak mempercepat generasi jawaban LLM;
timeout CPU 26 detik tetap berlaku dan perlu benchmark terpisah.

Dengan virtualenv dan PyTorch yang sudah tersedia, pasang dependency dari root repo:

```bash
./.venv/bin/python -m pip install -r requirements.txt -r requirements-chroma.txt
./.venv/bin/python -m pip check
```

Untuk image AMD resmi, pertahankan torch/torchvision ROCm; requirements Chroma tidak
memasang keduanya. Jangan gunakan `--no-deps` untuk `requirements-chroma.txt` karena
SDK membutuhkan dependency tambahan. Python 3.12 Linux digunakan untuk validasi
integrasi ini; Mac/GPU dan kecepatan model lokal belum diuji di cloud.

Tambahkan konfigurasi berikut ke `.env` lokal yang diabaikan Git; simpan key hanya
di file itu atau environment settings. Ambil key dari
[halaman SDK database](https://www.trychroma.com/aliaatikah12/aws-us-east-1/RAG-MATCH3-AMD/sdk?tab=env).

```dotenv
RAG_SEARCH_BACKEND=chroma
RAG_CONFIG=configs/chroma.json
CHROMA_HOST=api.trychroma.com
CHROMA_TENANT=11afbef7-38c1-4590-93de-3dc5dc9851f9
CHROMA_DATABASE=RAG-MATCH3-AMD
CHROMA_SCOPE=aliaatikah12
CHROMA_API_KEY=<isi-key-Chroma-di-mesinmu>
```

Untuk Mac CPU gunakan `RAG_DEVICE=cpu` dan `RAG_REQUIRE_ROCM=false`. Setelah konfigurasi
berubah, hentikan worker milik repo ini dengan SIGTERM ke PID dari `pgrep -fl worker.py`.
Lalu validasi Cloud dan ajukan pertanyaan:

```bash
./.venv/bin/python scripts/check_chroma.py
./.venv/bin/python ask.py '/path/AI_Eng.docx' 'apa materi di bulan 1 hari ke 8?'
```

`check_chroma.py` mengunggah dua dokumen sintetis, memeriksa read/search, lalu menghapus
collection unik yang dibuatnya sendiri. Tidak memuat generator atau vision. Endpoint
HTTPS yang diperlukan adalah `api.trychroma.com` untuk database dan
`embed.trychroma.com` untuk embedding. Jika memakai proxy-secret di cloud, key harus
diizinkan pada **kedua** tujuan tersebut. TLS verification dan proxy bawaan dipertahankan.

Collection dipisahkan menurut `CHROMA_SCOPE` (pengguna/organisasi), identitas corpus,
dan versi schema embedding. Berikan scope berbeda untuk data yang tidak boleh
bercampur; pemisahan collection bukan pengganti kontrol akses tenant/API key. Opsional
`RAG_CORPUS_ID` memberikan identitas tetap saat folder dipindah antarmesin; gunakan ID
unik per corpus, bukan satu ID yang sama untuk semua file `ask`.

Setiap collection dibuat dengan `get_or_create_collection` dan schema dense+sparse.
Chunk memiliki `document_id`, `chunk_index`, path asli, ID evidence dan lokasi sumber.
Chunk besar dipotong berdasarkan baris dengan batas default **12 KiB UTF-8**, termasuk
pemotongan baris panjang pada batas karakter, agar tidak melampaui limit **16 KiB**.
Search memakai `GroupBy(document_id)` untuk mengambil maksimum tiga chunk per dokumen
(ubah `CHROMA_CHUNKS_PER_DOCUMENT` bila perlu lebih banyak bukti dari satu dokumen).

Versi corpus diberi revision dan selalu difilter saat search. Upload yang gagal tidak
mengaktifkan indeks parsial. Indexing ulang versi yang sama hanya mengunggah chunk
yang belum ada; `ask` memakai ulang indeks bila inventory corpus belum berubah.
`ask --reindex` memaksa pembangunan ulang. Versi berbeda di-embed ulang dan revision
lama tetap tersimpan sehingga tidak menghapus data milik proses lain; kelola retensi
collection di dashboard untuk membatasi penyimpanan. Benchmark kualitas, waktu dan
biaya Cloud pada corpus sendiri; belum ada klaim bahwa API Cloud lebih cepat dari
pencarian NumPy lokal untuk corpus kecil. Teks chunk dikirim ke Chroma untuk embedding
dan penyimpanan, sedangkan generasi jawaban tetap lokal.

### Migrasi indeks lama

Tidak perlu mengunduh embedding BGE untuk Cloud. Skrip berikut membaca evidence dari
snapshot lokal, melakukan chunking, dan menghasilkan embedding Qwen + SPLADE baru.
Vektor NumPy lama tidak disalin karena ruang embedding berbeda. Snapshot lama dan
file asli tetap utuh. Corpus harus masih cocok dengan fingerprint snapshot.

```bash
# .env Cloud: gunakan RAG_STATE_DIR=state/chroma agar state lama tetap terpisah.
./.venv/bin/python scripts/migrate_chroma.py --snapshot state/index.json --dry-run
./.venv/bin/python scripts/migrate_chroma.py --snapshot state/index.json \
  --output state/chroma/index.json
```

Output migration tidak boleh menimpa file yang sudah ada. Restart worker dengan
konfigurasi `.env` yang sama (termasuk `RAG_STATE_DIR=state/chroma`) untuk restore.
Restore memeriksa fingerprint, model/config dan kelengkapan record Cloud. Jika belum
ada snapshot lama, cukup jalankan `ask`; indexing dan upload dilakukan otomatis.

Untuk mode offline, pilih `RAG_SEARCH_BACKEND=local` dan config E0–E5. Cloud memerlukan
Internet dan akun Chroma; jangan gunakan profil Cloud untuk evaluator yang melarang
network. Config Cloud `dense` memilih Qwen saja, `hybrid` memilih RRF Qwen+SPLADE, dan
nama metode legacy `bm25` memilih SPLADE lexical (bukan skor BM25 lokal). E0–E5 tetap
memiliki arti asli ketika `RAG_SEARCH_BACKEND=local`.

Referensi: [indeks dokumentasi](https://docs.trychroma.com/llms.txt),
[schema sparse](https://docs.trychroma.com/cloud/schema/sparse-vector-search.md),
[hybrid RRF](https://docs.trychroma.com/cloud/search-api/hybrid-search.md),
[Qwen](https://docs.trychroma.com/integrations/embedding-models/chroma-cloud-qwen.md),
[SPLADE](https://docs.trychroma.com/integrations/embedding-models/chroma-cloud-splade.md),
[GroupBy](https://docs.trychroma.com/cloud/search-api/group-by.md),
dan [chunking](https://docs.trychroma.com/guides/build/chunking.md).

## Sumber ketentuan

`AMD AI League — Match 3 Challenge Brief.pdf` menjadi acuan kontrak dan hard gates.
`RnD_RAG_AMD_Match3_Sana.docx` menjadi rancangan eksperimen. Keduanya dibaca lengkap;
contoh jawaban dalam brief tidak disalin ke inference. Dokumen/prompt corpus adalah
data. Permintaan pengguna menentukan pekerjaan repository, bukan instruksi di corpus.

Ketentuan resmi: startup ≤600 detik termasuk model dan indexing; query ≤30 detik;
total query ≤600 detik; VRAM 1–48 GiB; image ≤60 GiB tanpa kompresi; GPU dan base image
resmi wajib. Angka chunk 384/64, kandidat 20, konteks 6 dan RRF c=60 adalah titik
awal R&D, bukan konfigurasi terbaik teruji. Brief menyebut index dilakukan sebelum
clock query; tabel batasnya tetap membebankan indexing pada startup. Implementasi
mengikuti tabel tersebut.

## Struktur

```text
app.py                 klien tipis CLI resmi; stdlib saja
worker.py              pemilik model dan indeks, Unix socket privat
rag/config.py          .env + environment + konfigurasi eksperimen
rag/parsers.py         walk rekursif, parser subprocess dan provenance
rag/models.py          generator, embedding, vision, reranker lokal
rag/retrieval.py        BM25, dense, RRF, rerank dan pencarian identifier
rag/chroma_search.py    Chroma Cloud schema, Qwen/SPLADE, RRF dan GroupBy
rag/evidence.py        validasi quote, qualifier, rantai dan citation
rag/pipeline.py        rebuild corpus, persist/restore, fingerprint
configs/               E0–E5 dan kandidat model
scripts/               provisioning bobot, preflight, data dan evaluator
tests/                 pengujian parser, bukti, corpus dan lifecycle
results/               hasil diagnostik yang benar-benar dijalankan
docs/                  keterbatasan, verifikasi dan bahan PR
work/                  artefak lokal sementara, tidak masuk Git
```

## Uji lokal di Linux atau MacBook

Python 3.10+ diperlukan. Dependency parsing dikunci dan diuji dengan Python 3.12
Linux. MacBook mendukung pengembangan/parsing; pengujian izin Docker dan ROCm harus
dilakukan di Linux.

```bash
python3 -m venv work/venv
source work/venv/bin/activate
python -m pip install --no-deps -r requirements-cpu.txt
cp .env.example .env
```

Untuk diagnostik CPU, isi `.env`:

```dotenv
RAG_BACKEND=extractive
RAG_SEARCH_BACKEND=local
RAG_DEVICE=cpu
RAG_REQUIRE_ROCM=false
RAG_CONFIG=configs/E0.json
```

```bash
python -m unittest discover -s tests -v
python scripts/make_synthetic.py --output work/synthetic
python app.py --index work/synthetic/dev/corpus
python app.py --corpus work/synthetic/dev/corpus --query-id local_01 \
  --query 'What is the unit price at 20000 units?'
cat output/local_01_output.json
```

Backend `extractive` hanya mengambil satu pasangan key/value literal dengan konteks
query yang cocok; ia tidak membaca gambar, tidak mendukung dense/rerank, dan tidak
membuktikan kualitas generator. Untuk pipeline model gunakan backend `transformers`.
Worker dibootstrap oleh `--index` pada penggunaan lokal; output/log di `output/` dan
`state/`. Hentikan worker dengan SIGTERM ke PID dari `ping`/log atau jalankan
`python worker.py` di terminal terpisah lalu Ctrl-C. Restart worker setiap mengubah
backend/model/konfigurasi. Klien menolak konfigurasi yang berbeda dari worker.

## Bertanya dengan satu perintah

Setelah dependency dan `.env` disiapkan, gunakan `ask.py` untuk satu file atau
folder dokumen. Path relatif dihitung dari lokasi terminal saat ini. Jika pertanyaan
tidak diberikan, program memintanya lewat terminal:

```bash
python ask.py '/path/dokumen saya.pdf'
python ask.py '/path/folder-dokumen' 'Apa nomor komponen kipas produk PX-713?'
```

Untuk menggunakan nama perintah `ask`, aktifkan virtualenv dan tambahkan root repo
ke PATH sekali per terminal (Linux/macOS):

```bash
source work/venv/bin/activate
export PATH="$PWD:$PATH"  # jalankan dari root repository
ask '/path/dokumen saya.pdf'
ask '/path/folder-dokumen' 'Pertanyaan tentang dokumen'
```

`ask` otomatis menjalankan indexing lalu query, menampilkan jawaban, sumber asli,
dan lokasi JSON. File tunggal disalin ke `state/ask-inputs/` (atau `RAG_STATE_DIR`),
karena pipeline menerima folder; file asli tidak diubah. Indeks yang masih valid
dipakai ulang; dokumen yang berubah diindeks kembali. Gunakan `--reindex` untuk
memaksa rebuild. Worker/model yang
konfigurasinya sama tetap digunakan. Jalankan permintaan secara berurutan karena
worker memiliki satu indeks aktif.

Untuk uji CPU teks sederhana, gunakan `python ask.py --diagnostic '/path/file.txt'`
atau `ask --diagnostic '/path/file.txt'`. Ini memakai extractive/BM25, bukan generator
AI, dan tidak membaca gambar. Mode normal memakai konfigurasi `.env` dan memerlukan
bobot lokal untuk backend Transformers. Restart worker sebelum berpindah mode;
klien akan menolak worker dengan konfigurasi berbeda. Salinan dokumen, indeks, dan
hasil disimpan lokal dalam direktori yang diabaikan Git; jangan memakai root repo
sebagai corpus karena state/output/model harus berada di luar corpus.

## Konfigurasi yang benar-benar dibaca

Worker dan CLI membaca `.env` di root repository (atau `RAG_ENV_FILE`). Variabel yang
sudah diekspor mengambil prioritas, lalu `.env`, lalu default. Nilai relatif dihitung
dari root repository. Reader menerima `KEY=value` dan quote, tanpa ekspansi shell,
interpolasi, atau eksekusi. Jangan memakai `export KEY=...` di file `.env`.

| Variabel | Kewajiban / default |
|---|---|
| `RAG_BACKEND` | Default `transformers`; `extractive` hanya diagnostik |
| `RAG_SEARCH_BACKEND` | `chroma` untuk Cloud; default kode `local` untuk mempertahankan alur offline |
| `CHROMA_API_KEY` | Wajib untuk pencarian/embedding Cloud; tidak disimpan dalam snapshot/config signature |
| `CHROMA_HOST`, `CHROMA_TENANT`, `CHROMA_DATABASE` | Host Cloud dan identitas akun/database; lihat contoh di atas |
| `CHROMA_SCOPE`, `RAG_CORPUS_ID` | Scope wajib untuk Cloud; identitas corpus opsional |
| `CHROMA_CHUNK_BYTES`, `CHROMA_BATCH_SIZE` | Default 12288 byte dan 64 record per batch |
| `CHROMA_CHUNKS_PER_DOCUMENT` | Default 3; GroupBy membatasi chunk per source document |
| `RAG_CONFIG` | Opsional, default `configs/E0.json` |
| `RAG_DEVICE`, `RAG_REQUIRE_ROCM` | Default `cuda`, `true`; wajib sesuai GPU untuk submission |
| `RAG_MODEL_DIR` | Default `models`; snapshot lokal wajib untuk model nyata |
| `RAG_GENERATOR_PATH`, `RAG_VISION_PATH` | Opsional override; snapshot wajib, default `models/generator`, `models/vision` |
| `RAG_EMBEDDING_PATH` | Snapshot wajib pada dense/hybrid **lokal**; tidak dipakai pada Cloud |
| `RAG_RERANKER_PATH` | Snapshot wajib bila rerank; default `models/reranker` |
| `RAG_VERIFY_WEIGHTS` | Default `true`; SHA file dan revision diperiksa saat model load |
| `RAG_SOCKET` | Opsional, default `/tmp/match3-rag.sock`; unik per worker |
| `RAG_STATE_DIR`, `RAG_OUTPUT_DIR` | Opsional, default `state`, `output` |
| `RAG_QUERY_TIMEOUT`, `RAG_STARTUP_TIMEOUT` | Default 26, 570 detik; harus di bawah gate resmi |
| `RAG_PARSE_TIMEOUT`, `RAG_MAX_FILE_MB` | Opsional, default 15 detik/file dan 100 MiB/file |
| `RAG_MAX_NEW_TOKENS` | Opsional, default 512; ukur pengaruh ke JSON dan latensi |
| `HF_TOKEN` | Opsional, hanya provisioning model bila akses memerlukannya |

Generasi jawaban lokal tidak membutuhkan API key; search Cloud membutuhkan key Chroma.
`BASE_IMAGE` adalah argumen build
wajib, **bukan** environment runtime; harus berasal dari aturan resmi. Semua direktori
state/output/model dan label evaluasi harus di luar corpus. Tidak ada aturan berdasarkan
nama produk, subfolder, file, query-id, atau expected answer.

## Integrasi dan penyediaan model

Kandidat pengembangan: generator `Qwen/Qwen2.5-3B-Instruct`, embedding
`BAAI/bge-small-en-v1.5`, vision `Qwen/Qwen2.5-VL-3B-Instruct`, dan reranker opsional
`BAAI/bge-reranker-base`. Ini pilihan sementara, belum disahkan/diukur pada image
resmi. Loader memakai `AutoModelForCausalLM`, `AutoModel` dengan CLS BGE ternormalisasi,
`Qwen2_5_VLForConditionalGeneration`, dan `AutoModelForSequenceClassification`.
Vision dijalankan ketika indexing PNG/JPG serta halaman PDF scan/berisi gambar.
Model bertahan dalam worker, termasuk setelah indexing dan query.

Setelah aturan penyediaan bobot resmi diperiksa, lakukan provisioning **sebelum**
evaluasi di mesin yang boleh mengakses model hub. Periksa model card/lisensi untuk
setiap kandidat; pengunduh menyertakannya bila tersedia. Tidak ada token disimpan
dalam manifest. Jangan menganggap unduhan atau internet saat evaluasi diizinkan.

```bash
# Dalam base image resmi / environment pengembangan yang cocok, pertahankan torch.
python3 -m pip install --no-deps -r requirements.txt
python3 scripts/download_models.py --roles generator embedding vision reranker
python3 scripts/download_models.py --verify-only --roles generator embedding vision reranker
```

`configs/models.candidates.json` awal memakai ref `main`. Skrip menyelesaikan ref itu
ke commit SHA 40 karakter, mengunduh **SHA tersebut**, lalu menulis
`models/models.lock.json` dan `weights-manifest.json` per role dengan hash/ukuran tiap
file. Pengunduhan ulang yang reproducible:

```bash
python3 scripts/download_models.py --lock /path/to/archived/models.lock.json \
  --require-pinned --roles generator embedding vision reranker
```

Gunakan direktori target kosong untuk pengunduhan ulang. Arsipkan lock dan seluruh
snapshot di penyimpanan terpisah dari Git. `models/` tidak masuk Git, tetapi **sengaja
masuk build context** untuk dibundel ke image. Loader hanya menerima file lokal,
safetensors, dan `trust_remote_code=false`. Container mematikan akses hub.
Bobot belum diunduh dalam sesi ini: aturan resmi belum ada, dan host Hugging Face
tidak tercantum dalam policy jaringan workspace. Tidak ada klaim model forward pass,
akurasi vision, atau kompatibilitas AMD dari import kelas saja.

## Corpus dan kontrak CLI

```bash
python3 /app/app.py --index /app/corpus
python3 /app/app.py --corpus /app/corpus --query-id query_01 --query '...'
```

Output atomik: `/app/output/<query-id>_output.json`, selalu berisi string `answer`
dan list string `citations`. Tidak ada bukti: `{"answer":"","citations":[]}`.
Path citation relatif corpus; model hanya memilih ID/kutipan, tidak menulis path.
File penghubung identifier tetap dicantumkan pada rantai lintas file. Teks jawaban
harus muncul dalam quote; qualifier revisi/fiscal year dan identifier link yang
terpotong ditolak. Pemilihan hubungan semantik dan penyelesaian revisi tetap tugas
model dan perlu diuji pada corpus nyata.

Setiap `--index` membangun indeks **baru** dari corpus yang diberikan. Indeks lama
dinonaktifkan sebelum rebuild, termasuk bila rebuild gagal. Query menolak path lain
atau perubahan inventory/mtime/size/mode corpus. Tidak ada indexing/model load pada
query. Restart worker boleh memuat snapshot yang identitas corpus/config/model-nya
masih valid; invocation `--index` berikutnya tetap membangunnya ulang.

Parser membaca seluruh subdirektori, paragraf/tabel DOCX, semua sheet XLSX,
CSV dengan header, PDF per halaman/tabel, teks/log/source Python tanpa mengeksekusi
kode, dan vision gambar. File kosong/asing/symlink dilewati; error permission,
enkripsi, korupsi dan timeout dicatat per file. Corpus symlink tidak diikuti untuk
mencegah bukti keluar dari root. File >100 MiB default dilewati dan dicatat.
Provenance menyimpan `chunk_id`, path, content, kind, page/sheet/row/block/line/region.
Status parsial menunjukkan gambar tidak terbaca atau vision diagnostik nonaktif.

Error worker/timeout menghasilkan output kosong dengan exit code 1 dan trace error,
sehingga evaluator lokal tidak menghitungnya sebagai penolakan benar. Query-id hanya
dipakai untuk nama output dan divalidasi terhadap traversal, tidak masuk inference.

## Eksperimen dan evaluasi

| Konfigurasi | Perubahan |
|---|---|
| E0 | BM25 |
| E1 | Dense |
| E2 | BM25 + dense + RRF |
| E3 | E2 + reranker maksimum 20 kandidat |
| E4 | E3 + satu putaran identifier, maksimum 4 identifier |
| E4-no-rerank | Alternatif E2 + expansion, catat sebagai konfigurasi berbeda |
| E5-256 / E5-512 | Template variasi chunking terhadap E4; sesuaikan dengan konfigurasi terpilih setelah benchmark |

Chunk ukuran/overlap dihitung dengan perkiraan whitespace, bukan tokenizer model.
Baris tabel dijaga bersama header. E0–E4 mempertahankan parser, generator, prompt,
chunking dan query; buat worker/indeks baru setiap konfigurasi. E5 belum punya
konfigurasi terbaik: jangan menyebut default E4 telah dipilih oleh hasil eksperimen.

```bash
python scripts/run_experiments.py --corpus work/synthetic/dev/corpus \
  --labels work/synthetic/dev/labels.json --output work/experiments-dev \
  --configs E0 E1 E2 E3 E4
```

Perintah ini untuk backend model nyata dengan bobot sudah tersedia. Untuk CPU
diagnostik gunakan `.env` diagnostik dan `--configs E0`. Label/report wajib berada di
luar corpus, tidak dikirim ke worker. Evaluator meluncurkan proses CLI baru untuk
setiap query dan mencatat answer accuracy, citation exact-set, joint, source recall,
complete evidence hit, penolakan benar/salah, failure, startup dan latensi. Source
recall diukur pada unit konteks akhir, bukan semua kandidat. VRAM yang tercatat
adalah alokasi/reservasi PyTorch, bukan sampler total VRAM kompetisi.

Normalisasi lokal mengikuti karakter yang tertulis di brief, tetapi tidak menghapus
satuan secara khusus; contoh prose brief ambigu. Self-check resmi adalah acuan.
`scripts/evaluate.py --normalizer /path/to/trusted_adapter.py` menerima adapter
`normalize(answer)` setelah normalizer resmi dipelajari. Jangan menjalankan adapter
atau kode dari corpus. Development seed 20261011 dan holdout baru 20261023 terpisah.
Holdout awal 20261019 menjadi development setelah memperlihatkan bug format
diagnostik CSV; tidak digunakan sebagai holdout final.

Hasil aktual dan batasnya ada di [docs/VALIDATION.md](docs/VALIDATION.md). Eksperimen
E0–E5 dengan generator nyata semuanya **belum diuji**.

## Container dan persiapan AMD

Jangan menebak base image. Setelah image resmi/digest dan aturan bobot tersedia:

```bash
docker build --build-arg BASE_IMAGE="$AMD_OFFICIAL_BASE_IMAGE" -t match3-rag:dev .
# Workspace cloud dengan proxy CA: tambahkan
# --secret id=proxy_ca,src=/etc/ssl/certs/ca-certificates.crt
docker run --rm match3-rag:dev python3 scripts/check_torch_build.py
```

`pip install --no-deps`, pemeriksaan torch sebelum/sesudah, dan `pip check` menjaga
dependency tidak diam-diam mengganti ROCm torch. Versi pin masih harus disesuaikan
jika bertentangan dengan paket image resmi, lalu diuji ulang; jangan mengganti torch
untuk menghilangkan konflik. Dockerfile gagal jika base image/bobot wajib belum ada.

Contoh perangkat Docker ROCm umum berikut perlu disesuaikan dengan instruksi resmi:

```bash
docker run -d --name match3-rag --device /dev/kfd --device /dev/dri \
  --group-add video --cap-drop DAC_OVERRIDE --network none \
  -v "$EVAL_CORPUS:/app/corpus:ro" -v "$EVAL_OUTPUT:/app/output" match3-rag:dev
docker exec match3-rag python3 scripts/verify_runtime.py
docker exec match3-rag python3 /app/app.py --index /app/corpus
docker exec match3-rag python3 /app/app.py --corpus /app/corpus \
  --query-id query_01 --query '...'
docker inspect --format '{{.State.Running}}' match3-rag
docker image inspect --format '{{.Size}}' match3-rag:dev
```

Di GPU ukur startup dari container start, semua query, dan VRAM total yang disampling
terus-menerus memakai tooling panitia/ROCm, bukan hanya `peak_vram_bytes`. Jangan
menggunakan angka CPU dalam results sebagai bukti melewati gate AMD. Siapkan folder
mount dengan izin yang bisa dibaca user container ketika DAC_OVERRIDE dilepas.

Unduh starter kit dari halaman resmi ketika tersedia, baca instruksinya, jalankan
`setup.sh` di direktori kit, lalu self-check kit sesuai petunjuknya. Nama script
self-check/image resmi tidak tersedia, sehingga tidak dibuat-buat di sini. Gunakan
label mesin kit di evaluator tanpa memasukkannya ke corpus. Periksa aturan umum
Section 2, benchmark model dan eksperimen pada AMD, bekukan lock/config, kemudian
jalankan self-check dari build bersih. Proyek ini tidak melakukan deployment,
submission, merge otomatis, atau perubahan visibilitas repository.

## Mengambil perubahan ke MacBook

Branch implementasi: `feat/dynamic-rag-match3` di `sanacvn/RAG-Match3-AMD`. Ambil branch:

```bash
git clone https://github.com/sanacvn/RAG-Match3-AMD.git
cd RAG-Match3-AMD
git fetch origin
git switch --track origin/feat/dynamic-rag-match3
```

Untuk clone yang sudah ada gunakan tiga perintah terakhir tanpa `git clone`.
Judul/deskripsi PR ada di `docs/PR.md`. Jika akses push/PR tidak tersedia, paket
`git format-patch`/bundle yang disediakan bisa diaplikasikan di clone MacBook, lalu
`git push -u origin feat/dynamic-rag-match3` dan buka draft PR ke `main`.
`.env`, token, bobot, cache, indeks, output dan fixture generated tidak masuk Git.
