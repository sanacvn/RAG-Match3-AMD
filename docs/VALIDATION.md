# Verifikasi 11 Oktober 2026 (Asia/Jakarta)

## Hasil yang dijalankan

Pengujian dilakukan di Linux, Python 3.12, user non-root. Semua dependency di
`requirements.txt` diinstal dalam virtual environment terpisah dengan `--no-deps`;
PyTorch tidak ada di host dan tidak dipasang/diganti. Versi parsing yang diuji:
NumPy 2.2.6, PyMuPDF 1.25.5, python-docx 1.1.2, openpyxl 3.1.5, Pillow 11.2.1.

**17 tes unit/integrasi lulus** dalam environment bersih; `pip check` menyatakan
tidak ada dependency yang rusak. Suite menguji:

- Rebuild corpus dengan produk/nama/path/nilai berbeda; corpus lain, perubahan
  file, rebuild gagal, dan snapshot usang tidak memakai bukti lama.
- Corpus kosong, unknown type, permission denied, PDF terenkripsi, DOCX rusak,
  timeout per file, seluruh sheet, tabel DOCX/CSV, PDF, dan source tanpa eksekusi.
- Gambar dan scan PDF diteruskan ke interface vision memakai stub dalam tes parser.
  Ini membuktikan routing/provenance, **bukan** pembacaan teks oleh model vision.
- RRF, dense/hybrid/rerank interface dengan vector/score sintetis; bukan akurasi
  embedding atau reranker nyata.
- Evidence ID asing, quote palsu, extra source, rantai tanpa identifier, prefix
  identifier/nilai yang terpotong, qualifier revisi dan fiscal year.
- Proses query terpisah dengan PID/generasi worker tetap, output atomik, error
  worker menjadi JSON kosong + exit nonzero, query-id traversal ditolak.
- `--index` menunggu worker container saat model masih loading; konfigurasi klien
  yang berubah ditolak; budget indexing pertama termasuk waktu model load.
- `.env` dibaca dengan prioritas environment, tanpa mengeksekusi shell; manifest
  bobot mendeteksi file yang diubah.

Tes permission Docker dijalankan sebagai root, `--cap-drop DAC_OVERRIDE`,
`--network none`, image diagnostik `python:3.12-slim` (digest
`sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1`).
File chmod 000 menghasilkan status failed; file normal tetap parsed.
Image ini hanya alat tes izin, **bukan** base image submission.

## Diagnostik sintetis

`scripts/make_synthetic.py` menghasilkan 20 soal development dan 20 holdout baru,
dengan corpus/label terpisah. Sepuluh soal key/value sederhana, tabel, CSV, kode,
revisi lama/baru, dua gambar, rantai lintas file, dan satu unanswerable. Ground truth
diinspeksi dari konstruksi fixture; belum ada reviewer manusia independen.

Backend `extractive` + BM25 digunakan untuk tes plumbing CPU. Ia sengaja terbatas
pada key/value literal tunggal. Jawaban gambar/rantai/revisi membutuhkan generator
dan vision nyata; diagnostik menolaknya. Hasil lengkap ada di `results/`.

Joint accuracy development dan holdout: **11/20 (55%)**, dengan penolakan benar pada
satu soal unanswerable di setiap split. Tidak ada failure operasional. Source recall
dan complete evidence hit pada konteks akhir: **89.47%**; dua gambar tidak punya
evidence karena vision diagnostik nonaktif. False rejection: **9/19 (47.37%)** pada
soal answerable. Angka waktu aktual, startup, masing-masing query dan total query
disimpan dalam JSON hasil, bukan diekstrapolasi ke GPU/corpus kompetisi.

Holdout awal seed 20261019 memperlihatkan bug backend diagnostik yang mengambil
lebih dari satu field CSV sebagai nilai; split itu dipromosikan ke development.
Bug diperbaiki secara umum dengan menolak value gabungan. Holdout final seed
20261023 dibuat setelah perbaikan dan tidak dipakai untuk tuning.

Normalisasi ini pendekatan lokal, belum normalizer/self-check resmi. Confidence
tidak dipakai sebagai probabilitas atau kriteria penolakan tunggal.

## Status eksperimen model nyata

| Eksperimen / gate | Status |
|---|---|
| E0 BM25 + generator nyata | Belum diuji |
| E1 dense + generator | Belum diuji |
| E2 hybrid RRF | Belum diuji |
| E3 reranking | Belum diuji |
| E4 bounded expansion | Belum diuji |
| E5 chunking terbaik | Belum dipilih / diuji |
| Download bobot dan forward pass nyata | Belum dilakukan |
| ROCm torch pada base image resmi | Belum diuji |
| GPU benar-benar digunakan, VRAM total 1–48 GiB | Belum diuji |
| Startup ≤600s termasuk model dan corpus graded | Belum diuji |
| Setiap query ≤30s dan total ≤600s pada AMD | Belum diuji |
| Image resmi ≤60 GiB tanpa kompresi | Belum dibangun / diukur |
| Self-check starter kit | Belum tersedia / dijalankan |

Transformers 4.51.3, tokenizers 0.21.1, safetensors 0.5.3, dan Hugging Face Hub 0.30.2
berhasil diinstal; nama kelas Qwen vision tersedia. Tanpa torch/bobot, hal tersebut
tidak membuktikan pemuatan/inference model, AMD atau akurasi.

## Batas implementasi yang perlu diukur

- Pemilihan semantik sumber dan resolusi konflik revisi bergantung pada model.
  Validator membuktikan quote/ID/nilai/link dan menolak graph yang tidak lengkap,
  tetapi tidak bisa membuktikan semua fakta semantik atau menangkal seluruh
  prompt injection hanya lewat kecocokan teks. Tidak ada tool/exec corpus.
- Validator konservatif meminta nilai literal dalam quote serta satu rantai
  terhubung. Jawaban yang memerlukan perhitungan, beberapa rantai independen,
  atau resolusi revisi lintas file tanpa identifier eksplisit dapat ditolak.
- XLSX menggunakan cached values (`data_only`); formula tanpa cache tidak
  dihitung. Tabel dengan header bertingkat/merge yang rumit perlu benchmark.
- Chunk memakai estimasi whitespace, bukan jumlah token tokenizer generator.
  Sel tabel sangat panjang tidak dipecah; generator menolak context overflow.
- Query memeriksa fingerprint inventory/stat untuk kesegaran tanpa reparse.
  Modifikasi adversarial yang sengaja mempertahankan ukuran/mtime/mode tidak
  terdeteksi. SHA konten disimpan saat index untuk provenance.
- Timeout subprocess membatasi parser. `max_time` generation diperiksa antar
  langkah decoding; kernel GPU panjang dapat melampaui deadline internal.
  Klien tetap menulis refusal sebelum timeout lokal, tetapi worker dapat masih
  sibuk. Ukur antrean/latensi pada AMD dan jangan menganggap ini lulus hard gate.
- Ada batas ukuran file dan pixel vision agar waktu dapat dibatasi. Setiap
  skipped/partial/failed file terlihat dalam manifest; bukti yang hilang dapat
  mengurangi skor. Corpus graded lebih besar dari fixture ini.
- Alokasi/reservasi PyTorch bukan total VRAM sampled evaluator. Preflight GPU,
  pengukuran image dan sampler resmi masih diperlukan.

## Langkah final AMD

1. Peroleh aturan umum Section 2, exact base image/digest, starter kit, GPU dan aturan
   provisioning bobot/network. Tidak ada tautan download aktif di PDF unggahan.
2. Periksa lisensi dan kebijakan model; provision, arsipkan immutable lock/snapshot,
   lalu build image tanpa mengganti torch ROCm.
3. Jalankan preflight ROCm, setup/self-check starter kit, permission/encryption stress
   test, dan E0 sebelum eksperimen lain. Jangan menyalin label ke corpus.
4. Bandingkan E1–E4 pada development dengan komponen lain tetap; E5 hanya setelah
   konfigurasi terpilih. Bekukan lalu jalankan holdout terpisah.
5. Ukur keseluruhan startup/query, sampler VRAM, image uncompressed dan container
   yang tetap hidup; ulangi self-check dari build bersih. Submission tidak dikirim
   otomatis oleh proyek ini.
