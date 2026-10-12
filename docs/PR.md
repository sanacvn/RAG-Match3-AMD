# Implement dynamic-corpus offline RAG with persistent worker and evaluation tooling

Corpus evaluasi baru kini dibangun saat `--index`; query memakai indeks dan model
di worker persisten, dengan pengecekan path/fingerprint/config agar bukti corpus
lain atau indeks usang tidak dipakai. CLI menulis JSON atomik sesuai query-id
evaluator dan memberi output kosong saat bukti tidak cukup atau terjadi error.

Implementasi mencakup parser seluruh format brief dan provenance, isolated parser
timeouts, BM25/dense/RRF/rerank serta pencarian identifier satu putaran, pemilihan
evidence ID/quote/link dengan citation yang dipetakan oleh kode, integrasi generator
Qwen dan vision Qwen2.5-VL lokal serta embedding/reranker BGE. `.env.example` benar-benar
dibaca; provisioning offline mencatat immutable model revision dan hash bobot.
Dockerfile menerima base image resmi sebagai argumen wajib dan memakai pip
`--no-deps` serta pemeriksaan ROCm torch. Bobot, credential dan cache tidak masuk
Git. Label hanya dipakai evaluator/hasil diagnostik, tidak dikirim ke inference.

Validation: 17 tes unit/integrasi Linux lulus pada dependency pin dalam environment
terpisah. Tes root Docker `--cap-drop DAC_OVERRIDE` lulus. Diagnostik CPU dengan
20 development + 20 holdout mendapatkan joint 55%, source completeness 89.47%,
dan nol failure operasional. Hasil ini hanya backend key/value diagnostik dan
bukan benchmark model/GPU. Hasil per-query dan keterbatasan ada di `results/` dan
`docs/VALIDATION.md`.

Draft: exact official base image, All About the Challenges, starter kit, aturan
bobot/network dan GPU AMD belum tersedia. Bobot/forward pass serta eksperimen
E0–E5 nyata, kompatibilitas ROCm, hard gates runtime/VRAM/image dan self-check resmi
belum terverifikasi. Bukti semantik/revisi/vision masih perlu benchmark; validator
konservatif dapat menolak pertanyaan kompleks. Jangan merge otomatis, deploy, atau
kirim submission dari PR ini.
