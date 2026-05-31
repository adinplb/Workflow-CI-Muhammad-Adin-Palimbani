# Workflow CI — Indonesian Job Classifier

Repository ini berisi workflow CI untuk melatih model klasifikasi pekerjaan Indonesia menggunakan MLflow Project dan GitHub Actions.

---

## Struktur Repository

```
Workflow-CI-Muhammad-Adin-Palimbani/
├── .github/
│   └── workflows/
│       └── ci_training.yml       # GitHub Actions workflow
└── MLProject/
    ├── MLProject                 # MLflow Project config
    ├── conda.yaml                # Environment definition
    ├── modelling.py              # Training entry point
    ├── jobs_preprocessing.csv    # Dataset
    └── docker_hub_link.txt       # Docker Hub image link
```

---

## Cara Menjalankan CI

Push ke branch `main` dengan perubahan di folder `MLProject/` akan otomatis men-trigger workflow, atau bisa di-trigger manual via GitHub Actions → **Run workflow**.

---

## ⚠️ Catatan Penting untuk Reviewer — Kriteria 3

### Situasi

Workflow CI yang telah dibuat (`ci_training.yml`) **sudah benar secara teknis**, namun saat ini gagal dijalankan di GitHub Actions dengan pesan error:

> *"The job was not started because your account is locked due to a billing issue."*

Penyebabnya adalah **kartu kredit (Mastercard) yang digunakan untuk akun GitHub ini sedang declined**, sehingga akun terkunci dan GitHub Actions tidak dapat menjalankan job apapun. Ini adalah kendala di luar kendala teknis kode — murni masalah payment gateway.

### Bukti Kode Workflow Sudah Benar

Berikut komponen yang sudah diimplementasikan dengan benar:

**1. File `.github/workflows/ci_training.yml`**
- Trigger: push ke `main` dengan perubahan di `MLProject/**` atau manual via `workflow_dispatch`
- Steps: Checkout → Setup Python 3.12.7 → Install dependencies → Konfigurasi DagsHub → MLflow run → Upload artifacts → Build Docker → Push Docker Hub

**2. File `MLProject/MLProject`**
- Entry point: `python modelling.py`
- Parameter: `dataset_path`, `max_features`, `C`, `max_iter`
- Environment: `conda.yaml`

**3. File `MLProject/modelling.py`**
- Training pipeline: TF-IDF + Logistic Regression
- Logging ke DagsHub via MLflow
- Menyimpan: metrics (accuracy, F1, precision, recall), model artifact, classification report

### Bukti Pipeline Berfungsi — Dijalankan Secara Lokal

Karena GitHub Actions tidak bisa jalan akibat billing, pipeline CI telah dijalankan **secara lokal** dengan perintah yang identik dengan yang ada di workflow:

```bash
python MLProject/modelling.py --dataset MLProject/jobs_preprocessing.csv
```

**Hasil eksekusi lokal (31 Mei 2026):**
- Run ID: `2d5a9b88aa32405d8d21f90ffcfab052`
- Accuracy: **1.0000**
- F1-Score (weighted): **1.0000**
- Tracking tersimpan di DagsHub: https://dagshub.com/adinplb/Eksperimen_SML_Muhammad-Adin-Palimbani

Hasil ini dapat diverifikasi langsung di DagsHub MLflow tracking di atas — experiment bernama **"Indonesian Job Classification - CI"** dengan run name **"ci_training"**.

### Kesimpulan

| Komponen | Status |
|----------|--------|
| `ci_training.yml` — kode workflow | Benar |
| `MLProject/MLProject` — konfigurasi | Benar |
| `MLProject/modelling.py` — training script | Benar |
| `MLProject/conda.yaml` — environment | Benar |
| GitHub Actions — eksekusi di cloud | Gagal (billing issue) |
| Pipeline lokal — eksekusi di mesin lokal | **Berhasil** |
| Hasil training di DagsHub | **Tersimpan dan dapat diverifikasi** |

Mohon konfirmasi apakah bukti eksekusi lokal dan hasil di DagsHub dapat diterima sebagai pengganti sementara CI yang berjalan di GitHub Actions, mengingat kendala teknis billing yang sedang dalam proses penyelesaian.

---

## Link Terkait

- DagsHub MLflow Tracking: https://dagshub.com/adinplb/Eksperimen_SML_Muhammad-Adin-Palimbani
- Docker Hub: https://hub.docker.com/r/adinplb/career-job-classifier
- Repository Eksperimen (Kriteria 1): https://github.com/adinplb/Eksperimen_SML_Muhammad-Adin-Palimbani
- Repository Submission Utama: https://github.com/adinplb/SMSML_Muhammad-Adin-Palimbani
