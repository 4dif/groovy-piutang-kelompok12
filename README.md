# Prediksi Risiko Piutang Transaksi - PT. Groovy Layan Satwa
Kelompok 12 - Pembelajaran Mesin (S1), Universitas Nusa Mandiri - Semester Gasal 2026/2027

| Berkas | Isi |
|---|---|
| `Kelompok12_Groovy_Piutang_T02-T06.ipynb` | Notebook Google Colab end-to-end (Tugas 02-06): pembersihan data, baseline, GridSearchCV, Permutation Importance, SHAP, serialisasi, Optuna, threshold |
| `groovy_common.py` | Pembersihan data & preprocessor (Winsorizer, encoder, scaler) |
| `optuna_tugas06_groovy.py` | Optuna TPE 40 trial x 2 model + Stratified 5-Fold CV + threshold dari out-of-fold |
| `tpe_lite.py` | Cadangan TPE bila paket optuna tidak tersedia |
| `model_final_tugas06.joblib` | `{'pipeline': HistGradientBoosting pasca-optimasi, 'threshold': 0.34}` |
| `webapp_groovy_piutang_check/` | Prototipe web Flask (Tugas 05): form, prediksi massal CSV, REST API, review supervisor |

## Menjalankan
1. Buka notebook di Google Colab, unggah `detil_penjualan_2026_gabungan.csv`, lalu *Runtime > Run all*.
2. Atau lokal: `pip install optuna scikit-learn==1.9.1 pandas numpy joblib` lalu `python optuna_tugas06_groovy.py detil_penjualan_2026_gabungan.csv`.

**Data:** file transaksi PT. Groovy Layan Satwa tidak disertakan di repositori publik (data internal perusahaan). Minta file ke tim.

## Hasil utama (Test Set, n = 1.228)
- HistGradientBoosting + Optuna: F1-Macro 0,7319 | ROC-AUC 0,8458 | PR-AUC 0,7111 (threshold 0,50)
- Threshold 0,34 (biaya FN:FP = 3:1, dipilih dari out-of-fold): Recall Piutang 0,9248, FN 31 kasus
