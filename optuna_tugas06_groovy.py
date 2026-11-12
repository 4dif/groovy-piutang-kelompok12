# ====================================================================
# TUGAS 06 (KELOMPOK 12) - PT. Groovy Layan Satwa
# Optuna TPE 40 trial x 2 keluarga model + Stratified 5-Fold CV + Decision Threshold Tuning
# Colab:  !pip install optuna   lalu:  python optuna_tugas06_groovy.py
# ====================================================================
import sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, average_precision_score
from groovy_common import load_clean, make_preprocessor, FITUR

PATH = sys.argv[1] if len(sys.argv) > 1 else 'detil_penjualan_2026_gabungan.csv'
RS, N_TRIALS = 42, 40
df = load_clean(PATH)
X, y = df[FITUR], df['Status_Piutang']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

def pipe(model):  # preprocessing ikut di-fit di dalam tiap fold -> bebas leakage
    return Pipeline([('prep', make_preprocessor()), ('model', model)])
def build(nama, p):
    if nama == 'RandomForest':
        return pipe(RandomForestClassifier(n_jobs=-1, random_state=RS, **{'class_weight': 'balanced', **p}))
    return pipe(HistGradientBoostingClassifier(class_weight='balanced', random_state=RS, **p))

def suggest(nama, trial):
    if nama == 'RandomForest':
        return dict(n_estimators=trial.suggest_int('n_estimators', 50, 400), max_depth=trial.suggest_int('max_depth', 4, 30),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 30), min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
                    max_features=trial.suggest_float('max_features', 0.15, 1.0),
                    class_weight=trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']))
    return dict(learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True), max_iter=trial.suggest_int('max_iter', 50, 500),
                max_depth=trial.suggest_int('max_depth', 3, 12), min_samples_leaf=trial.suggest_int('min_samples_leaf', 5, 120),
                l2_regularization=trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True), max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 8, 64))

def objective_for(nama):
    return lambda trial: cross_val_score(build(nama, suggest(nama, trial)), X_train, y_train, cv=cv, scoring='f1_macro').mean()

hasil = {}
for nama in ['RandomForest', 'HistGradientBoosting']:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RS))
        st.optimize(objective_for(nama), n_trials=N_TRIALS)
        st.trials_dataframe().to_csv(f'riwayat_optuna_{nama}.csv', index=False)
        bp, bv = st.best_params, st.best_value
    except ImportError:  # cadangan tanpa optuna: TPE ringan (tpe_lite.py), hasil identik dengan laporan
        from tpe_lite import Study, Int, Float, Cat
        space = ({'n_estimators': Int(50, 400), 'max_depth': Int(4, 30), 'min_samples_split': Int(2, 30), 'min_samples_leaf': Int(1, 20),
                  'max_features': Float(0.15, 1.0), 'class_weight': Cat(['balanced', 'balanced_subsample'])} if nama == 'RandomForest' else
                 {'learning_rate': Float(0.01, 0.3, log=True), 'max_iter': Int(50, 500), 'max_depth': Int(3, 12), 'min_samples_leaf': Int(5, 120),
                  'l2_regularization': Float(1e-4, 10.0, log=True), 'max_leaf_nodes': Int(8, 64)})
        st = Study(space, seed=RS)
        st.optimize(lambda p: cross_val_score(build(nama, p), X_train, y_train, cv=cv, scoring='f1_macro').mean(), N_TRIALS)
        _, (bp, bv) = st.best
    default_cv = cross_val_score(build(nama, {}), X_train, y_train, cv=cv, scoring='f1_macro').mean()
    hasil[nama] = (bp, bv)
    print(f'{nama}: default CV F1-Macro {default_cv:.4f} -> terbaik {bv:.4f} | {bp}')

# Model final dipilih dari skor CV (bukan test set)
nama_final = max(hasil, key=lambda k: hasil[k][1])
best_model = build(nama_final, hasil[nama_final][0]).fit(X_train, y_train)
y_probs = best_model.predict_proba(X_test)[:, 1]
print('\nModel final:', nama_final, '| ROC-AUC', round(roc_auc_score(y_test, y_probs), 4), '| PR-AUC', round(average_precision_score(y_test, y_probs), 4))
print(classification_report(y_test, (y_probs >= 0.5).astype(int), target_names=['Lunas (0)', 'Piutang (1)'], digits=4))

# Threshold dari prediksi OUT-OF-FOLD data latih: minimalkan biaya 3*FN + 1*FP
oof = cross_val_predict(build(nama_final, hasil[nama_final][0]), X_train, y_train, cv=cv, method='predict_proba')[:, 1]
grid = np.round(np.arange(0.20, 0.81, 0.01), 2)
biaya = [3 * ((oof < t) & (y_train == 1)).sum() + ((oof >= t) & (y_train == 0)).sum() for t in grid]
threshold_custom = float(grid[int(np.argmin(biaya))])
print('Threshold optimal (OOF, biaya 3:1):', threshold_custom)
for t in sorted({0.30, 0.40, 0.50, 0.60, 0.70, threshold_custom}):
    y_pred_tuned = (y_probs >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tuned).ravel()
    print(f'Threshold {t:.2f}: precision={tp/(tp+fp):.4f} recall={tp/(tp+fn):.4f} F1={f1_score(y_test, y_pred_tuned):.4f} FN={fn} FP={fp}')
joblib.dump({'pipeline': best_model, 'threshold': threshold_custom}, 'model_final_tugas06.joblib', compress=3)
