# Modul bersama Kelompok 12: pembersihan data Minggu 02 & preprocessor Minggu 03
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

def load_clean(path):
    df = pd.read_csv(path, sep=';')
    fmt = '%d-%m-%Y %H:%M:%S'
    t_order = pd.to_datetime(df['Waktu Order'], format=fmt)
    t_bayar = pd.to_datetime(df['Waktu Bayar'], format=fmt)
    metode = df['Metode Pembayaran']
    out = pd.DataFrame({
        'Layanan': (metode.str.split(',').str[0]
                    .str.replace(r'^(Bank Transfer|Kartu Debit/Kredit) - ', '', regex=True)
                    .str.replace(r'[- ](Piutang|Trans|QR|Card|Cash|Tokopedia)$', '', regex=True)
                    .str.replace('Day Care', 'Daycare').str.strip()),
        'Jenis_Order': df['Jenis Order'],
        'Total_Penjualan': df['Total Penjualan (Rp)'],
        'Jumlah_Item': df['Produk'].fillna(df['Produk'].mode()[0]).str.count(r',(?!\s?\d)') + 1,
        'Jam_Order': t_order.dt.hour,
        'Hari_Order': t_order.dt.dayofweek,
        'Bulan_Order': t_order.dt.month,
        'Durasi_Bayar_Menit': (t_bayar - t_order).dt.total_seconds() / 60,
        'Status_Piutang': metode.str.contains('Piutang').astype(int),
    })
    out.loc[metode.eq('Cash'), 'Layanan'] = 'Sales'
    return out

FITUR_KAT = ['Layanan', 'Jenis_Order']
FITUR_ORD = ['Hari_Order', 'Bulan_Order']
FITUR_ROBUST = ['Total_Penjualan', 'Jumlah_Item']
FITUR_STD = ['Jam_Order']
FITUR = FITUR_KAT + FITUR_ORD + FITUR_ROBUST + FITUR_STD
class Winsorizer(BaseEstimator, TransformerMixin):
    """Memotong nilai di atas persentil q yang dipelajari dari data latih."""
    def __init__(self, q=0.99):
        self.q = q
    def fit(self, X, y=None):
        self.cap_ = np.nanquantile(np.asarray(X, dtype=float), self.q, axis=0)
        return self
    def transform(self, X):
        return np.minimum(np.asarray(X, dtype=float), self.cap_)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)

def make_preprocessor():
    return ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='ignore'), FITUR_KAT),
        ('ord', 'passthrough', FITUR_ORD),
        ('robust', Pipeline([('wins', Winsorizer(0.99)), ('sc', RobustScaler())]), FITUR_ROBUST),
        ('std', StandardScaler(), FITUR_STD),
    ])

