# -*- coding: utf-8 -*-
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from math import radians, sin, cos, sqrt, atan2, log10
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
import time
import warnings
import gc
from tqdm import tqdm

# Geocoding için
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    print("Uyarı: 'geopy' kütüphanesi bulunamadı. Bölge isimleri alınamayacak.")
    print("Kurulum için: pip install geopy")

# !!! YENİ: Harita görselleştirmesi için kütüphaneler !!!
try:
    import folium
    from branca.colormap import linear # Renk skalası için
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Uyarı: 'folium' ve 'branca' kütüphaneleri bulunamadı. Harita çıktısı oluşturulamayacak.")
    print("Kurulum için: pip install folium branca")


# Uyarıları Yönetme
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')

# --- 1. Ayarlar ve Sabitler ---
MIN_LATITUDE = 35.0; MAX_LATITUDE = 43.0
MIN_LONGITUDE = 25.0; MAX_LONGITUDE = 45.0
GRID_RES_LAT = 0.2; GRID_RES_LON = 0.2 
DATA_YEARS = 50 
END_TIME = datetime.now(timezone.utc)
START_TIME = END_TIME - timedelta(days=DATA_YEARS*365.25)
USGS_API_BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
# Özellik/Tahmin Pencereleri
FEATURE_WINDOW_DAYS = 90; PREDICTION_WINDOW_DAYS = 30; TIME_STEP_DAYS = 7
# Eşikler
MAGNITUDE_THRESHOLD = 5.0; B_VALUE_MIN_QUAKES = 25; B_VALUE_MC_METHOD = 'MAXC'
# Gelecek Tahmini Ayarları
FUTURE_PREDICTION_PROB_THRESHOLD = 0.5
# Çıktı Dosyaları
TIMESTAMP_STR = END_TIME.strftime('%Y%m%d_%H%M')
PREDICTION_OUTPUT_FILE = f"deprem_tahminleri_{TIMESTAMP_STR}.txt"
# !!! YENİ: Harita dosyası adı !!!
PREDICTION_MAP_FILE = f"deprem_olasılık_haritası_{TIMESTAMP_STR}.html"
XGB_TREE_METHOD = 'hist'; API_SLEEP_TIME = 1.0

# --- Özellik İsimleri ---
feature_names = ["Deprem Sayısı", "Ort. Büyüklük", "Maks. Büyüklük", "Enerji Vekili", "Ort. Zaman Farkı (sn)", "Std. Zaman Farkı (sn)", "b-değeri", "Mc (Tamamlama Mag.)", "Ort. Derinlik"]

# --- 2. Yardımcı Fonksiyonlar ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0; lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad; dlat = lat2_rad - lat1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a)); return R * c

def fetch_usgs_data_paginated(start_time, end_time, min_lat, max_lat, min_lon, max_lon, min_mag=1.0):
    all_features_list = []
    offset = 1
    limit = 20000
    total_fetched = 0
    max_retries = 5
    request_timeout = 120

    print(f"USGS API'den veri çekiliyor ({start_time.date()} - {end_time.date()})...")
    print(f"Bölge: Lat[{min_lat}-{max_lat}], Lon[{min_lon}-{max_lon}], MinMag={min_mag}")

    response = None
    while True:
        params = {
            'format': 'geojson','starttime': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'endtime': end_time.strftime('%Y-%m-%dT%H:%M:%S'),'minlatitude': min_lat,
            'maxlatitude': max_lat,'minlongitude': min_lon, 'maxlongitude': max_lon,
            'minmagnitude': min_mag,'limit': limit,'offset': offset,'orderby': 'time-asc'
        }
        retries = 0; request_successful = False; count_returned = 0
        while retries < max_retries:
            try:
                response = requests.get(USGS_API_BASE_URL, params=params, timeout=request_timeout)
                response.raise_for_status(); data = response.json(); features = data.get('features', [])
                count_returned = len(features); total_fetched += count_returned; all_features_list.extend(features)
                if count_returned < limit: request_successful = True; break
                offset += limit; time.sleep(API_SLEEP_TIME); request_successful = True; break
            except requests.exceptions.Timeout: retries += 1; print(f"\nHata: Timeout (offset {offset}, deneme {retries}). {5*retries}sn bekle..."); time.sleep(5 * retries)
            except requests.exceptions.HTTPError as e:
                 print(f"\nHata: HTTP (offset {offset}): {e.response.status_code} {e.response.reason}"); response_text = ""
                 if e.response is not None and e.response.text is not None: response_text = str(e.response.text).lower()
                 if e.response.status_code == 400 and "offset" in response_text: request_successful = True; break
                 elif e.response.status_code == 429: time.sleep(30 * (retries + 1))
                 elif e.response.status_code >= 500: time.sleep(15 * (retries + 1))
                 else: break
                 retries += 1
            except (requests.exceptions.RequestException, json.JSONDecodeError, Exception) as e: retries += 1; print(f"\nHata: Diğer (offset {offset}, deneme {retries}): {e}. {5*retries}sn bekle..."); time.sleep(5 * retries)
        if not request_successful and retries == max_retries: print(f"\nHata: Max deneme (offset {offset})."); break
        if request_successful and count_returned < limit: break
        response_text_outer = "";
        if response is not None and response.text is not None: response_text_outer = str(response.text).lower()
        if response is not None and response.status_code == 400 and "offset" in response_text_outer: break
        if not request_successful and retries < max_retries: print("Kurtarılamayan hata."); break
    if total_fetched > 0: print(f"...Veri çekme tamamlandı. Toplam {total_fetched} olay alındı.")
    else: print("Veri çekme tamamlandı ancak hiç olay alınamadı.")
    if not all_features_list: return pd.DataFrame()
    print("Veriler DataFrame'e dönüştürülüyor...")
    earthquakes_data = []; processed_ids = set()
    for feature in all_features_list:
        feature_id = feature.get('id');
        if feature_id in processed_ids: continue
        try:
            prop = feature.get('properties', {}); geom = feature.get('geometry', {}); coords = geom.get('coordinates', [None, None, None]); time_ms = prop.get('time')
            dt_object = datetime.fromtimestamp(time_ms / 1000.0, tz=timezone.utc) if time_ms is not None else None; mag = prop.get('mag')
            if dt_object is None or coords[1] is None or coords[0] is None or mag is None: continue
            earthquakes_data.append({'id': feature_id, 'timestamp': dt_object,'latitude': coords[1], 'longitude': coords[0], 'depth': coords[2],'magnitude': mag, 'mag_type': prop.get('magType', ''), 'place': prop.get('place')})
            processed_ids.add(feature_id)
        except Exception as e: print(f"Uyarı: Olay işlenirken hata ({feature_id}): {e}"); continue
    if not earthquakes_data: return pd.DataFrame()
    df = pd.DataFrame(earthquakes_data); df = df.dropna(subset=['timestamp', 'latitude', 'longitude', 'magnitude'])
    if df.empty: return pd.DataFrame()
    df = df.sort_values('timestamp').reset_index(drop=True)
    try: df['latitude'] = df['latitude'].astype(np.float32); df['longitude'] = df['longitude'].astype(np.float32); df['depth'] = pd.to_numeric(df['depth'], errors='coerce').astype(np.float32); df['magnitude'] = df['magnitude'].astype(np.float32)
    except Exception as e: print(f"Veri tipi dönüştürme hatası: {e}")
    print(f"Geçerli {len(df)} deprem DataFrame'e yüklendi.")
    return df

def calculate_b_value(magnitudes, mc_method='MAXC', min_quakes=30, mag_bin_width=0.1):
    magnitudes = pd.Series(magnitudes).dropna().astype(np.float32)
    n_magnitudes = len(magnitudes)
    if n_magnitudes < min_quakes: return np.nan, np.nan
    try:
        min_mag_val, max_mag_val = magnitudes.min(), magnitudes.max()
        if max_mag_val > min_mag_val + 15: max_mag_val = min_mag_val + 15
        if min_mag_val >= max_mag_val: return np.nan, min_mag_val
        bins = np.arange(min_mag_val, max_mag_val + mag_bin_width, mag_bin_width)
        if len(bins) < 2 : return np.nan, np.nan
        hist, bin_edges = np.histogram(magnitudes, bins=bins)
        if np.all(hist == 0): return np.nan, np.nan
        mag_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        max_freq_bin_index = np.argmax(hist)
        if max_freq_bin_index >= len(mag_centers): return np.nan, np.nan
        mc = mag_centers[max_freq_bin_index]
        mags_above_mc = magnitudes[magnitudes >= mc]
        n_above_mc = len(mags_above_mc)
        if n_above_mc < max(3, min_quakes / 10): return np.nan, mc
        mean_mag_above_mc = mags_above_mc.mean()
        mc_adjusted = mc - (mag_bin_width / 2.0)
        if mean_mag_above_mc <= mc_adjusted + 1e-6 : return np.nan, mc
        b_value = (1.0 / (mean_mag_above_mc - mc_adjusted)) * np.log10(np.exp(1))
        if not (0.4 < b_value < 2.5): return np.nan, mc
        return b_value, mc
    except Exception: return np.nan, np.nan

def generate_features_for_cells(target_time, earthquake_df, lat_bins, lon_bins, feature_window_days, b_value_min_quakes, b_value_mc_method):
    feature_start = target_time - timedelta(days=feature_window_days)
    feature_end = target_time
    relevant_quakes = earthquake_df[(earthquake_df['timestamp'] >= feature_start) & (earthquake_df['timestamp'] < feature_end)].copy()
    n_lat = len(lat_bins) - 1; n_lon = len(lon_bins) - 1
    full_index = pd.MultiIndex.from_product([range(n_lat), range(n_lon)], names=['lat_bin', 'lon_bin'])
    cell_features = pd.DataFrame(index=full_index, columns=feature_names, dtype=np.float32)

    if relevant_quakes.empty: cell_features[['Deprem Sayısı', 'Enerji Vekili', 'Ort. Büyüklük', 'Maks. Büyüklük']] = 0; return cell_features
    relevant_quakes['lat_bin'] = pd.cut(relevant_quakes['latitude'], bins=lat_bins, labels=False, right=False)
    relevant_quakes['lon_bin'] = pd.cut(relevant_quakes['longitude'], bins=lon_bins, labels=False, right=False)
    relevant_quakes = relevant_quakes.dropna(subset=['lat_bin', 'lon_bin', 'latitude', 'longitude'])
    if relevant_quakes.empty: cell_features[['Deprem Sayısı', 'Enerji Vekili', 'Ort. Büyüklük', 'Maks. Büyüklük']] = 0; return cell_features
    relevant_quakes['lat_bin'] = relevant_quakes['lat_bin'].astype(int); relevant_quakes['lon_bin'] = relevant_quakes['lon_bin'].astype(int)
    relevant_quakes = relevant_quakes[(relevant_quakes['lat_bin'] >= 0) & (relevant_quakes['lat_bin'] < n_lat) & (relevant_quakes['lon_bin'] >= 0) & (relevant_quakes['lon_bin'] < n_lon)]
    if relevant_quakes.empty: cell_features[['Deprem Sayısı', 'Enerji Vekili', 'Ort. Büyüklük', 'Maks. Büyüklük']] = 0; return cell_features

    grouped = relevant_quakes.groupby(['lat_bin', 'lon_bin'], observed=False)
    counts = grouped.size(); mean_mag = grouped['magnitude'].mean(); max_mag = grouped['magnitude'].max(); mean_depth = grouped['depth'].mean()
    if not counts.empty: cell_features.loc[counts.index, 'Deprem Sayısı'] = counts
    if not mean_mag.empty: cell_features.loc[mean_mag.index, 'Ort. Büyüklük'] = mean_mag
    if not max_mag.empty: cell_features.loc[max_mag.index, 'Maks. Büyüklük'] = max_mag
    if not mean_depth.empty: cell_features.loc[mean_depth.index, 'Ort. Derinlik'] = mean_depth # Allows NaNs
    energy_proxy = grouped['magnitude'].apply(lambda x: np.sum(np.power(10.0, 1.5 * np.clip(x.astype(np.float64), -2, 9))))
    if not energy_proxy.empty: cell_features.loc[energy_proxy.index, 'Enerji Vekili'] = energy_proxy
    def calc_time_stats(x_ts):
        if len(x_ts) < 2: return np.nan, np.nan
        diffs = x_ts.sort_values().diff().dt.total_seconds().dropna(); mean_diff, std_diff = np.nan, np.nan
        if not diffs.empty and len(diffs) > 0: mean_diff = diffs.mean(); std_diff = diffs.std() if len(diffs) > 1 else 0.0
        return mean_diff, std_diff
    time_stats = grouped['timestamp'].apply(calc_time_stats)
    if not time_stats.empty: cell_features.loc[time_stats.index, 'Ort. Zaman Farkı (sn)'] = time_stats.apply(lambda x: x[0]); cell_features.loc[time_stats.index, 'Std. Zaman Farkı (sn)'] = time_stats.apply(lambda x: x[1])
    b_mc_values = grouped['magnitude'].apply(lambda x: calculate_b_value(x, mc_method=b_value_mc_method, min_quakes=b_value_min_quakes))
    if not b_mc_values.empty: cell_features.loc[b_mc_values.index, 'b-değeri'] = b_mc_values.apply(lambda x: x[0]); cell_features.loc[b_mc_values.index, 'Mc (Tamamlama Mag.)'] = b_mc_values.apply(lambda x: x[1])
    cell_features['Deprem Sayısı'] = cell_features['Deprem Sayısı'].fillna(0); cell_features['Enerji Vekili'] = cell_features['Enerji Vekili'].fillna(0); cell_features['Ort. Büyüklük'] = cell_features['Ort. Büyüklük'].fillna(0); cell_features['Maks. Büyüklük'] = cell_features['Maks. Büyüklük'].fillna(0)
    for col in cell_features.columns:
        if cell_features[col].dtype != np.float32: cell_features[col] = pd.to_numeric(cell_features[col], errors='coerce').astype(np.float32)
    return cell_features

# --- 3. Grid Oluşturma ---
lat_bins = np.arange(MIN_LATITUDE, MAX_LATITUDE + GRID_RES_LAT, GRID_RES_LAT, dtype=np.float32)
lon_bins = np.arange(MIN_LONGITUDE, MAX_LONGITUDE + GRID_RES_LON, GRID_RES_LON, dtype=np.float32)
n_lat_cells = len(lat_bins) - 1; n_lon_cells = len(lon_bins) - 1
print(f"\nGrid oluşturuldu: {n_lat_cells}x{n_lon_cells} = {n_lat_cells * n_lon_cells} hücre")
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2; lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2

# --- 4. Veri Çekme ve Ön İşleme ---
earthquake_catalog = fetch_usgs_data_paginated(START_TIME, END_TIME, MIN_LATITUDE, MAX_LATITUDE, MIN_LONGITUDE, MAX_LONGITUDE, min_mag=1.5) # Sadece USGS kullanalım şimdilik
if earthquake_catalog is None or earthquake_catalog.empty: exit("Veri çekilemedi veya boş.")
earthquake_catalog['timestamp'] = pd.to_datetime(earthquake_catalog['timestamp'])

# --- 5. Zaman Adımlı Özellik ve Hedef Oluşturma ---
print("\nZaman adımlı özellik mühendisliği başlıyor...")
all_step_features, all_step_targets, all_step_timestamps, all_step_indices = [], [], [], []
start_loop_time = time.time()
loop_start_time = earthquake_catalog['timestamp'].min() + timedelta(days=FEATURE_WINDOW_DAYS)
if loop_start_time > earthquake_catalog['timestamp'].max(): exit("Hata: Özellik penceresi çok uzun.")
loop_end_time = earthquake_catalog['timestamp'].max() - timedelta(days=PREDICTION_WINDOW_DAYS)
if loop_start_time > loop_end_time: print(f"Uyarı: Veri aralığı yetersiz."); loop_end_time = loop_start_time - timedelta(days=1)
current_time = loop_start_time
total_steps = 0
if loop_end_time >= loop_start_time: total_steps = int((loop_end_time - loop_start_time).total_seconds() / timedelta(days=TIME_STEP_DAYS).total_seconds()) + 1
feature_pbar = tqdm(total=total_steps, desc="Özellik Hesaplama Adımları")
step_count = 0
while current_time <= loop_end_time:
    step_count += 1
    features_df = generate_features_for_cells(current_time, earthquake_catalog, lat_bins, lon_bins, FEATURE_WINDOW_DAYS, B_VALUE_MIN_QUAKES, B_VALUE_MC_METHOD) # B_VALUE_MC_METHOD eklendi
    prediction_start = current_time; prediction_end = current_time + timedelta(days=PREDICTION_WINDOW_DAYS)
    prediction_quakes = earthquake_catalog[(earthquake_catalog['timestamp'] > prediction_start) & (earthquake_catalog['timestamp'] <= prediction_end)].copy()
    targets = pd.Series(0, index=features_df.index, dtype=np.int8)
    if not prediction_quakes.empty:
        prediction_quakes['lat_bin'] = pd.cut(prediction_quakes['latitude'], bins=lat_bins, labels=False, right=False)
        prediction_quakes['lon_bin'] = pd.cut(prediction_quakes['longitude'], bins=lon_bins, labels=False, right=False)
        prediction_quakes = prediction_quakes.dropna(subset=['lat_bin', 'lon_bin'])
        if not prediction_quakes.empty:
            prediction_quakes['lat_bin'] = prediction_quakes['lat_bin'].astype(int); prediction_quakes['lon_bin'] = prediction_quakes['lon_bin'].astype(int)
            prediction_quakes = prediction_quakes[(prediction_quakes['lat_bin'] >= 0) & (prediction_quakes['lat_bin'] < n_lat_cells) & (prediction_quakes['lon_bin'] >= 0) & (prediction_quakes['lon_bin'] < n_lon_cells)]
            large_quake_cells = prediction_quakes[prediction_quakes['magnitude'] >= MAGNITUDE_THRESHOLD]
            if not large_quake_cells.empty:
                target_indices = pd.MultiIndex.from_frame(large_quake_cells[['lat_bin', 'lon_bin']])
                valid_target_indices = targets.index.intersection(target_indices)
                if not valid_target_indices.empty: targets.loc[valid_target_indices] = 1
    step_data = features_df.copy(); step_data['target'] = targets; step_data_reset = step_data.reset_index(); step_data_clean = step_data_reset.copy()
    for col in feature_names:
        if step_data_clean[col].isnull().any(): median_val = step_data_clean[col].median(); step_data_clean[col] = step_data_clean[col].fillna(median_val if not pd.isnull(median_val) else 0)
    if not step_data_clean.empty:
        all_step_features.append(step_data_clean[feature_names].values.astype(np.float32)); all_step_targets.append(step_data_clean['target'].values.astype(np.int8)); all_step_timestamps.extend([current_time] * len(step_data_clean)); all_step_indices.append(step_data_clean[['lat_bin', 'lon_bin']].values.astype(np.int16))
    feature_pbar.update(1)
    if step_count % 50 == 0: feature_pbar.set_description(f"Özellik Hesabı {current_time.date()}")
    del features_df, targets, prediction_quakes, step_data, step_data_reset, step_data_clean; gc.collect()
    if 'target_indices' in locals(): del target_indices
    if 'valid_target_indices' in locals(): del valid_target_indices
    if 'large_quake_cells' in locals(): del large_quake_cells
    current_time += timedelta(days=TIME_STEP_DAYS)
feature_pbar.close()
end_loop_time_proc = time.time(); print(f"...Özellik mühendisliği tamamlandı ({end_loop_time_proc - start_loop_time:.1f} saniye).")
if not all_step_features: exit("\nHiç özellik/hedef çifti oluşturulamadı.")
print("Numpy dizileri birleştiriliyor...");
try: X = np.concatenate(all_step_features, axis=0).astype(np.float32); y = np.concatenate(all_step_targets, axis=0).astype(np.int8); timestamps = np.array(all_step_timestamps); cell_indices = np.concatenate(all_step_indices, axis=0).astype(np.int16)
except ValueError as e: exit(f"Hata: Numpy dizileri birleştirilemedi: {e}")
del all_step_features, all_step_targets, all_step_timestamps, all_step_indices; gc.collect(); print("Birleştirme tamamlandı.")
print(f"\nToplam {len(X)} zaman-hücre örneği oluşturuldu."); print(f"Özellik sayısı: {X.shape[1]}")
if len(X) == 0: exit("Hiç örnek oluşturulamadı.")
target_counts = np.bincount(y); print(f"Hedef Yok (0): {target_counts[0] if len(target_counts)>0 else 0}"); print(f"Hedef Var (1): {target_counts[1] if len(target_counts)>1 else 0}")
if len(target_counts) < 2 or target_counts[1] == 0: exit("\nHedef sınıfta (1) hiç örnek yok! Model eğitilemiyor.")

# --- 6. Zaman Serisi Çapraz Doğrulama ---
print("\nZaman Serisi Çapraz Doğrulama ile XGBoost Modeli Eğitimi..."); n_splits = 5; tscv = TimeSeriesSplit(n_splits=n_splits); fold_metrics = []
pos_count = (y == 1).sum(); neg_count = (y == 0).sum(); scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1; print(f"XGB scale_pos_weight: {scale_pos_weight:.2f}")
params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'eta': 0.1, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8,'min_child_weight': 1, 'gamma': 0.1, 'lambda': 1, 'alpha': 0, 'seed': 42, 'nthread': -1, 'scale_pos_weight': scale_pos_weight, 'tree_method': XGB_TREE_METHOD}
if XGB_TREE_METHOD == 'gpu_hist': params['gpu_id'] = 0
num_boost_round = 150; early_stopping_rounds = 15; all_test_preds, all_test_true = [], []; final_model = None; best_iteration_from_cv = num_boost_round
print("Çapraz doğrulama katmanları işleniyor...")
cv_pbar = tqdm(tscv.split(X), total=n_splits, desc="CV Katmanları")
for fold, (train_index, test_index) in enumerate(cv_pbar):
    cv_pbar.set_description(f"CV Katman {fold + 1}/{n_splits}"); X_train, X_test = X[train_index], X[test_index]; y_train, y_test = y[train_index], y[test_index]
    if len(X_test) == 0: continue
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names); dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names); watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
    if fold == n_splits - 1: best_iteration_from_cv = model.best_iteration
    y_pred_proba = model.predict(dtest, iteration_range=(0, model.best_iteration)); y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = (y_pred == y_test).mean(); precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0); roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
    fold_metrics.append({'acc': accuracy, 'prec': precision, 'rec': recall, 'f1': f1, 'auc': roc_auc}); all_test_preds.extend(y_pred_proba); all_test_true.extend(y_test)
    cv_pbar.set_postfix(AUC=f"{roc_auc:.3f}", F1=f"{f1:.3f}", Recall=f"{recall:.3f}")
    del X_train, X_test, y_train, y_test, dtrain, dtest, model; gc.collect()
cv_pbar.close()

# --- 7. Genel Değerlendirme ---
if fold_metrics: print("\n--- Genel Çapraz Doğrulama Sonuçları (Ortalama) ---\n", pd.DataFrame(fold_metrics).mean())
else: print("\nHiç CV katmanı işlenmedi.")

# --- 8. Final Model Eğitimi ---
print("\n--- Final Model Eğitimi (Tüm Veriyle) ---")
if len(X) > 0:
    print(f"Final eğitim seti boyutu: {len(X)}"); dfinaltrain = xgb.DMatrix(X, label=y, feature_names=feature_names); watchlist_final = [(dfinaltrain, 'train')]; final_num_rounds = max(50, best_iteration_from_cv + 10); print(f"Final model {final_num_rounds} tur ile eğitiliyor...")
    final_model = xgb.train(params, dfinaltrain, num_boost_round=final_num_rounds, evals=watchlist_final, verbose_eval=50)
    print("\n--- Özellik Önem Sıralaması (Final Model) ---")
    try:
        importance = final_model.get_score(importance_type='gain');
        if importance: fmap = {f'f{i}': name for i, name in enumerate(feature_names)}; mapped_importance = {fmap.get(f, f): score for f, score in importance.items()}; sorted_importance = sorted(mapped_importance.items(), key=lambda item: item[1], reverse=True); [print(f"- {f}: {s:.2f}") for f, s in sorted_importance]
        else: print("Önem skorları alınamadı.")
    except Exception as e: print(f"Özellik önemi hatası: {e}")
    del X, y, timestamps, cell_indices, dfinaltrain; gc.collect()
else: print("Final model için veri yok."); final_model = None

# --- 9. GELECEK TAHMİNİ VE GÖRSELLEŞTİRME ---
print("\n--- GELECEK TAHMİNİ VE HARİTA OLUŞTURMA ---"); print(f"Uyarı: Deneysel olasılıklar...")
high_prob_cells_output = []
geolocator = None; reverse = None
if GEOPY_AVAILABLE:
    try: geolocator = Nominatim(user_agent="my_earthquake_predictor_v3"); reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1.1, return_value_on_exception=None); print("Geocoder başlatıldı.")
    except Exception as e: print(f"Hata: Geocoder başlatılamadı: {e}"); GEOPY_AVAILABLE = False

if final_model is not None:
    print(f"\n'{END_TIME.date()}' itibarıyla özellikler hesaplanıyor...")
    now_features_df = generate_features_for_cells(END_TIME, earthquake_catalog, lat_bins, lon_bins, FEATURE_WINDOW_DAYS, B_VALUE_MIN_QUAKES, B_VALUE_MC_METHOD)
    now_features_filled_df = now_features_df.copy()
    for col in feature_names:
         if now_features_filled_df[col].isnull().any(): now_features_filled_df[col] = now_features_filled_df[col].fillna(0)

    if not now_features_filled_df.empty:
        print(f"{len(now_features_filled_df)} hücre için özellikler hazırlandı."); X_now = now_features_filled_df[feature_names].values.astype(np.float32); dnow = xgb.DMatrix(X_now, feature_names=feature_names); print("Olasılıklar tahmin ediliyor...")
        try: best_iter = getattr(final_model, 'best_iteration', final_num_rounds); future_probabilities = final_model.predict(dnow, iteration_range=(0, best_iter))
        except Exception as e: print(f"Tahmin hatası: {e}"); future_probabilities = None
        if future_probabilities is not None:
            predictions_df = now_features_filled_df.reset_index(); predictions_df['probability'] = future_probabilities
            map_prob_threshold = max(0.4, FUTURE_PREDICTION_PROB_THRESHOLD - 0.2)
            map_cells = predictions_df[predictions_df['probability'] >= map_prob_threshold].sort_values('probability', ascending=False)
            high_prob_cells_for_text = predictions_df[predictions_df['probability'] >= FUTURE_PREDICTION_PROB_THRESHOLD].sort_values('probability', ascending=False)

            if not map_cells.empty: # Harita için hücre varsa devam et
                print(f"Harita için eşiği ({map_prob_threshold*100:.0f}%) geçen {len(map_cells)} hücre bulundu.")
                print("Bölge isimleri alınıyor...")
                geo_pbar = tqdm(map_cells.iterrows(), total=len(map_cells), desc="Bölge İsimleri (Harita)")
                for _, row in geo_pbar:
                    lat_idx, lon_idx, prob = int(row['lat_bin']), int(row['lon_bin']), row['probability']; region_name, gmaps_link = "Bilinmiyor", "N/A"
                    if 0 <= lat_idx < len(lat_centers) and 0 <= lon_idx < len(lon_centers):
                        lat, lon = lat_centers[lat_idx], lon_centers[lon_idx]; gmaps_link = f"https://www.google.com/maps?q={lat:.4f},{lon:.4f}"
                        if GEOPY_AVAILABLE and reverse is not None:
                            try:
                                location = reverse(f"{lat:.6f}, {lon:.6f}", language='tr', addressdetails=True, timeout=10)
                                if location: address = location.raw.get('address', {}); parts = [address.get(k) for k in ['village', 'town', 'suburb', 'county', 'state', 'country'] if address.get(k)]; region_name = ", ".join(parts) if parts else location.address
                                else: region_name = "Bölge bulunamadı"
                            except Exception: region_name = "Geocoder Hatası"
                        else: region_name = "Geocoder Yok"
                        # Çıktı listesine sadece TEXT dosyası için olanları ekle
                        if prob >= FUTURE_PREDICTION_PROB_THRESHOLD:
                             high_prob_cells_output.append({'latitude': lat,'longitude': lon,'probability': prob * 100,'region': region_name,'gmaps_link': gmaps_link})
                        # Harita verisine ekle (her hücre için)
                        map_cells.loc[row.name, 'lat_center'] = lat
                        map_cells.loc[row.name, 'lon_center'] = lon
                        map_cells.loc[row.name, 'region_name'] = region_name
                geo_pbar.close(); print("Bölge isimleri alma tamamlandı.")

                # --- Harita Oluşturma (Folium varsa) ---
                if FOLIUM_AVAILABLE:
                    print(f"\n'{PREDICTION_MAP_FILE}' haritası oluşturuluyor...")
                    try:
                        map_center_lat = map_cells['lat_center'].iloc[0] if not map_cells.empty else (MIN_LATITUDE + MAX_LATITUDE) / 2
                        map_center_lon = map_cells['lon_center'].iloc[0] if not map_cells.empty else (MIN_LONGITUDE + MAX_LONGITUDE) / 2
                        m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=6, tiles='CartoDB positron')

                        min_prob_map = map_cells['probability'].min()
                        max_prob_map = map_cells['probability'].max()

                        colormap = linear.YlOrRd_09.scale(vmin=min_prob_map, vmax=max_prob_map)
                        colormap.caption = f'M{MAGNITUDE_THRESHOLD}+ Deprem Olasılığı ({PREDICTION_WINDOW_DAYS} Günlük)'

                        # Haritaya hücreleri (dikdörtgen olarak) ekle
                        for _, row in map_cells.iterrows():
                            lat_idx, lon_idx = int(row['lat_bin']), int(row['lon_bin'])
                            prob = row['probability']
                            region = row['region_name']
                            lat_c = row['lat_center']
                            lon_c = row['lon_center']

                            # Hücre sınırlarını hesapla
                            lat_start, lat_end = lat_bins[lat_idx], lat_bins[lat_idx+1]
                            lon_start, lon_end = lon_bins[lon_idx], lon_bins[lon_idx+1]
                            bounds = [[lat_start, lon_start], [lat_end, lon_end]]

                            # Popup metnini oluştur
                            popup_html = f"""
                            <b>Koordinat:</b> {lat_c:.2f}, {lon_c:.2f}<br>
                            <b>Tahmini Bölge:</b> {region}<br>
                            <b>Olasılık (M{MAGNITUDE_THRESHOLD}+):</b> {prob*100:.1f}%<br>
                            <a href="https://www.google.com/maps?q={lat_c:.4f},{lon_c:.4f}" target="_blank">Google Haritalar</a>
                            """
                            iframe = folium.IFrame(html=popup_html, width=250, height=100)
                            popup = folium.Popup(iframe, max_width=2650)

                            # Dikdörtgeni çiz
                            folium.Rectangle(
                                bounds=bounds,
                                popup=popup,
                                tooltip=f"{prob*100:.1f}% - {region}", # Mouse üzerine gelince görünen yazı
                                color='#333333', # Kenarlık rengi
                                weight=0.5,      # Kenarlık kalınlığı
                                fill=True,
                                fillColor=colormap(prob), # Olasılığa göre renk
                                fillOpacity=0.6  # Dolgu şeffaflığı
                            ).add_to(m)

                        # Haritaya renk skalasını ekle
                        colormap.add_to(m)

                        # Haritayı HTML olarak kaydet
                        m.save(PREDICTION_MAP_FILE)
                        print(f"Harita başarıyla '{PREDICTION_MAP_FILE}' olarak kaydedildi.")

                    except Exception as map_e:
                        print(f"Hata: Harita oluşturulurken hata oluştu: {map_e}")
                else:
                    print("Folium kütüphanesi bulunamadığı için harita oluşturulamadı.")
            elif not high_prob_cells_for_text.empty:
                print(f"Text dosyası için eşiği ({FUTURE_PREDICTION_PROB_THRESHOLD*100:.0f}%) geçen {len(high_prob_cells_for_text)} hücre bulundu, ancak harita için yeterli değil.")
                geo_pbar_text = tqdm(high_prob_cells_for_text.iterrows(), total=len(high_prob_cells_for_text), desc="Bölge İsimleri (Text)")
                for _, row in geo_pbar_text:
                    lat_idx, lon_idx, prob = int(row['lat_bin']), int(row['lon_bin']), row['probability']; region_name, gmaps_link = "Bilinmiyor", "N/A"
                    if 0 <= lat_idx < len(lat_centers) and 0 <= lon_idx < len(lon_centers):
                        lat, lon = lat_centers[lat_idx], lon_centers[lon_idx]; gmaps_link = f"https://www.google.com/maps?q={lat:.4f},{lon:.4f}"
                        if GEOPY_AVAILABLE and reverse is not None:
                            try:
                                location = reverse(f"{lat:.6f}, {lon:.6f}", language='tr', addressdetails=True, timeout=10)
                                if location: address = location.raw.get('address', {}); parts = [address.get(k) for k in ['village', 'town', 'suburb', 'county', 'state', 'country'] if address.get(k)]; region_name = ", ".join(parts) if parts else location.address
                                else: region_name = "Bölge bulunamadı"
                            except Exception: region_name = "Geocoder Hatası"
                        else: region_name = "Geocoder Yok"
                        high_prob_cells_output.append({'latitude': lat,'longitude': lon,'probability': prob * 100,'region': region_name,'gmaps_link': gmaps_link})
                geo_pbar_text.close(); print("Bölge isimleri alma tamamlandı (sadece text için).")

            else:
                print(f"Belirtilen eşiği ({map_prob_threshold*100:.0f}% veya {FUTURE_PREDICTION_PROB_THRESHOLD*100:.0f}%) geçen hücre bulunamadı.")
        else: print("Olasılıklar hesaplanamadı.")
    else: print("Gelecek tahmini için hücre bulunamadı.")
else: print("Final model yok, tahmin yapılamıyor.")

# --- Dosyaya Yazma ---
if high_prob_cells_output:
    print(f"\nYüksek olasılıklı tahminler '{PREDICTION_OUTPUT_FILE}' dosyasına yazılıyor...")
    try:
        with open(PREDICTION_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"# Gelecek {PREDICTION_WINDOW_DAYS} Gün İçin M{MAGNITUDE_THRESHOLD}+ Deprem Olasılık Tahminleri\n# Tahmin Tarihi: {END_TIME.strftime('%Y-%m-%d %H:%M:%S UTC')}\n# Olasılık Eşiği: {FUTURE_PREDICTION_PROB_THRESHOLD*100:.0f}%\n# UYARI: Deneysel sonuçlardır.\n")
            f.write("-" * 120 + "\n"); f.write(f"{'Enlem':<10} {'Boylam':<10} {'Olasılık (%)':<15} {'Tahmini Bölge':<60} {'Google Haritalar Linki'}\n"); f.write("-" * 120 + "\n")
            for p in high_prob_cells_output: f.write(f"{p['latitude']:<10.2f} {p['longitude']:<10.2f} {p['probability']:<15.1f} {p['region'][:58]:<60} {p['gmaps_link']}\n")
        print("Tahminler başarıyla yazıldı.")
    except IOError as e: print(f"Hata: Dosyaya yazılamadı: {e}")
    except Exception as e: print(f"Hata: Dosyaya yazma hatası: {e}")
else: print("\nDosyaya yazılacak tahmin bulunmuyor.")

print("\n--- PROGRAM SONU ---")
print("!!! TEKRAR UYARI: Bu sonuçlar bilimsel kesinlik taşımaz ve gerçek deprem tahmini değildir !!!")
