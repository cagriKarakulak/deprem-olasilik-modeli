import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from math import radians, sin, cos, sqrt, atan2, log10
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc
import time
import warnings
import gc
from tqdm import tqdm
import hashlib

try:
    from scipy.ndimage import generic_filter, uniform_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Uyarı: 'scipy' kütüphanesi bulunamadı. Komşu özellikleri optimize edilemeyecek (yavaş yöntem kullanılacak veya devre dışı bırakılacak).")
    print("Kurulum için: pip install scipy")

try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    print("Uyarı: 'geopy' kütüphanesi bulunamadı. Bölge isimleri alınamayacak.")
    print("Kurulum için: pip install geopy")

try:
    import folium
    from branca.colormap import linear
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Uyarı: 'folium' ve 'branca' kütüphaneleri bulunamadı. Harita çıktısı oluşturulamayacak.")
    print("Kurulum için: pip install folium branca")

try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Uyarı: 'optuna' kütüphanesi bulunamadı. Hiperparametre optimizasyonu yapılamayacak.")
    print("Kurulum için: pip install optuna")

try:
    ISTANBUL_TZ = pytz.timezone('Europe/Istanbul')
    print(f"Saat dilimi 'Europe/Istanbul' olarak ayarlandı.")
except pytz.UnknownTimeZoneError:
    print("Hata: 'Europe/Istanbul' saat dilimi bulunamadı. UTC kullanılacak.")
    from datetime import timezone
    ISTANBUL_TZ = timezone.utc

CONFIG = {
    "MIN_LATITUDE": 35.0,
    "MAX_LATITUDE": 43.0,
    "MIN_LONGITUDE": 25.0,
    "MAX_LONGITUDE": 45.0,
    "GRID_RES_LAT": 0.2,
    "GRID_RES_LON": 0.2,
    "DATA_YEARS": 50,
    "END_TIME": datetime.now(ISTANBUL_TZ),
    "USGS_API_URL": "https://earthquake.usgs.gov/fdsnws/event/1/query",
    "EMSC_API_URL": "https://www.seismicportal.eu/fdsnws/event/1/query",
    "API_MIN_MAGNITUDE": 1.5,
    "API_TIMEOUT": 180,
    "API_MAX_RETRIES": 5,
    "API_SLEEP_TIME": 1.0,
    "DEDUPLICATION_TIME_WINDOW": 15,
    "DEDUPLICATION_DIST_WINDOW": 50,
    "FEATURE_WINDOWS_DAYS": [90],
    "PREDICTION_WINDOW_DAYS": 30,
    "TIME_STEP_DAYS": 7,
    "MAGNITUDE_THRESHOLD": 5.0,
    "B_VALUE_MIN_QUAKES": 25,
    "B_VALUE_MC_METHOD": 'MAXC',
    "ENABLE_NEIGHBOR_FEATURES": False,
    "USE_FAST_NEIGHBORS": SCIPY_AVAILABLE and True,
    "XGB_TREE_METHOD": 'hist',
    "N_SPLITS_CV": 5,
    "XGB_EARLY_STOPPING_ROUNDS": 15,
    "XGB_NUM_BOOST_ROUND": 200,
    "ENABLE_OPTUNA": OPTUNA_AVAILABLE and False,
    "OPTUNA_N_TRIALS": 25,
    "OPTUNA_TIMEOUT": 300,
    "OPTUNA_CV_SPLITS": 3,
    "OPTUNA_DATA_FRACTION": 0.5,
    "FUTURE_PREDICTION_PROB_THRESHOLD": 0.5,
    "OUTPUT_TIMESTAMP_STR": datetime.now(ISTANBUL_TZ).strftime('%Y%m%d_%H%M%S_%Z'),
    "PREDICTION_OUTPUT_FILE": None,
    "PREDICTION_MAP_FILE": None,
}

CONFIG["START_TIME"] = CONFIG["END_TIME"] - timedelta(days=CONFIG["DATA_YEARS"] * 365.25)
if CONFIG["END_TIME"].tzinfo is not None and CONFIG["START_TIME"].tzinfo is None:
    CONFIG["START_TIME"] = ISTANBUL_TZ.localize(CONFIG["START_TIME"].replace(tzinfo=None))
elif CONFIG["END_TIME"].tzinfo is not None:
     CONFIG["START_TIME"] = CONFIG["START_TIME"].astimezone(ISTANBUL_TZ)

CONFIG["PREDICTION_OUTPUT_FILE"] = f"deprem_tahminleri_{CONFIG['OUTPUT_TIMESTAMP_STR']}.txt"
CONFIG["PREDICTION_MAP_FILE"] = f"deprem_olasılık_haritası_{CONFIG['OUTPUT_TIMESTAMP_STR']}.html"

if CONFIG["ENABLE_NEIGHBOR_FEATURES"] and CONFIG["USE_FAST_NEIGHBORS"] and not SCIPY_AVAILABLE:
    print("Uyarı: Hızlı komşu özelliği hesaplaması isteniyor ancak 'scipy' bulunamadı.")
    CONFIG["USE_FAST_NEIGHBORS"] = False

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', message='All-NaN slice encountered')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')
warnings.filterwarnings('ignore', message='invalid value encountered in double_scalars')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    try:
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
    except (ValueError, TypeError):
        distance = np.nan
    return distance

def generate_deduplication_id(timestamp, latitude, longitude, magnitude):
    time_str = timestamp.strftime('%Y%m%d%H%M%S')
    lat_str = f"{latitude:.2f}"
    lon_str = f"{longitude:.2f}"
    mag_str = f"{magnitude:.1f}"
    combined = f"{time_str}_{lat_str}_{lon_str}_{mag_str}"
    return hashlib.md5(combined.encode()).hexdigest()[:16]

def fetch_earthquake_data_paginated(api_url, source_name, start_time, end_time, min_lat, max_lat, min_lon, max_lon, min_mag, timeout, max_retries, sleep_time):
    all_features_list = []
    offset = 1
    limit = 20000
    total_fetched = 0

    if start_time.tzinfo is None:
        start_time = ISTANBUL_TZ.localize(start_time)
    if end_time.tzinfo is None:
        end_time = ISTANBUL_TZ.localize(end_time)

    try:
        start_time_utc_str = start_time.astimezone(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S')
        end_time_utc_str = end_time.astimezone(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S')
    except Exception as tz_err:
        print(f"Hata: Zaman dilimi dönüştürme hatası ({source_name}): {tz_err}. UTC varsayılıyor.")
        start_time_utc_str = start_time.strftime('%Y-%m-%dT%H:%M:%S')
        end_time_utc_str = end_time.strftime('%Y-%m-%dT%H:%M:%S')

    print(f"\n{source_name} API'den veri çekiliyor ({start_time.strftime('%Y-%m-%d')} - {end_time.strftime('%Y-%m-%d')} {start_time.tzinfo})...")
    print(f"Bölge: Lat[{min_lat}-{max_lat}], Lon[{min_lon}-{max_lon}], MinMag={min_mag}")

    while True:
        params = {
            'format': 'geojson',
            'starttime': start_time_utc_str,
            'endtime': end_time_utc_str,
            'minlatitude': min_lat,
            'maxlatitude': max_lat,
            'minlongitude': min_lon,
            'maxlongitude': max_lon,
            'minmagnitude': min_mag,
            'limit': limit,
            'offset': offset,
            'orderby': 'time-asc'
        }
        retries = 0
        request_successful = False
        count_returned = 0
        response = None

        while retries < max_retries:
            try:
                response = requests.get(api_url, params=params, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                features = data.get('features', [])
                count_returned = len(features)
                total_fetched += count_returned
                all_features_list.extend(features)
                print(f"\r{source_name}: {total_fetched} olay alındı (offset {offset}, son istek: {count_returned})...", end="")

                if count_returned < limit:
                    request_successful = True
                    break

                offset += limit
                time.sleep(sleep_time)
                request_successful = True
                break

            except requests.exceptions.Timeout:
                retries += 1
                wait_time = 5 * retries
                print(f"\n{source_name} Hata: Timeout (offset {offset}, deneme {retries}/{max_retries}). {wait_time}sn bekleniyor...")
                time.sleep(wait_time)
            except requests.exceptions.HTTPError as e:
                 print(f"\n{source_name} Hata: HTTP Hatası (offset {offset}, deneme {retries+1}/{max_retries}): {e.response.status_code} {e.response.reason}")
                 response_text = ""
                 is_offset_error = False
                 if e.response is not None and e.response.text is not None:
                     response_text = str(e.response.text).lower()
                     is_offset_error = e.response.status_code == 400 and ("offset" in response_text or "parameter" in response_text or "page size" in response_text or "start index" in response_text)

                 if is_offset_error:
                     print(f"{source_name}: Offset/parametre hatası alındı, muhtemelen veri sonuna ulaşıldı.")
                     request_successful = True
                     break
                 elif e.response.status_code == 429:
                     wait_time = 30 * (retries + 1)
                     print(f"{source_name}: Hız limiti aşıldı (Rate Limit). {wait_time}sn bekleniyor...")
                     time.sleep(wait_time)
                 elif e.response.status_code >= 500:
                     wait_time = 15 * (retries + 1)
                     print(f"{source_name}: Sunucu hatası ({e.response.status_code}). {wait_time}sn bekleniyor...")
                     time.sleep(wait_time)
                 else:
                     print(f"{source_name}: Kurtarılamayan HTTP hatası: {e}")
                     break
                 retries += 1
            except (requests.exceptions.RequestException, json.JSONDecodeError, Exception) as e:
                retries += 1
                wait_time = 5 * retries
                print(f"\n{source_name} Hata: Diğer Hata (offset {offset}, deneme {retries}/{max_retries}): {type(e).__name__} - {e}. {wait_time}sn bekleniyor...")
                time.sleep(wait_time)

        if not request_successful and retries == max_retries:
            print(f"\n{source_name} Hata: Maksimum deneme sayısına ulaşıldı (offset {offset}). Veri alımı bu noktada durduruldu.")
            break
        if request_successful and count_returned < limit:
             break

        if response is not None and response.status_code == 400:
             response_text_outer = str(response.text).lower()
             if "offset" in response_text_outer or "parameter" in response_text_outer or "page size" in response_text_outer or "start index" in response_text_outer:
                 print(f"\n{source_name}: Offset hatası (dış kontrol), muhtemelen veri sonu.")
                 break

        if not request_successful:
            print(f"\n{source_name}: Kurtarılamayan bir hata nedeniyle veri alımı durduruldu.")
            break

    print(f"\n...{source_name} veri çekme tamamlandı. Toplam {total_fetched} olay API'den alındı.")

    if not all_features_list:
        return pd.DataFrame()

    print(f"{source_name}: Alınan veriler DataFrame'e dönüştürülüyor...")
    earthquakes_data = []
    processed_ids = set()

    for feature in tqdm(all_features_list, desc=f"İşleniyor ({source_name})", leave=False):
        feature_id = feature.get('id')
        if not feature_id or feature_id in processed_ids:
            continue

        try:
            properties = feature.get('properties', {})
            geometry = feature.get('geometry', {})
            coordinates = geometry.get('coordinates', [None, None, None])
            time_ms = properties.get('time')

            mag = properties.get('mag')
            mag_type = properties.get('magType', '')
            place = properties.get('place')

            if time_ms is None or coordinates[1] is None or coordinates[0] is None or mag is None or mag < -1.0:
                continue

            try:
                dt_object = datetime.fromtimestamp(time_ms / 1000.0, tz=pytz.utc)
            except (ValueError, TypeError):
                 continue

            depth = coordinates[2] if coordinates[2] is not None else np.nan

            earthquakes_data.append({
                'source_id': f"{source_name}_{feature_id}",
                'timestamp': dt_object,
                'latitude': coordinates[1],
                'longitude': coordinates[0],
                'depth': depth,
                'magnitude': mag,
                'mag_type': mag_type,
                'place': place,
                'source': source_name
            })
            processed_ids.add(feature_id)
        except Exception as e:
            continue

    if not earthquakes_data:
        print(f"{source_name}: DataFrame'e eklenecek geçerli olay bulunamadı.")
        return pd.DataFrame()

    df = pd.DataFrame(earthquakes_data)
    df = df.dropna(subset=['timestamp', 'latitude', 'longitude', 'magnitude'])

    try:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').astype(np.float32)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').astype(np.float32)
        df['depth'] = pd.to_numeric(df['depth'], errors='coerce').astype(np.float32)
        df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce').astype(np.float32)
        df = df.dropna(subset=['latitude', 'longitude', 'magnitude'])
    except Exception as e:
        print(f"\n{source_name} Hata: Veri tipi dönüştürme sırasında: {e}")
        return pd.DataFrame()

    print(f"{source_name}: {len(df)} geçerli deprem DataFrame'e yüklendi.")
    return df.sort_values('timestamp').reset_index(drop=True)

def merge_and_deduplicate_catalogs(catalogs, time_window_sec, dist_window_km):
    if not catalogs:
        return pd.DataFrame()

    print("\nKataloglar birleştiriliyor ve tekilleştiriliyor...")
    valid_catalogs = [df for df in catalogs if df is not None and not df.empty]
    if not valid_catalogs:
        print("Birleştirmek için geçerli katalog bulunamadı.")
        return pd.DataFrame()

    combined_df = pd.concat(valid_catalogs, ignore_index=True)
    print(f"Birleştirme sonrası toplam olay sayısı: {len(combined_df)}")
    if combined_df.empty:
        return combined_df

    combined_df = combined_df.sort_values(['timestamp', 'magnitude'], ascending=[True, False]).reset_index(drop=True)

    duplicates = pd.Series(False, index=combined_df.index)
    processed_indices = set()

    print("Tekilleştirme işlemi başlıyor...")
    pbar_dedup = tqdm(total=len(combined_df), desc="Tekilleştirme", leave=False, mininterval=1.0)

    for i in range(len(combined_df)):
        if i in processed_indices:
            pbar_dedup.update(1)
            continue

        ts1 = combined_df.iloc[i]['timestamp']
        lat1 = combined_df.iloc[i]['latitude']
        lon1 = combined_df.iloc[i]['longitude']

        time_lower = ts1 - timedelta(seconds=time_window_sec)
        time_upper = ts1 + timedelta(seconds=time_window_sec)

        potential_indices = combined_df[(combined_df['timestamp'] >= time_lower) &
                                        (combined_df['timestamp'] <= time_upper) &
                                        (combined_df.index > i)].index

        if len(potential_indices) == 0:
             processed_indices.add(i)
             pbar_dedup.update(1)
             continue

        for j in potential_indices:
            if j in processed_indices:
                continue

            lat2 = combined_df.iloc[j]['latitude']
            lon2 = combined_df.iloc[j]['longitude']
            distance = haversine(lat1, lon1, lat2, lon2)

            if pd.notna(distance) and distance <= dist_window_km:
                duplicates[j] = True
                processed_indices.add(j)

        processed_indices.add(i)
        pbar_dedup.update(1)

    pbar_dedup.close()

    final_df = combined_df[~duplicates].reset_index(drop=True)
    num_removed = len(combined_df) - len(final_df)
    print(f"...Tekilleştirme tamamlandı. {num_removed} kopya olay kaldırıldı.")
    print(f"Son katalog boyutu: {len(final_df)}")

    del combined_df, duplicates, processed_indices
    gc.collect()

    return final_df

def calculate_b_value(magnitudes, mc_method='MAXC', min_quakes=30, mag_bin_width=0.1):
    magnitudes = pd.Series(magnitudes).dropna().astype(np.float32)
    n_magnitudes = len(magnitudes)

    if n_magnitudes < min_quakes:
        return np.nan, np.nan

    try:
        min_mag = magnitudes.min()
        bins = np.arange(np.floor(min_mag*10)/10, magnitudes.max() + mag_bin_width, mag_bin_width)
        if len(bins) < 2:
             return np.nan, np.nan

        hist, edges = np.histogram(magnitudes, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2

        if np.all(hist == 0):
             return np.nan, np.nan

        mc = np.nan
        if mc_method == 'MAXC':
            idx = np.argmax(hist)
            if idx < len(centers):
                 mc = centers[idx]
            else:
                 mc = np.median(magnitudes)
        else:
            mc = centers[np.argmax(hist)] if len(centers) > np.argmax(hist) else np.median(magnitudes)

        mags_mc = magnitudes[magnitudes >= mc]
        n_mc = len(mags_mc)

        if n_mc < max(5, min_quakes / 5):
            return np.nan, mc

        mean_mc = mags_mc.mean()
        mc_corr = mc - (mag_bin_width / 2.0)

        if mean_mc <= mc_corr + 1e-6:
             return np.nan, mc

        b_val = (1.0 / (mean_mc - mc_corr)) * np.log10(np.exp(1))

        if not (0.4 < b_val < 2.5):
            return np.nan, mc

        return b_val, mc

    except Exception as e:
        return np.nan, np.nan

def generate_single_window_features_optimized(target_time, earthquake_df, lat_bins, lon_bins, feature_window_days, b_value_min_quakes, b_value_mc_method, feature_suffix, precomputed_bins=None):
    feature_start = target_time - timedelta(days=feature_window_days)
    feature_end = target_time

    relevant_quakes = earthquake_df[(earthquake_df['timestamp'] >= feature_start) &
                                    (earthquake_df['timestamp'] < feature_end)].copy()

    n_lat = len(lat_bins) - 1
    n_lon = len(lon_bins) - 1
    full_index = pd.MultiIndex.from_product([range(n_lat), range(n_lon)], names=['lat_bin', 'lon_bin'])

    current_feature_names = [
        f"Deprem Sayısı_{feature_suffix}",
        f"Ort Büyüklük_{feature_suffix}",
        f"Maks Büyüklük_{feature_suffix}",
        f"Enerji Vekili_{feature_suffix}",
        f"Ort Zaman Farkı (sn)_{feature_suffix}",
        f"Std Zaman Farkı (sn)_{feature_suffix}",
        f"b-değeri_{feature_suffix}",
        f"Mc (Tamamlama Mag)_{feature_suffix}",
        f"Ort Derinlik_{feature_suffix}",
    ]
    cell_features = pd.DataFrame(0.0, index=full_index, columns=current_feature_names, dtype=np.float32)
    nan_cols = [
        f"Ort Zaman Farkı (sn)_{feature_suffix}",
        f"Std Zaman Farkı (sn)_{feature_suffix}",
        f"b-değeri_{feature_suffix}",
        f"Mc (Tamamlama Mag)_{feature_suffix}",
        f"Ort Derinlik_{feature_suffix}"
    ]
    for col in nan_cols:
        cell_features[col] = np.nan

    if relevant_quakes.empty:
        return cell_features

    if precomputed_bins is None:
        lat_labels = np.arange(n_lat)
        lon_labels = np.arange(n_lon)
        relevant_quakes['lat_bin'] = pd.cut(relevant_quakes['latitude'], bins=lat_bins, labels=lat_labels, right=False, include_lowest=True)
        relevant_quakes['lon_bin'] = pd.cut(relevant_quakes['longitude'], bins=lon_bins, labels=lon_labels, right=False, include_lowest=True)
    else:
        relevant_quakes = relevant_quakes.join(precomputed_bins, on=relevant_quakes.index)

    relevant_quakes = relevant_quakes.dropna(subset=['lat_bin', 'lon_bin'])
    if relevant_quakes.empty:
        return cell_features

    relevant_quakes['lat_bin'] = relevant_quakes['lat_bin'].astype(int)
    relevant_quakes['lon_bin'] = relevant_quakes['lon_bin'].astype(int)

    grouped = relevant_quakes.groupby(['lat_bin', 'lon_bin'], observed=True)

    agg_funcs = {
        'magnitude': ['mean', 'max'],
        'depth': 'mean',
        'timestamp': 'count'
    }
    basic_stats = grouped.agg(agg_funcs)
    basic_stats.columns = ['Ort Büyüklük', 'Maks Büyüklük', 'Ort Derinlik', 'Deprem Sayısı']

    cell_features.loc[basic_stats.index, f'Deprem Sayısı_{feature_suffix}'] = basic_stats['Deprem Sayısı']
    cell_features.loc[basic_stats.index, f'Ort Büyüklük_{feature_suffix}'] = basic_stats['Ort Büyüklük']
    cell_features.loc[basic_stats.index, f'Maks Büyüklük_{feature_suffix}'] = basic_stats['Maks Büyüklük']
    cell_features.loc[basic_stats.index, f'Ort Derinlik_{feature_suffix}'] = basic_stats['Ort Derinlik']

    clipped_mag = np.clip(relevant_quakes['magnitude'].astype(np.float64), -2, 9)
    relevant_quakes['energy_term'] = np.power(10.0, 1.5 * clipped_mag)
    energy_proxy = relevant_quakes.groupby(['lat_bin', 'lon_bin'], observed=True)['energy_term'].sum()
    cell_features.loc[energy_proxy.index, f'Enerji Vekili_{feature_suffix}'] = energy_proxy

    def calc_time_stats(x_ts):
        if len(x_ts) < 2:
            return np.nan, np.nan
        diffs = x_ts.sort_values().diff().dt.total_seconds().dropna()
        if diffs.empty:
             return np.nan, np.nan
        mean_diff = diffs.mean()
        std_diff = diffs.std() if len(diffs) > 1 else 0.0
        return mean_diff, std_diff

    time_stats_results = grouped['timestamp'].apply(calc_time_stats)
    if not time_stats_results.empty:
        cell_features.loc[time_stats_results.index, f'Ort Zaman Farkı (sn)_{feature_suffix}'] = time_stats_results.apply(lambda x: x[0])
        cell_features.loc[time_stats_results.index, f'Std Zaman Farkı (sn)_{feature_suffix}'] = time_stats_results.apply(lambda x: x[1])

    counts = cell_features[f'Deprem Sayısı_{feature_suffix}']
    cells_for_bvalue = counts[counts >= b_value_min_quakes].index

    if not cells_for_bvalue.empty:
        b_mc_values = relevant_quakes[
            relevant_quakes.set_index(['lat_bin', 'lon_bin']).index.isin(cells_for_bvalue)
        ].groupby(['lat_bin', 'lon_bin'], observed=True)['magnitude'].apply(
            lambda x: calculate_b_value(x, mc_method=b_value_mc_method, min_quakes=b_value_min_quakes)
        )

        if not b_mc_values.empty:
            cell_features.loc[b_mc_values.index, f'b-değeri_{feature_suffix}'] = b_mc_values.apply(lambda x: x[0])
            cell_features.loc[b_mc_values.index, f'Mc (Tamamlama Mag)_{feature_suffix}'] = b_mc_values.apply(lambda x: x[1])

    cell_features[f'Deprem Sayısı_{feature_suffix}'].fillna(0, inplace=True)
    cell_features[f'Ort Büyüklük_{feature_suffix}'].fillna(0, inplace=True)
    cell_features[f'Maks Büyüklük_{feature_suffix}'].fillna(0, inplace=True)
    cell_features[f'Enerji Vekili_{feature_suffix}'].fillna(0, inplace=True)

    del relevant_quakes, grouped, basic_stats, energy_proxy, time_stats_results
    if 'b_mc_values' in locals(): del b_mc_values
    gc.collect()

    return cell_features

def add_neighbor_features_fast(features_df, feature_suffixes, n_lat, n_lon):
    if not SCIPY_AVAILABLE:
        print("Scipy kütüphanesi bulunamadı, hızlı komşu özelliği hesaplaması atlanıyor.")
        return features_df

    print("Komşu hücre özellikleri ekleniyor (Hızlı Yöntem - Scipy)...")
    new_neighbor_features_all = {}

    footprint = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=bool)

    base_features = ["Deprem Sayısı", "Ort Büyüklük", "Maks Büyüklük", "Enerji Vekili", "b-değeri"]

    start_time = time.time()
    processed_suffixes = 0
    for suffix in feature_suffixes:
        print(f"  Suffix işleniyor: {suffix}")
        neighbor_agg_suffix = pd.DataFrame(index=features_df.index, dtype=np.float32)
        features_exist_for_suffix = False

        for base_feat in base_features:
            feat_name = f"{base_feat}_{suffix}"
            if feat_name not in features_df.columns:
                continue
            features_exist_for_suffix = True

            grid_data = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
            valid_idx = features_df.index
            values = features_df[feat_name].values

            rows = valid_idx.get_level_values(0).to_numpy()
            cols = valid_idx.get_level_values(1).to_numpy()

            grid_data[rows, cols] = values

            valid_mask = ~np.isnan(grid_data)
            count_grid = generic_filter(valid_mask.astype(np.float32), np.sum, footprint=footprint, mode='constant', cval=0.0)

            sum_grid = generic_filter(np.nan_to_num(grid_data), np.sum, footprint=footprint, mode='constant', cval=0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_grid = sum_grid / count_grid
            mean_grid[count_grid == 0] = 0.0

            def safe_nanmax(arr):
                 return np.nanmax(arr) if not np.all(np.isnan(arr)) else np.nan

            max_grid = generic_filter(grid_data, safe_nanmax, footprint=footprint, mode='constant', cval=np.nan)
            max_grid = np.nan_to_num(max_grid, nan=0.0)

            mean_col_name = f"Komşu_Ort_{base_feat}_{suffix}"
            max_col_name = f"Komşu_Maks_{base_feat}_{suffix}"
            count_col_name = f"Komşu_Sayısı_{base_feat}_{suffix}"

            neighbor_agg_suffix[mean_col_name] = mean_grid[rows, cols]
            neighbor_agg_suffix[max_col_name] = max_grid[rows, cols]
            neighbor_agg_suffix[count_col_name] = count_grid[rows, cols]

        if features_exist_for_suffix:
             new_neighbor_features_all[suffix] = neighbor_agg_suffix
             processed_suffixes += 1

    if not new_neighbor_features_all:
        print("...Hiçbir suffix için komşu özelliği hesaplanamadı.")
        return features_df

    combined_features_df = pd.concat([features_df] + list(new_neighbor_features_all.values()), axis=1)

    end_time = time.time()
    print(f"...Komşu özellikleri eklendi ({processed_suffixes} suffix işlendi, {len(combined_features_df.columns) - len(features_df.columns)} yeni sütun, {end_time - start_time:.2f} saniye).")

    return combined_features_df.astype(np.float32)

def create_feature_target_matrix(earthquake_df, lat_bins, lon_bins, config):
    print("\nZaman adımlı özellik ve hedef matrisi oluşturuluyor...")
    all_step_features = []
    all_step_targets = []
    all_step_timestamps = []
    all_step_indices = []

    min_time = earthquake_df['timestamp'].min()
    max_time = earthquake_df['timestamp'].max()

    max_feature_window = max(config["FEATURE_WINDOWS_DAYS"]) if config["FEATURE_WINDOWS_DAYS"] else 0
    start_loop_time = min_time + timedelta(days=max_feature_window)

    end_loop_time = max_time - timedelta(days=config["PREDICTION_WINDOW_DAYS"])

    if start_loop_time >= end_loop_time:
        print(f"Uyarı: Veri aralığı ({min_time.date()} - {max_time.date()}) özellik ve hedef pencereleri için yetersiz.")
        print(f"Başlangıç Zamanı: {start_loop_time.date()}, Bitiş Zamanı: {end_loop_time.date()}")
        return None, None, None, None, []

    n_lat = len(lat_bins) - 1
    n_lon = len(lon_bins) - 1

    print("Hücre atamaları önceden hesaplanıyor...")
    lat_labels = np.arange(n_lat)
    lon_labels = np.arange(n_lon)
    precomputed_bins = pd.DataFrame(index=earthquake_df.index)
    precomputed_bins['lat_bin'] = pd.cut(earthquake_df['latitude'], bins=lat_bins, labels=lat_labels, right=False, include_lowest=True)
    precomputed_bins['lon_bin'] = pd.cut(earthquake_df['longitude'], bins=lon_bins, labels=lon_labels, right=False, include_lowest=True)
    print("...Hücre atamaları tamamlandı.")

    current_time = start_loop_time
    time_step = timedelta(days=config["TIME_STEP_DAYS"])

    total_steps = 0
    if time_step.total_seconds() > 0 and end_loop_time >= start_loop_time:
         total_steps = int((end_loop_time - start_loop_time).total_seconds() / time_step.total_seconds()) + 1
    elif end_loop_time >= start_loop_time:
         total_steps = 1

    print(f"Tahmini toplam adım sayısı: {total_steps}")
    pbar = tqdm(total=total_steps, desc="Özellik/Hedef Hesaplama", mininterval=1.0)
    step_count = 0
    all_feature_names = []

    while current_time <= end_loop_time:
        step_count += 1
        step_start_time = time.time()

        window_features_list = []
        current_step_feature_names = []
        for window_days in config["FEATURE_WINDOWS_DAYS"]:
            window_suffix = f"{window_days}d"
            features_df_window = generate_single_window_features_optimized(
                current_time, earthquake_df, lat_bins, lon_bins, window_days,
                config["B_VALUE_MIN_QUAKES"], config["B_VALUE_MC_METHOD"], window_suffix,
                precomputed_bins=precomputed_bins
            )
            window_features_list.append(features_df_window)
            current_step_feature_names.extend(features_df_window.columns.tolist())

        if not window_features_list:
            current_time += time_step
            pbar.update(1)
            continue

        features_df = pd.concat(window_features_list, axis=1)
        del window_features_list
        gc.collect()

        if config["ENABLE_NEIGHBOR_FEATURES"]:
            window_suffixes = [f"{d}d" for d in config["FEATURE_WINDOWS_DAYS"]]
            if config["USE_FAST_NEIGHBORS"]:
                features_df = add_neighbor_features_fast(features_df, window_suffixes, n_lat, n_lon)
            else:
                print(f"Uyarı: Adım {current_time.date()} - Hızlı komşu özellikleri kullanılamıyor/devre dışı.")
            current_step_feature_names = features_df.columns.tolist()

        if not all_feature_names:
            all_feature_names = features_df.columns.tolist()
            print(f"Toplam {len(all_feature_names)} özellik sütunu belirlendi.")

        prediction_start = current_time
        prediction_end = current_time + timedelta(days=config["PREDICTION_WINDOW_DAYS"])

        target_quakes = earthquake_df[
            (earthquake_df['timestamp'] > prediction_start) &
            (earthquake_df['timestamp'] <= prediction_end) &
            (earthquake_df['magnitude'] >= config["MAGNITUDE_THRESHOLD"])
        ].copy()

        targets = pd.Series(0, index=features_df.index, dtype=np.int8)

        if not target_quakes.empty:
            target_bins = precomputed_bins.loc[target_quakes.index].dropna(subset=['lat_bin', 'lon_bin'])
            if not target_bins.empty:
                target_bins['lat_bin'] = target_bins['lat_bin'].astype(int)
                target_bins['lon_bin'] = target_bins['lon_bin'].astype(int)
                target_indices = pd.MultiIndex.from_frame(target_bins[['lat_bin', 'lon_bin']])
                valid_target_indices = targets.index.intersection(target_indices)
                if not valid_target_indices.empty:
                    targets.loc[valid_target_indices] = 1

        features_filled = features_df.copy()
        fill_values = {}
        for col in features_filled.columns:
             if features_filled[col].isnull().any():
                 if 'Ort' in col or 'Maks' in col or 'Sayı' in col or 'Enerji' in col:
                     fill_values[col] = 0.0
                 else:
                     fill_values[col] = 0.0
        features_filled.fillna(fill_values, inplace=True)
        features_filled = features_filled.astype(np.float32)

        if not features_filled.empty:
            all_step_features.append(features_filled.values)
            all_step_targets.append(targets.values)
            step_indices_df = features_filled.reset_index()[['lat_bin', 'lon_bin']]
            all_step_indices.append(step_indices_df.values.astype(np.int16))
            all_step_timestamps.extend([current_time] * len(features_filled))

        pbar.update(1)
        step_duration = time.time() - step_start_time
        if step_count % 50 == 0 :
            pbar.set_description(f"Özellik/Hedef {current_time.date()} ({step_duration:.2f}s)")

        del features_df, targets, target_quakes, features_filled, step_indices_df
        if 'target_bins' in locals(): del target_bins
        if 'target_indices' in locals(): del target_indices
        if 'valid_target_indices' in locals(): del valid_target_indices
        if 'fill_values' in locals(): del fill_values
        gc.collect()

        current_time += time_step

    pbar.close()
    del precomputed_bins
    gc.collect()
    print("...Özellik ve hedef matrisi oluşturma tamamlandı.")

    if not all_step_features or not all_step_targets:
        print("\nUyarı: Hiç özellik veya hedef verisi oluşturulamadı.")
        return None, None, None, None, []

    print("Numpy dizileri birleştiriliyor...")
    try:
        if not all_step_features or not all_step_targets or not all_step_indices:
             raise ValueError("Özellik, hedef veya indeks listelerinden en az biri boş.")

        X = np.concatenate(all_step_features, axis=0).astype(np.float32)
        y = np.concatenate(all_step_targets, axis=0).astype(np.int8)
        timestamps = np.array(all_step_timestamps)
        cell_indices = np.concatenate(all_step_indices, axis=0).astype(np.int16)
    except ValueError as e:
        print(f"Hata: Numpy dizileri birleştirilemedi: {e}")
        return None, None, None, None, []

    del all_step_features, all_step_targets, all_step_timestamps, all_step_indices
    gc.collect()
    print("...Numpy dizilerini birleştirme tamamlandı.")

    print(f"\nToplam oluşturulan örnek sayısı: {len(X)}")
    if X.size > 0:
        print(f"Özellik sayısı: {X.shape[1]}")
    else:
        print("Hiç örnek oluşturulamadı.")
        return None, None, None, None, []

    target_counts = np.bincount(y)
    print(f"Hedef Dağılımı: Yok (0) = {target_counts[0] if len(target_counts)>0 else 0}, Var (1) = {target_counts[1] if len(target_counts)>1 else 0}")
    if len(target_counts) < 2 or target_counts[1] == 0:
        print("\nUyarı: Hedef sınıfta (1) hiç örnek bulunamadı! Model eğitimi anlamlı olmayabilir.")

    if not all_feature_names and X.shape[1] > 0:
         all_feature_names = [f'f{i}' for i in range(X.shape[1])]
         print("Uyarı: Özellik isimleri alınamadı, F0, F1... olarak atanıyor.")
    elif X.shape[1] != len(all_feature_names):
         print(f"Uyarı: Matris sütun sayısı ({X.shape[1]}) ile özellik ismi sayısı ({len(all_feature_names)}) eşleşmiyor! Düzeltiliyor.")
         if X.shape[1] < len(all_feature_names):
             all_feature_names = all_feature_names[:X.shape[1]]
         else:
             all_feature_names.extend([f'f{i}' for i in range(len(all_feature_names), X.shape[1])])

    return X, y, timestamps, cell_indices, all_feature_names

def calculate_pr_auc(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

def tune_hyperparameters_optuna(X_train_val, y_train_val, feature_names_list, config):
    if not config["ENABLE_OPTUNA"] or not OPTUNA_AVAILABLE:
        print("Optuna optimizasyonu devre dışı veya kütüphane bulunamadı.")
        return None

    print("\n--- Optuna ile Hiperparametre Optimizasyonu Başlatılıyor ---")

    data_fraction = max(0.1, min(1.0, config.get("OPTUNA_DATA_FRACTION", 1.0)))
    if data_fraction < 1.0:
        num_samples = len(X_train_val)
        start_index = int(num_samples * (1 - data_fraction))
        X_opt, y_opt = X_train_val[start_index:], y_train_val[start_index:]
        print(f"Optuna için verinin son %{int(data_fraction*100)}'lik kısmı kullanılıyor ({len(X_opt)} örnek).")
    else:
        X_opt, y_opt = X_train_val, y_train_val
        print(f"Optuna için tüm eğitim/validasyon verisi kullanılıyor ({len(X_opt)} örnek).")

    if len(X_opt) < 100:
         print("Uyarı: Optuna için yetersiz sayıda örnek.")
         return None

    tscv_optuna = TimeSeriesSplit(n_splits=config.get("OPTUNA_CV_SPLITS", 3))

    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'booster': 'gbtree',
            'tree_method': config['XGB_TREE_METHOD'],
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'lambda': trial.suggest_float('lambda', 1e-6, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-6, 1.0, log=True),
            'seed': 42,
            'nthread': -1
        }
        if config['XGB_TREE_METHOD'] == 'gpu_hist':
            params['gpu_id'] = 0

        pr_aucs = []
        try:
            for fold, (train_idx, val_idx) in enumerate(tscv_optuna.split(X_opt)):
                X_train, X_val = X_opt[train_idx], X_opt[val_idx]
                y_train, y_val = y_opt[train_idx], y_opt[val_idx]

                if len(X_val) == 0: continue

                pos_count = (y_train == 1).sum()
                neg_count = (y_train == 0).sum()
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
                params['scale_pos_weight'] = scale_pos_weight

                dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names_list)
                dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names_list)
                watchlist = [(dval, 'eval')]

                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=config['XGB_NUM_BOOST_ROUND'],
                    evals=watchlist,
                    early_stopping_rounds=config['XGB_EARLY_STOPPING_ROUNDS'],
                    verbose_eval=False
                )

                y_pred_proba = model.predict(dval, iteration_range=(0, model.best_iteration))

                if len(np.unique(y_val)) > 1:
                    pr_auc = calculate_pr_auc(y_val, y_pred_proba)
                    pr_aucs.append(pr_auc)
                else:
                    pr_aucs.append(0.0)

                del dtrain, dval, model, X_train, X_val, y_train, y_val
                gc.collect()

            return np.mean(pr_aucs) if pr_aucs else 0.0

        except Exception as e:
             print(f"Optuna denemesi {trial.number} sırasında hata: {e}")
             return 0.0

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    try:
        study.optimize(
            objective,
            n_trials=config['OPTUNA_N_TRIALS'],
            timeout=config.get('OPTUNA_TIMEOUT', None),
            gc_after_trial=True,
            n_jobs=1
        )
    except KeyboardInterrupt:
        print("Optuna kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"Optuna optimizasyonu sırasında genel hata: {e}")

    if not study.trials or study.best_trial is None:
        print("Optuna başarılı bir deneme tamamlayamadı.")
        return None

    print("\n--- Optuna Optimizasyonu Tamamlandı ---")
    try:
        print(f"Toplam deneme sayısı: {len(study.trials)}")
        print(f"En İyi Skor (Ortalama PR AUC): {study.best_value:.4f}")
        print("En İyi Hiperparametreler:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        return study.best_params
    except Exception as e:
        print(f"Optuna sonuçları alınırken hata: {e}")
        return None

def train_evaluate_cv(X, y, feature_names_list, config, best_params=None):
    print("\n--- Zaman Serisi Çapraz Doğrulama ile Eğitim ve Değerlendirme ---")
    tscv = TimeSeriesSplit(n_splits=config['N_SPLITS_CV'])
    metrics = []
    oof_predictions = np.zeros(len(y))
    oof_indices = []

    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'aucpr'],
        'seed': 42,
        'nthread': -1,
        'tree_method': config['XGB_TREE_METHOD']
    }
    if config['XGB_TREE_METHOD'] == 'gpu_hist':
        params['gpu_id'] = 0

    if best_params:
        print("Optuna tarafından bulunan en iyi parametreler kullanılıyor.")
        params.update(best_params)
    else:
        print("Varsayılan XGBoost parametreleri kullanılıyor.")
        params.update({
            'eta': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'min_child_weight': 3, 'gamma': 0.2, 'lambda': 0.5, 'alpha': 0.1
        })

    print("Çapraz doğrulama katmanları işleniyor...")
    pbar = tqdm(tscv.split(X), total=config['N_SPLITS_CV'], desc="CV Katmanları", mininterval=1.0)
    best_iteration_overall = config['XGB_NUM_BOOST_ROUND']

    for fold, (train_idx, test_idx) in enumerate(pbar):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(X_test) == 0:
            print(f"Uyarı: CV Katman {fold+1} atlandı (test seti boş).")
            continue

        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        params['scale_pos_weight'] = scale_pos_weight

        pbar.set_description(f"CV {fold+1}/{config['N_SPLITS_CV']} (Eğitim:{len(X_train)}, Test:{len(X_test)}, SPW:{scale_pos_weight:.1f})")

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names_list)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names_list)
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]

        try:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=config['XGB_NUM_BOOST_ROUND'],
                evals=watchlist,
                early_stopping_rounds=config['XGB_EARLY_STOPPING_ROUNDS'],
                verbose_eval=False
            )

            if fold == config['N_SPLITS_CV'] - 1:
                 best_iteration_overall = model.best_iteration

            y_pred_proba = model.predict(dtest, iteration_range=(0, model.best_iteration))
            y_pred_binary = (y_pred_proba > 0.5).astype(int)

            accuracy = (y_pred_binary == y_test).mean()
            precision, recall, f1_score, _ = precision_recall_fscore_support(
                y_test, y_pred_binary, average='binary', zero_division=0
            )
            roc_auc = 0.5
            pr_auc = 0.0

            if len(np.unique(y_test)) > 1:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                pr_auc = calculate_pr_auc(y_test, y_pred_proba)

            metrics.append({
                'fold': fold + 1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'best_iter': model.best_iteration
            })

            oof_predictions[test_idx] = y_pred_proba
            oof_indices.extend(test_idx.tolist())

            pbar.set_postfix(ROC=f"{roc_auc:.3f}", PR=f"{pr_auc:.3f}", F1=f"{f1_score:.3f}")

        except Exception as e:
            print(f"\nCV Katman {fold+1} sırasında hata: {e}")
            continue
        finally:
             if 'model' in locals(): del model
             del X_train, X_test, y_train, y_test, dtrain, dtest
             gc.collect()

    pbar.close()

    if not metrics:
        print("\nUyarı: Çapraz doğrulama tamamlanamadı, hiç metrik hesaplanmadı.")
        return None, params, config['XGB_NUM_BOOST_ROUND'], pd.DataFrame()

    metrics_df = pd.DataFrame(metrics)
    print("\n--- Çapraz Doğrulama Sonuçları ---")
    print(metrics_df.round(4))
    print("\n--- Ortalama Çapraz Doğrulama Metrikleri ---")
    print(metrics_df.mean().round(4))

    print("\n--- Out-of-Fold (OOF) Değerlendirmesi ---")
    valid_oof_indices = sorted(list(set(oof_indices)))
    if valid_oof_indices:
        y_true_oof = y[valid_oof_indices]
        y_pred_proba_oof = oof_predictions[valid_oof_indices]
        y_pred_binary_oof = (y_pred_proba_oof > 0.5).astype(int)

        if len(np.unique(y_true_oof)) > 1:
            print("OOF Sınıflandırma Raporu (Eşik=0.5):")
            print(classification_report(y_true_oof, y_pred_binary_oof, zero_division=0))
            oof_roc_auc = roc_auc_score(y_true_oof, y_pred_proba_oof)
            oof_pr_auc = calculate_pr_auc(y_true_oof, y_pred_proba_oof)
            print(f"OOF ROC AUC: {oof_roc_auc:.4f}")
            print(f"OOF PR AUC: {oof_pr_auc:.4f}")
        else:
            print("OOF tahminlerinde yalnızca bir sınıf bulundu, detaylı rapor oluşturulamıyor.")
    else:
        print("Hiç OOF tahmini yapılamadı.")

    print(f"\nFinal model için kullanılacak tahmini en iyi iterasyon sayısı (son katmandan): {best_iteration_overall}")

    return None, params, best_iteration_overall, metrics_df

def train_final_model(X, y, feature_names_list, params, num_boost_round, config):
    print("\n--- Nihai Modelin Eğitimi (Tüm Veri Üzerinde) ---")
    if X is None or len(X) == 0:
        print("Nihai model eğitimi için veri bulunamadı.")
        return None

    print(f"Eğitim seti boyutu: {len(X)} örnek")
    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    params['scale_pos_weight'] = scale_pos_weight
    print(f"Nihai model için Scale Pos Weight: {scale_pos_weight:.2f}")

    if len(feature_names_list) != X.shape[1]:
         print(f"Hata: Nihai modelde özellik sayısı ({X.shape[1]}) ile isim listesi ({len(feature_names_list)}) uyuşmuyor!")
         feature_names_list = [f'f{i}' for i in range(X.shape[1])]

    dtrain_final = xgb.DMatrix(X, label=y, feature_names=feature_names_list)
    watchlist_final = [(dtrain_final, 'train')]

    final_boost_rounds = max(50, num_boost_round + 10)
    print(f"Nihai model {final_boost_rounds} tur eğitilecek (CV en iyi tur: {num_boost_round})...")

    try:
        final_model = xgb.train(
            params,
            dtrain_final,
            num_boost_round=final_boost_rounds,
            evals=watchlist_final,
            verbose_eval=50
        )
        print("...Nihai model eğitimi tamamlandı.")

        print("\n--- Özellik Önem Sıralaması (Gain) ---")
        try:
            importance = final_model.get_score(importance_type='gain')
            if importance:
                sorted_importance = sorted(importance.items(), key=lambda item: item[1], reverse=True)
                for i, (feature, score) in enumerate(sorted_importance):
                    if i < 20:
                        print(f"- {feature}: {score:.3f}")
                    elif i == 20:
                        print(f"...ve {len(sorted_importance) - 20} diğer özellik.")
                        break
            else:
                 print("Özellik önem skorları alınamadı.")
        except Exception as imp_e:
            print(f"Özellik önemi alınırken hata: {imp_e}")

        return final_model

    except Exception as e:
        print(f"Nihai model eğitimi sırasında hata: {e}")
        return None
    finally:
        del dtrain_final
        gc.collect()

def generate_future_predictions(final_model, earthquake_df, lat_bins, lon_bins, feature_names_list, config):
    print("\n--- GELECEK TAHMİNİ OLUŞTURULUYOR ---")
    if final_model is None:
        print("Hata: Eğitilmiş nihai model bulunamadı.")
        return None

    prediction_time = config["END_TIME"]
    print(f"Tahmin için özellikler '{prediction_time.strftime('%Y-%m-%d %H:%M:%S %Z')}' itibarıyla hesaplanıyor...")

    n_lat = len(lat_bins) - 1
    n_lon = len(lon_bins) - 1

    print("Hücre atamaları hesaplanıyor (tahmin için)...")
    lat_labels_fut = np.arange(n_lat)
    lon_labels_fut = np.arange(n_lon)
    precomputed_bins_future = pd.DataFrame(index=earthquake_df.index)
    precomputed_bins_future['lat_bin'] = pd.cut(earthquake_df['latitude'], bins=lat_bins, labels=lat_labels_fut, right=False, include_lowest=True)
    precomputed_bins_future['lon_bin'] = pd.cut(earthquake_df['longitude'], bins=lon_bins, labels=lon_labels_fut, right=False, include_lowest=True)
    print("...Hücre atamaları tamamlandı (tahmin için).")

    future_features_list = []
    current_feature_names_future = []
    for window_days in config["FEATURE_WINDOWS_DAYS"]:
        window_suffix = f"{window_days}d"
        features_df_window_fut = generate_single_window_features_optimized(
            prediction_time, earthquake_df, lat_bins, lon_bins, window_days,
            config["B_VALUE_MIN_QUAKES"], config["B_VALUE_MC_METHOD"], window_suffix,
            precomputed_bins=precomputed_bins_future
        )
        future_features_list.append(features_df_window_fut)
        current_feature_names_future.extend(features_df_window_fut.columns.tolist())

    if not future_features_list:
        print("Gelecek tahmini için özellik hesaplanamadı.")
        return None

    now_features_df = pd.concat(future_features_list, axis=1)
    del future_features_list, precomputed_bins_future
    gc.collect()

    if config["ENABLE_NEIGHBOR_FEATURES"]:
        window_suffixes_fut = [f"{d}d" for d in config["FEATURE_WINDOWS_DAYS"]]
        if config["USE_FAST_NEIGHBORS"]:
            now_features_df = add_neighbor_features_fast(now_features_df, window_suffixes_fut, n_lat, n_lon)
        else:
            print("Uyarı: Tahmin için hızlı komşu özellikleri kullanılamıyor/devre dışı.")
        current_feature_names_future = now_features_df.columns.tolist()

    model_features = final_model.feature_names
    current_features_set = set(current_feature_names_future)
    model_features_set = set(model_features)

    if current_features_set != model_features_set:
        print("\n!!! UYARI: Tahmin için hesaplanan özellikler ile modelin eğitildiği özellikler farklı !!!")
        common_features = list(model_features_set.intersection(current_features_set))
        missing_in_current = list(model_features_set - current_features_set)
        extra_in_current = list(current_features_set - model_features_set)

        print(f"Modelin Beklediği ({len(model_features)}): ...{model_features[-5:]}")
        print(f"Şu An Hesaplanan ({len(current_features_set)}): ...{current_feature_names_future[-5:]}")
        if missing_in_current: print(f"Tahminde Eksik Olanlar ({len(missing_in_current)}): {missing_in_current[:10]}...")
        if extra_in_current: print(f"Tahminde Fazla Olanlar ({len(extra_in_current)}): {extra_in_current[:10]}...")

        if not common_features:
             print("Hata: Model ve tahmin arasında hiç ortak özellik bulunamadı! Tahmin yapılamaz.")
             return None
        elif len(common_features) < len(model_features) * 0.8:
             print("Uyarı: Ortak özellik sayısı beklenenden çok düşük. Tahminler güvenilir olmayabilir.")

        print(f"Sadece ortak olan {len(common_features)} özellik kullanılacak.")
        features_to_predict_with = common_features
        now_features_df = now_features_df[features_to_predict_with]
    else:
         features_to_predict_with = model_features
         now_features_df = now_features_df[features_to_predict_with]

    now_features_filled = now_features_df.copy()
    fill_values_future = {}
    for col in features_to_predict_with:
         if now_features_filled[col].isnull().any():
             if 'Ort' in col or 'Maks' in col or 'Sayı' in col or 'Enerji' in col:
                 fill_values_future[col] = 0.0
             else:
                 fill_values_future[col] = 0.0
    now_features_filled.fillna(fill_values_future, inplace=True)
    now_features_filled = now_features_filled.astype(np.float32)

    if now_features_filled.empty:
        print("Gelecek tahmini için doldurulmuş özellik verisi boş.")
        return None

    print(f"Tahmin için {len(now_features_filled)} hücrenin özellikleri hazırlandı.")

    X_now = now_features_filled.values
    dnow = xgb.DMatrix(X_now, feature_names=features_to_predict_with)

    print("Model ile olasılıklar tahmin ediliyor...")
    try:
        best_iteration = getattr(final_model, 'best_iteration', -1)
        if best_iteration == -1:
            best_iteration = final_model.num_boosted_rounds()

        probabilities = final_model.predict(dnow, iteration_range=(0, best_iteration))
        print("...Tahmin işlemi tamamlandı.")

    except Exception as e:
        print(f"Hata: Gelecek tahminleri yapılırken: {e}")
        print("Modelin beklediği özellikler:", final_model.feature_names)
        print("Tahmin için sağlanan özellikler:", features_to_predict_with)
        if np.isnan(X_now).any():
             nan_cols = now_features_filled.columns[np.isnan(X_now).any(axis=0)].tolist()
             print(f"Uyarı: Tahmin matrisinde (X_now) NaN değerler bulundu! Sütunlar: {nan_cols}")
        return None

    predictions_df = now_features_filled.reset_index()
    predictions_df['probability'] = probabilities.astype(np.float32)

    return predictions_df[['lat_bin', 'lon_bin', 'probability']]

def get_region_name(lat, lon, geolocator, reverse_func):
    global GEOPY_AVAILABLE
    if not GEOPY_AVAILABLE or geolocator is None or reverse_func is None:
        return "Geocoder Kullanılamıyor"
    try:
        location = reverse_func(f"{lat:.6f}, {lon:.6f}", language='tr', addressdetails=True, timeout=10)
        if location and location.raw and 'address' in location.raw:
            address = location.raw.get('address', {})
            parts = [
                address.get(k) for k in ['village', 'town', 'suburb', 'city_district', 'county', 'state']
                if address.get(k)
            ]
            name = ", ".join(filter(None, parts))
            if not name:
                 name = location.address
            return name if name else "Bölge Adı Bulunamadı"
        else:
            return "Bölge Bilgisi Alınamadı"
    except Exception as e:
        return "Geocoder Hatası"

def create_prediction_map(predictions_df, lat_centers, lon_centers, lat_bins, lon_bins, config):
    global GEOPY_AVAILABLE
    if not FOLIUM_AVAILABLE:
        print("Folium kütüphanesi bulunamadı, harita oluşturma atlanıyor.")
        return

    print(f"\n'{config['PREDICTION_MAP_FILE']}' haritası oluşturuluyor...")

    map_threshold = config["FUTURE_PREDICTION_PROB_THRESHOLD"]
    map_cells = predictions_df[predictions_df['probability'] >= map_threshold].copy()

    if map_cells.empty:
        print(f"Harita için ana olasılık eşiğini ({map_threshold*100:.1f}%) geçen hücre bulunamadı.")
        return

    print(f"Harita için ana olasılık eşiğini ({map_threshold*100:.1f}%) geçen {len(map_cells)} hücre bulundu.")

    map_cells['lat_center'] = map_cells['lat_bin'].apply(lambda i: lat_centers[int(i)] if 0 <= int(i) < len(lat_centers) else np.nan)
    map_cells['lon_center'] = map_cells['lon_bin'].apply(lambda i: lon_centers[int(i)] if 0 <= int(i) < len(lon_centers) else np.nan)
    map_cells = map_cells.dropna(subset=['lat_center', 'lon_center'])

    if map_cells.empty:
        print("Harita için geçerli koordinatlara sahip hücre kalmadı.")
        return

    geolocator = None
    reverse_func = None
    if GEOPY_AVAILABLE:
        try:
            geolocator = Nominatim(user_agent=f"eq_pred_{config['OUTPUT_TIMESTAMP_STR']}")
            reverse_func = RateLimiter(geolocator.reverse, min_delay_seconds=1.1, return_value_on_exception=None)
            print("Geocoder başlatıldı (harita için).")
        except Exception as e:
            print(f"Uyarı: Geocoder başlatılamadı: {e}. Bölge isimleri alınamayabilir.")
            GEOPY_AVAILABLE = False

    print("Bölge isimleri alınıyor (sadece harita hücreleri için)...")
    map_cells['region_name'] = "İşleniyor..."
    geo_pbar = tqdm(map_cells.iterrows(), total=len(map_cells), desc="Bölge İsimleri (Harita)", mininterval=1.0)
    for index, row in geo_pbar:
        map_cells.loc[index, 'region_name'] = get_region_name(row['lat_center'], row['lon_center'], geolocator, reverse_func)
    geo_pbar.close()
    print("...Bölge isimleri alma tamamlandı (harita).")

    try:
        center_cell = map_cells.loc[map_cells['probability'].idxmax()]
        map_center_lat = center_cell['lat_center']
        map_center_lon = center_cell['lon_center']

        m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=6, tiles='CartoDB positron')

        min_prob_map = map_cells['probability'].min()
        max_prob_map = map_cells['probability'].max()

        vmin_color = map_threshold
        if max_prob_map <= vmin_color * 1.05:
             max_prob_map = vmin_color * 1.05

        colormap = linear.YlOrRd_09.scale(vmin=vmin_color, vmax=max_prob_map)
        colormap.caption = f'M{config["MAGNITUDE_THRESHOLD"]}+ Olasılığı (%) [{vmin_color*100:.0f}-{max_prob_map*100:.0f}] - Sonraki {config["PREDICTION_WINDOW_DAYS"]} Gün ({ISTANBUL_TZ.zone})'

        print("Haritaya hücreler ekleniyor...")
        map_cells_pbar = tqdm(map_cells.iterrows(), total=len(map_cells), desc="Harita Hücreleri", leave=False, mininterval=1.0)
        for _, row in map_cells_pbar:
            lat_idx = int(row['lat_bin'])
            lon_idx = int(row['lon_bin'])
            probability = row['probability']
            region = row['region_name']
            lat_center = row['lat_center']
            lon_center = row['lon_center']

            if 0 <= lat_idx < len(lat_bins) - 1 and 0 <= lon_idx < len(lon_bins) - 1:
                bounds = [
                    [lat_bins[lat_idx], lon_bins[lon_idx]],
                    [lat_bins[lat_idx+1], lon_bins[lon_idx+1]]
                ]
            else:
                 continue

            popup_html = f"""
            <b>Koordinat:</b> {lat_center:.3f}, {lon_center:.3f}<br>
            <b>Tahmini Bölge:</b> {region}<br>
            <b>Olasılık: {probability*100:.1f}%</b><br>
            <a href='https://www.google.com/maps?q={lat_center:.4f},{lon_center:.4f}' target='_blank'>Google Haritalar'da Aç</a>
            """
            iframe = folium.IFrame(html=popup_html, width=300, height=100)
            popup = folium.Popup(iframe, max_width=300)

            tooltip_text = f"{probability*100:.1f}% - {region[:30]}{'...' if len(region)>30 else ''}"

            folium.Rectangle(
                bounds=bounds,
                popup=popup,
                tooltip=tooltip_text,
                color='#333333',
                weight=0.5,
                fill=True,
                fillColor=colormap(probability),
                fillOpacity=0.65
            ).add_to(m)

        colormap.add_to(m)

        m.save(config["PREDICTION_MAP_FILE"])
        print(f"Harita başarıyla '{config['PREDICTION_MAP_FILE']}' olarak kaydedildi.")

    except Exception as map_e:
        print(f"Hata: Harita oluşturulurken bir sorun oluştu: {map_e}")

def save_predictions_to_file(predictions_df, lat_centers, lon_centers, config):
    global GEOPY_AVAILABLE # Explicitly declare as global
    text_cells = predictions_df[predictions_df['probability'] >= config["FUTURE_PREDICTION_PROB_THRESHOLD"]].copy()

    if text_cells.empty:
        print(f"\nMetin dosyasına yazılacak kadar yüksek olasılıklı ({config['FUTURE_PREDICTION_PROB_THRESHOLD']*100:.1f}%) hücre bulunamadı.")
        return

    print(f"\nYüksek olasılıklı {len(text_cells)} tahmin '{config['PREDICTION_OUTPUT_FILE']}' dosyasına yazılıyor...")

    if 'lat_center' not in text_cells.columns:
        text_cells['lat_center'] = text_cells['lat_bin'].apply(lambda i: lat_centers[int(i)] if 0 <= int(i) < len(lat_centers) else np.nan)
        text_cells['lon_center'] = text_cells['lon_bin'].apply(lambda i: lon_centers[int(i)] if 0 <= int(i) < len(lon_centers) else np.nan)
        text_cells.rename(columns={'lat_center': 'latitude', 'lon_center': 'longitude'}, inplace=True)
    else:
         text_cells.rename(columns={'lat_center': 'latitude', 'lon_center': 'longitude'}, inplace=True)

    text_cells = text_cells.dropna(subset=['latitude', 'longitude'])
    if text_cells.empty:
        print("Metin dosyası için geçerli koordinatlara sahip hücre kalmadı.")
        return

    if 'region_name' not in text_cells.columns:
        geolocator_txt = None
        reverse_func_txt = None
        if GEOPY_AVAILABLE: # Check the global variable
            try:
                geolocator_txt = Nominatim(user_agent=f"eq_pred_{config['OUTPUT_TIMESTAMP_STR']}_txt")
                reverse_func_txt = RateLimiter(geolocator_txt.reverse, min_delay_seconds=1.1, return_value_on_exception=None)
                print("Geocoder başlatıldı (metin dosyası için).")
            except Exception as e:
                print(f"Uyarı: Geocoder başlatılamadı (metin): {e}.")
                GEOPY_AVAILABLE = False

        print("Bölge isimleri alınıyor (metin dosyası)...")
        text_cells['region_name'] = "İşleniyor..."
        geo_pbar_text = tqdm(text_cells.iterrows(), total=len(text_cells), desc="Bölge İsimleri (Text)", mininterval=1.0)
        for index, row in geo_pbar_text:
            text_cells.loc[index, 'region_name'] = get_region_name(row['latitude'], row['longitude'], geolocator_txt, reverse_func_txt)
        geo_pbar_text.close()
        print("...Bölge isimleri alma tamamlandı (metin dosyası).")
    else:
        print("Bölge isimleri harita oluşturma adımından alınıyor (metin dosyası için).")
        if 'region_name' not in text_cells.columns or text_cells['region_name'].isnull().all():
             text_cells['region_name'] = "Bölge Adı Alınamadı"

    text_cells['gmaps_link'] = text_cells.apply(
        lambda row: f"https://www.google.com/maps?q={row['latitude']:.4f},{row['longitude']:.4f}",
        axis=1
    )

    text_cells = text_cells.sort_values('probability', ascending=False)

    try:
        with open(config["PREDICTION_OUTPUT_FILE"], 'w', encoding='utf-8') as f:
            f.write(f"# GELECEK {config['PREDICTION_WINDOW_DAYS']} GÜN İÇİN M{config['MAGNITUDE_THRESHOLD']}+ DEPREM OLASILIKLARI\n")
            f.write(f"# Tahmin Oluşturma Tarihi: {config['END_TIME'].strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
            f.write(f"# Veri Yılı: {config['DATA_YEARS']}, Grid Çözünürlüğü: {config['GRID_RES_LAT']}° x {config['GRID_RES_LON']}°\n")
            f.write(f"# Özellik Pencere(ler)i: {config['FEATURE_WINDOWS_DAYS']} gün, Zaman Adımı: {config['TIME_STEP_DAYS']} gün\n")
            neighbor_status = "Devre Dışı"
            if config['ENABLE_NEIGHBOR_FEATURES']:
                neighbor_status = "Etkin (Hızlı)" if config['USE_FAST_NEIGHBORS'] else "Etkin (Yavaş/Scipy Yok)"
            f.write(f"# Komşu Özellikleri: {neighbor_status}\n")
            optuna_status = "Etkin" if config['ENABLE_OPTUNA'] else "Devre Dışı"
            f.write(f"# Optuna Hiperparametre Optimizasyonu: {optuna_status}\n")
            f.write(f"# Gösterilen Olasılık Eşiği (Bu dosya): {config['FUTURE_PREDICTION_PROB_THRESHOLD']*100:.1f}%\n")
            f.write("# " + "-"*77 + "\n")
            f.write("# !!! UYARI: BU SONUÇLAR DENEYSELDİR VE BİLİMSEL KESİNLİK TAŞIMAZ !!!\n")
            f.write("# !!! GERÇEK ZAMANLI BİR TAHMİN SİSTEMİ DEĞİLDİR VE HAYATİ KARARLAR İÇİN KULLANILMAMALIDIR !!!\n")
            f.write("# !!! LÜTFEN RESMİ KURUMLARIN (AFAD, KOERI, USGS, EMSC VB.) AÇIKLAMALARINI TAKİP EDİNİZ !!!\n")
            f.write("# " + "-"*77 + "\n\n")

            header_line = "-" * 130
            header = f"{'Enlem':<10} {'Boylam':<10} {'Olasılık (%)':<15} {'Tahmini Bölge (En Yakın)':<70} {'Google Haritalar Linki'}"
            f.write(header_line + "\n")
            f.write(header + "\n")
            f.write(header_line + "\n")

            for _, p_row in text_cells.iterrows():
                region_str = str(p_row.get('region_name', 'Bilinmiyor'))[:68]
                f.write(f"{p_row['latitude']:<10.3f} {p_row['longitude']:<10.3f} {p_row['probability']*100:<15.1f} {region_str:<70} {p_row['gmaps_link']}\n")

            f.write(header_line + "\n")

        print(f"Tahminler başarıyla '{config['PREDICTION_OUTPUT_FILE']}' dosyasına yazıldı.")

    except IOError as e:
        print(f"Hata: Tahmin dosyasına yazılamadı: {e}")
    except Exception as e:
        print(f"Hata: Tahmin dosyasına yazma sırasında beklenmedik bir hata oluştu: {e}")

def main():
    print("\n" + "="*60)
    print("--- DEPREM OLASILIK TAHMİN MODELİ (Optimize v3 - Harita Ayarlı) ---")
    print(f"--- Çalışma Zamanı: {datetime.now(ISTANBUL_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    print("="*60)
    start_time_script = time.time()

    print("\n--- Gerekli Kütüphaneler ---")
    print(f"Scipy (Hızlı Komşu): {'VAR' if SCIPY_AVAILABLE else 'YOK (Performans etkilenebilir)'}")
    print(f"Geopy (Bölge İsimleri): {'VAR' if GEOPY_AVAILABLE else 'YOK (Bölge isimleri alınamayacak)'}")
    print(f"Folium/Branca (Harita): {'VAR' if FOLIUM_AVAILABLE else 'YOK (Harita çıktısı oluşturulmayacak)'}")
    print(f"Optuna (Optimizasyon): {'VAR' if OPTUNA_AVAILABLE else 'YOK (Hiperparametre optimizasyonu yapılamayacak)'}")
    print(f"Pytz (Saat Dilimi): {'VAR' if 'pytz' in globals() else 'YOK (UTC kullanılacak)'}")
    print("-" * 30)

    lat_bins = np.arange(CONFIG["MIN_LATITUDE"], CONFIG["MAX_LATITUDE"] + CONFIG["GRID_RES_LAT"], CONFIG["GRID_RES_LAT"], dtype=np.float32)
    lon_bins = np.arange(CONFIG["MIN_LONGITUDE"], CONFIG["MAX_LONGITUDE"] + CONFIG["GRID_RES_LON"], CONFIG["GRID_RES_LON"], dtype=np.float32)
    n_lat_cells = len(lat_bins) - 1
    n_lon_cells = len(lon_bins) - 1
    print(f"\nGrid Sistemi: {n_lat_cells} Enlem x {n_lon_cells} Boylam = {n_lat_cells * n_lon_cells} Toplam Hücre")
    print(f"Çözünürlük: {CONFIG['GRID_RES_LAT']}° Enlem x {CONFIG['GRID_RES_LON']}° Boylam")
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2.0
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2.0

    print("\n--- Adım 1: Veri Çekme ve İşleme ---")
    usgs_df = fetch_earthquake_data_paginated(
        CONFIG["USGS_API_URL"], "USGS", CONFIG["START_TIME"], CONFIG["END_TIME"],
        CONFIG["MIN_LATITUDE"], CONFIG["MAX_LATITUDE"], CONFIG["MIN_LONGITUDE"], CONFIG["MAX_LONGITUDE"],
        CONFIG["API_MIN_MAGNITUDE"], CONFIG["API_TIMEOUT"], CONFIG["API_MAX_RETRIES"], CONFIG["API_SLEEP_TIME"]
    )
    emsc_df = fetch_earthquake_data_paginated(
        CONFIG["EMSC_API_URL"], "EMSC", CONFIG["START_TIME"], CONFIG["END_TIME"],
        CONFIG["MIN_LATITUDE"], CONFIG["MAX_LATITUDE"], CONFIG["MIN_LONGITUDE"], CONFIG["MAX_LONGITUDE"],
        CONFIG["API_MIN_MAGNITUDE"], CONFIG["API_TIMEOUT"], CONFIG["API_MAX_RETRIES"], CONFIG["API_SLEEP_TIME"]
    )

    earthquake_catalog = merge_and_deduplicate_catalogs(
        [usgs_df, emsc_df],
        CONFIG["DEDUPLICATION_TIME_WINDOW"], CONFIG["DEDUPLICATION_DIST_WINDOW"]
    )
    del usgs_df, emsc_df
    gc.collect()

    if earthquake_catalog.empty:
        print("\nHata: Deprem kataloğu oluşturulamadı veya boş. Program sonlandırılıyor.")
        exit()

    if not earthquake_catalog.empty and pd.api.types.is_datetime64_any_dtype(earthquake_catalog['timestamp']):
        if earthquake_catalog['timestamp'].dt.tz is None:
            print(f"\nUyarı: Zaman damgaları UTC veya başka bir TZ olmadan okundu. {ISTANBUL_TZ} varsayılıyor.")
            try:
                earthquake_catalog['timestamp'] = earthquake_catalog['timestamp'].dt.tz_localize(ISTANBUL_TZ, ambiguous='infer')
            except Exception as tz_loc_err:
                 print(f"Hata: Zaman damgalarını yerelleştirirken: {tz_loc_err}. UTC kullanılacak.")
                 earthquake_catalog['timestamp'] = earthquake_catalog['timestamp'].dt.tz_localize(pytz.utc, ambiguous='infer')
        elif earthquake_catalog['timestamp'].dt.tz != ISTANBUL_TZ:
            print(f"\nZaman damgaları {earthquake_catalog['timestamp'].dt.tz} bölgesinden {ISTANBUL_TZ} saat dilimine dönüştürülüyor...")
            try:
                earthquake_catalog['timestamp'] = earthquake_catalog['timestamp'].dt.tz_convert(ISTANBUL_TZ)
                print("...Dönüştürme tamamlandı.")
            except Exception as tz_conv_err:
                print(f"Hata: Zaman damgaları dönüştürülürken: {tz_conv_err}. Orijinal TZ ({earthquake_catalog['timestamp'].dt.tz}) korunuyor.")
        else:
            print(f"\nZaman damgaları zaten {ISTANBUL_TZ} saat diliminde.")

    print(f"\nSon Birleştirilmiş Katalog Bilgileri:")
    print(f"  Toplam Olay Sayısı: {len(earthquake_catalog)}")
    if not earthquake_catalog.empty:
        min_ts = earthquake_catalog['timestamp'].min()
        max_ts = earthquake_catalog['timestamp'].max()
        print(f"  Zaman Aralığı: {min_ts.strftime('%Y-%m-%d %H:%M %Z')} - {max_ts.strftime('%Y-%m-%d %H:%M %Z')}")

    print("\n--- Adım 2: Özellik ve Hedef Matrisi Oluşturma ---")
    X, y, timestamps, cell_indices, feature_names_list = create_feature_target_matrix(
        earthquake_catalog, lat_bins, lon_bins, CONFIG
    )

    if X is None or y is None or not feature_names_list:
        print("\nHata: Özellik/hedef matrisi oluşturulamadı. Program sonlandırılıyor.")
        exit()

    print("\n--- Adım 3: Model Eğitimi ve Değerlendirme ---")
    best_params = None
    if CONFIG["ENABLE_OPTUNA"]:
        best_params = tune_hyperparameters_optuna(X, y, feature_names_list, CONFIG)
    else:
        print("\nOptuna hiperparametre optimizasyonu atlandı (konfigürasyonda devre dışı).")

    _, trained_params, best_iteration_from_cv, cv_metrics_df = train_evaluate_cv(
        X, y, feature_names_list, CONFIG, best_params
    )

    final_model = train_final_model(
        X, y, feature_names_list, trained_params, best_iteration_from_cv, CONFIG
    )

    del X, y, timestamps, cell_indices
    gc.collect()

    print("\n--- Adım 4: Gelecek Tahmini ve Çıktılar ---")
    if final_model is not None:
        predictions_df = generate_future_predictions(
            final_model, earthquake_catalog, lat_bins, lon_bins, feature_names_list, CONFIG
        )

        if predictions_df is not None and not predictions_df.empty:
            print(f"\nGelecek için {len(predictions_df)} hücrenin olasılığı tahmin edildi.")
            save_predictions_to_file(predictions_df, lat_centers, lon_centers, CONFIG)
            create_prediction_map(predictions_df, lat_centers, lon_centers, lat_bins, lon_bins, CONFIG)
        else:
            print("\nGelecek tahmini oluşturulamadı veya boş sonuç döndü.")
    else:
        print("\nNihai model eğitilemediği için gelecek tahmini yapılamıyor.")

    end_time_script = time.time()
    total_time = end_time_script - start_time_script
    print("\n" + "="*60)
    print("--- PROGRAM BAŞARIYLA TAMAMLANDI ---")
    print(f"Toplam Çalışma Süresi: {total_time:.2f} saniye ({total_time/60:.2f} dakika)")
    print(f"Çıktı Dosyaları:")
    print(f"  - Tahmin Metin Dosyası: {CONFIG['PREDICTION_OUTPUT_FILE']}")
    if FOLIUM_AVAILABLE: print(f"  - Tahmin Haritası: {CONFIG['PREDICTION_MAP_FILE']}")
    print("="*60)
    print("\n" + "="*60)
    print(" !!! YASAL UYARI VE ÖNEMLİ NOTLAR !!!")
    print(" ="*60)
    print(f"* Bu program, gelecek {CONFIG['PREDICTION_WINDOW_DAYS']} gün için M{CONFIG['MAGNITUDE_THRESHOLD']}+ deprem olma olasılığını İSTATİSTİKSEL olarak tahmin etmeye çalışır.")
    print("* Sonuçlar TAMAMEN DENEYSELDİR ve BİLİMSEL KESİNLİK VEYA DOĞRULUK GARANTİSİ TAŞIMAZ.")
    print("* Bu bir GERÇEK ZAMANLI DEPREM TAHMİN SİSTEMİ DEĞİLDİR.")
    print("* Üretilen sonuçlar ASLA ve ASLA hayati kararlar almak için kullanılmamalıdır.")
    print("* Güvenilir ve güncel deprem bilgileri için lütfen AFAD, Kandilli Rasathanesi (KOERI), USGS, EMSC gibi resmi kurumların açıklamalarını takip ediniz.")
    print("="*60)

if __name__ == "__main__":
    try:
        import pytz
    except ImportError:
        print("*"*70)
        print("HATA: 'pytz' kütüphanesi bulunamadı.")
        print("Saat dilimi desteği için gereklidir. Lütfen kurun:")
        print("pip install pytz")
        print("*"*70)
        print("Uyarı: pytz bulunamadığı için UTC saat dilimi kullanılacak.")
        from datetime import timezone
        ISTANBUL_TZ = timezone.utc

    main()
