# Deneysel Deprem Olasılık Tahmin Modeli (Türkiye ve Çevresi)

![Untitled](https://github.com/user-attachments/assets/f7656075-5fa0-4d49-87d6-466ccdc28490)
![Untitled2](https://github.com/user-attachments/assets/edb358f0-a1c8-4c83-941c-7bd67ced55d5)


Bu proje, Türkiye ve yakın çevresindeki geçmiş deprem verilerini kullanarak, belirli grid hücrelerinde gelecekteki belirli bir zaman diliminde (varsayılan olarak 30 gün) M5.0 veya üzeri bir deprem olma **olasılığını** tahmin etmeye çalışan **deneysel** bir makine öğrenmesi modelidir.

**!!! ÇOK ÖNEMLİ UYARILAR !!!**

*   Bu proje **tamamen deneyseldir** ve yalnızca araştırma ve eğitim amaçlıdır.
*   Model, geçmiş verilerdeki istatistiksel korelasyonları öğrenir. Bu korelasyonların gelecekte devam edeceğinin veya nedensellik içerdiğinin **hiçbir garantisi yoktur.**
*   Tahminler, belirli bir bölgede riskin *istatistiksel olarak* artmış olabileceğine dair **olasılık** değerleridir. Gerçek bir depremin olacağı veya olmayacağı anlamına **gelmez.** Kesin zaman, yer veya büyüklük tahmini **yapılamaz.**
*   Modelin performansı (özellikle gerçek depremleri yakalama oranı - Recall) sınırlı olabilir ve **birçok büyük depremi gözden kaçırabilir.** Aynı zamanda **yanlış alarmlar** (düşük Precision) üretebilir.
*   Çıktıların yorumlanması sismoloji uzmanlığı gerektirir.
*   **BU KOD VE ÜRETTİĞİ SONUÇLAR, GERÇEK HAYATTA KARAR ALMA, UYARI SİSTEMİ VEYA HERHANGİ BİR KRİTİK UYGULAMA İÇİN KESİNLİKLE KULLANILMAMALIDIR!**

## Projenin Amacı ve Yöntemi

Proje, belirlenen coğrafi bölgeyi (varsayılan olarak Türkiye ve çevresi) grid hücrelerine böler. Her hücre için, geçmişteki belirli bir zaman penceresindeki (varsayılan 90 gün) deprem aktivitesine dayalı özellikler (deprem sayısı, ortalama/maksimum büyüklük, enerji, b-değeri, Mc, zaman farkları vb.) hesaplar. Bu özellikler kullanılarak bir XGBoost makine öğrenmesi modeli eğitilir. Model, her hücre için bir sonraki zaman penceresinde (varsayılan 30 gün) eşik büyüklüğün (varsayılan M5.0) üzerinde bir deprem olup olmayacağını (0 veya 1) sınıflandırmaya çalışır. Finalde, en güncel verilere dayanarak her hücre için bu olayın gerçekleşme olasılığını tahmin eder ve belirli bir olasılık eşiğinin üzerindeki hücreleri raporlar.

## Özellikler

*   USGS API'sinden (ve potansiyel olarak diğer FDSN kaynaklarından) tarihsel deprem verilerini çeker.
*   Veri çekme sırasında sayfalandırma ve hata yönetimi uygular.
*   Basit bir veri tekilleştirme (deduplication) adımı içerir (farklı kaynaklar kullanıldığında).
*   Bölgeyi grid hücrelerine böler.
*   Zaman serisi yaklaşımıyla her hücre için özellikler üretir (b-değeri, enerji vb. dahil).
*   `TimeSeriesSplit` kullanarak zaman serisine uygun çapraz doğrulama yapar.
*   XGBoost sınıflandırma modeli eğitir (`scale_pos_weight` ile dengesiz sınıfları dikkate alır).
*   `tqdm` ile ilerleme çubukları gösterir.
*   En güncel verilere göre gelecek 30 gün için her hücrenin M5.0+ deprem olasılığını tahmin eder.
*   `geopy` (varsa) kullanarak yüksek olasılıklı hücreler için yaklaşık bölge isimlerini ve Google Haritalar linklerini alır.
*   Yüksek olasılıklı tahminleri bir `.txt` dosyasına kaydeder.

## Veri Kaynakları

*   **Ana Kaynak (Mevcut Kodda):** [USGS Earthquake Catalog API](https://earthquake.usgs.gov/fdsnws/event/1/)
*   **Potansiyel Diğer Kaynaklar (Geliştirilebilir):**
    *   [EMSC FDSNWS](https://www.seismicportal.eu/fdsnws/event/1/)
    *   [KOERI (Kandilli)](http://www.koeri.boun.edu.tr/scripts/lst0.asp) (API/FDSN erişimi araştırılmalı, web scraping stabil değildir)
    *   [ISC (International Seismological Centre)](http://www.isc.ac.uk/iscbulletin/) (Tarihsel veriler için en iyi kaynaklardan biri, indirilmesi gerekir)

**Önemli Not:** Uzun tarihsel veriler (örn. 10+ yıl) için API kullanmak çok verimsiz ve limitlere takılabilir. Ciddi modelleme için **ISC, KOERI gibi kaynaklardan önceden indirilmiş, temizlenmiş ve homojenize edilmiş katalogları** kullanmak şiddetle tavsiye edilir.

## Teknoloji ve Kütüphaneler

*   Python 3.x
*   pandas
*   numpy
*   scikit-learn
*   xgboost
*   requests
*   tqdm
*   geopy (Opsiyonel - Bölge isimleri için)
*   folium
*   branca

## Kurulum

1.  Python 3.x kurulu olduğundan emin olun.
2.  Bir terminal veya komut istemcisi açın.
3.  Proje dizinine gidin.
4.  Gerekli kütüphaneleri kurmak için aşağıdaki komutu çalıştırın:
    ```bash
    pip install -r requirements.txt
    ```

## Kullanım

1.  Terminalde proje dizinindeyken aşağıdaki komutu çalıştırın:
    ```bash
    python deprem_tahmin.py
    ```
2.  Script çalışmaya başlayacak, veri çekecek, özellikleri hesaplayacak, modeli eğitecek ve son olarak gelecek 30 gün için yüksek olasılıklı tahminleri konsola ve `deprem_tahminleri_YYYYMMDD_HHMM.txt` adlı bir dosyaya yazacaktır.
3.  İşlemler (özellikle veri çekme ve özellik hesaplama) veri miktarına ve pencere boyutlarına bağlı olarak uzun sürebilir.

## Yapılandırma

Scriptin başındaki "Ayarlar ve Sabitler" bölümünden bazı parametreleri değiştirebilirsiniz:

*   `MIN_LATITUDE`, `MAX_LATITUDE`, `MIN_LONGITUDE`, `MAX_LONGITUDE`: Çalışılacak coğrafi bölge.
*   `GRID_RES_LAT`, `GRID_RES_LON`: Grid hücrelerinin boyutu (derece cinsinden). Daha küçük değerler daha fazla detay ama daha fazla hesaplama yükü demektir.
*   `DATA_YEARS`: API'den çekilecek tarihsel veri süresi (yıl olarak). **Dikkat:** API ile çok uzun süreler çekmek sorunlu olabilir.
*   `FEATURE_WINDOW_DAYS`, `PREDICTION_WINDOW_DAYS`, `TIME_STEP_DAYS`: Özellik mühendisliği ve tahmin için zaman pencereleri.
*   `MAGNITUDE_THRESHOLD`: "Büyük" kabul edilecek minimum deprem büyüklüğü (hedef değişken için).
*   `B_VALUE_MIN_QUAKES`: b-değeri hesaplamak için gereken minimum deprem sayısı.
*   `FUTURE_PREDICTION_PROB_THRESHOLD`: Raporlanacak minimum olasılık eşiği.
*   `PREDICTION_OUTPUT_FILE`: Çıktı dosyasının adı.
*   `XGB_TREE_METHOD`: `'hist'` (CPU için hızlı) veya `'gpu_hist'` (GPU varsa çok daha hızlı).

## Geliştirme Fikirleri

*   API yerine önceden indirilmiş ve temizlenmiş tarihsel kataloglar (ISC, KOERI vb.) kullanmak.
*   Farklı kaynaklardan gelen veriler için daha gelişmiş tekilleştirme (deduplication) algoritmaları uygulamak.
*   Magnitüd homojenizasyonu yapmak (tüm veriyi Mw'ye dönüştürmek).
*   Eksik verileri (özelliklerdeki NaN) basitçe doldurmak yerine daha gelişmiş imputation teknikleri kullanmak (örn. `sklearn.impute.KNNImputer`).
*   Daha fazla özellik eklemek:
    *   Bilinen fay hatlarına uzaklık (`geopandas`, `shapely` ile).
    *   GPS verilerinden elde edilen gerinim oranları (erişilebilir veri kaynakları araştırılmalı).
    *   Özelliklerin zaman içindeki değişim oranları.
    *   Komşu hücrelerin özellikleri.
*   XGBoost için hiperparametre optimizasyonu yapmak (`GridSearchCV`, `RandomizedSearchCV`, `Optuna` vb.).
*   Farklı makine öğrenmesi modellerini denemek (LightGBM, CatBoost, derin öğrenme modelleri - LSTM, ConvLSTM).
*   Tahmin eşik değerini (şu an 0.5 veya `FUTURE_PREDICTION_PROB_THRESHOLD`) Precision-Recall eğrisi gibi yöntemlerle optimize etmek.
*   Sonuçları görselleştirmek (örn. risk haritaları oluşturmak - `matplotlib`, `folium`). (EKLENDI)

## Katkıda Bulunma

Katkıda bulunmak isterseniz lütfen önce bir "issue" açarak veya mevcut bir "issue" üzerinden iletişime geçin.

## Sorumluluk Reddi

Bu yazılım "olduğu gibi" sağlanmaktadır ve herhangi bir garanti verilmemektedir. Yazılımın kullanımından doğabilecek doğrudan veya dolaylı hiçbir zarardan geliştirici(ler) sorumlu tutulamaz. Deprem tahmini oldukça karmaşık ve belirsiz bir alandır. Bu aracı kullanarak elde edilen bilgilere dayanarak önemli kararlar vermeyin. Her zaman resmi kurumların (AFAD, KOERI vb.) uyarı ve bilgilerini takip edin.

## Lisans

Bu proje [MIT Lisansı](https://opensource.org/licenses/MIT) altında lisanslanmıştır.
