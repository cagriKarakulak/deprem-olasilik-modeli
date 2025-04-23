# Deneysel Deprem Olasılık Tahmin Modeli (Türkiye ve Çevresi)

![Untitled](https://github.com/user-attachments/assets/3049b125-45c3-46d5-a1d7-06ba849bb153)

Bu Python betiği, belirli bir coğrafi bölge (varsayılan olarak Türkiye ve çevresi) için gelecekteki belirli bir zaman aralığında (varsayılan olarak 30 gün) belirli bir büyüklük eşiğinin (varsayılan M5.0) üzerindeki depremlerin **istatistiksel olasılığını** tahmin etmeye çalışan **deneysel** bir modeldir.

**!!! ÇOK ÖNEMLİ UYARILAR !!!**

*   Bu proje **tamamen deneyseldir** ve yalnızca araştırma ve eğitim amaçlıdır.
*   Model, geçmiş verilerdeki istatistiksel korelasyonları öğrenir. Bu korelasyonların gelecekte devam edeceğinin veya nedensellik içerdiğinin **hiçbir garantisi yoktur.**
*   Tahminler, belirli bir bölgede riskin *istatistiksel olarak* artmış olabileceğine dair **olasılık** değerleridir. Gerçek bir depremin olacağı veya olmayacağı anlamına **gelmez.** Kesin zaman, yer veya büyüklük tahmini **yapılamaz.**
*   Modelin performansı (özellikle gerçek depremleri yakalama oranı - Recall) sınırlı olabilir ve **birçok büyük depremi gözden kaçırabilir.** Aynı zamanda **yanlış alarmlar** (düşük Precision) üretebilir.
*   Çıktıların yorumlanması sismoloji uzmanlığı gerektirir.
*   **BU KOD VE ÜRETTİĞİ SONUÇLAR, GERÇEK HAYATTA KARAR ALMA, UYARI SİSTEMİ VEYA HERHANGİ BİR KRİTİK UYGULAMA İÇİN KESİNLİKLE KULLANILMAMALIDIR!**

## Projenin Amacı ve Yöntemi

Model, USGS ve EMSC gibi kamuya açık sismik veri kaynaklarından geçmiş deprem verilerini çeker, bu verileri birleştirir ve tekilleştirir. Ardından bölgeyi bir grid (ızgara) sistemine böler ve her bir hücre için belirli zaman pencerelerindeki sismik aktiviteye dayalı özellikler (deprem sayısı, ortalama/maksimum büyüklük, enerji vekili, b-değeri vb.) hesaplar. İsteğe bağlı olarak komşu hücrelerin etkileşimlerini de dikkate alabilir. Bu özellikler kullanılarak, her hücre için gelecek tahmin penceresinde (örn. 30 gün) hedef büyüklükte bir deprem olup olmayacağını tahmin etmek üzere bir XGBoost makine öğrenimi modeli eğitilir. Son olarak, model en güncel verilere dayanarak gelecek için olasılık tahminleri üretir ve bunları bir metin dosyasına ve isteğe bağlı olarak interaktif bir HTML haritasına kaydeder.

## Özellikler

*   **Veri Toplama:** USGS ve EMSC API'larından belirlenen zaman aralığı ve bölge için deprem verisi çeker.
*   **Veri İşleme:** Farklı kaynaklardan gelen verileri birleştirir ve zaman/mekan yakınlığına göre kopya kayıtları kaldırır (tekilleştirme).
*   **Grid Sistemi:** Çalışma bölgesini belirlenen çözünürlükte enlem/boylam hücrelerine ayırır.
*   **Özellik Mühendisliği:** Her hücre ve zaman adımı için çeşitli sismik özellikler hesaplar:
    *   Deprem Sayısı
    *   Ortalama ve Maksimum Büyüklük
    *   Enerji Vekili (Gutenberg-Richter ilişkisine dayalı)
    *   Depremler Arası Ortalama/Standart Sapma Zaman Farkı
    *   b-değeri ve Tamamlanma Büyüklüğü (Mc)
    *   Ortalama Derinlik
    *   *(İsteğe Bağlı)* Komşu hücrelerden toplanan özellikler (komşu özelliklerinin ortalaması/maksimumu/sayısı).
*   **Hedef Tanımlama:** Hedef değişkeni, özellik hesaplama zamanını takip eden tahmin penceresi içinde bir hücrede önemli bir depremin (>= Büyüklük Eşiği) meydana gelip gelmediği olarak tanımlar.
*   **Makine Öğrenimi Modeli:** Oluşturulan özellikler ve hedefler üzerinde bir XGBoost sınıflandırıcı modeli eğitir.
*   **Zaman Serisi Çapraz Doğrulama:** Modelin performansını zamana duyarlı çapraz doğrulama bölümleri kullanarak değerlendirir.
*   **Hiperparametre Optimizasyonu (İsteğe Bağlı):** Potansiyel olarak daha iyi XGBoost hiperparametrelerini bulmak için Optuna kütüphanesini kullanır.
*   **Gelecek Tahmini:** Eğitilmiş modeli ve en güncel verileri kullanarak *bir sonraki* tahmin penceresi için her bir grid hücresindeki önemli bir deprem olasılığını tahmin eder.
*   **Çıktı:**
    *   Tanımlanmış bir eşiğin üzerindeki tahmin edilen olasılıklara sahip grid hücrelerini listeleyen bir `.txt` dosyası oluşturur (yaklaşık konum adları için `geopy` gerektirir).
    *   Tahmin edilen olasılıkları bir eşiğin üzerinde görselleştiren interaktif bir `.html` haritası oluşturur (`folium` ve `branca` gerektirir).
    *   İlerleme durumunu, yapılandırma ayrıntılarını ve değerlendirme metriklerini konsola yazdırır.


## Teknoloji ve Kütüphaneler

*   Python 3.x
*   requests
*   pandas
*   numpy
*   pytz
*   scikit-learn
*   xgboost
*   tqdm
*   scipy
*   geopy
*   folium
*   branca
*   optuna

## Kurulum

1.  Python 3.x kurulu olduğundan emin olun.
2.  Bir terminal veya komut istemcisi açın.
3.  Proje dizinine gidin.
4.  Gerekli kütüphaneleri kurmak için aşağıdaki komutu çalıştırın:
    ```bash
    pip install -r requirements.txt
    ```

## Kullanım
1.   requirements.txt dosyasında listelenen tüm gerekli (ve istenen isteğe bağlı) kütüphanelerin kurulu olduğundan emin olun.
2.   Gerekirse deprem_tahmin.py içindeki CONFIG sözlüğünü değiştirin.
3.  Terminalde proje dizinindeyken aşağıdaki komutu çalıştırın:
    ```bash
    python deprem_tahmin.py
    ```
4.   Betik, ilerleme güncellemelerini ve sonuçları konsola yazdıracaktır.
5.   Tamamlandığında, çıktı dosyaları için mevcut dizini kontrol edin (örn. deprem_tahminleri_YYYYMMDD_HHMMSS_TZ.txt ve deprem_olasılık_haritası_YYYYMMDD_HHMMSS_TZ.html).

## Yapılandırma (CONFIG)
Anahtar parametreler deprem_tahmin.py betiğinin başındaki CONFIG sözlüğü içinde ayarlanabilir. Önemli olanlardan bazıları şunlardır:

    MIN/MAX_LATITUDE, MIN/MAX_LONGITUDE: Çalışma alanının coğrafi sınırları.

    GRID_RES_LAT, GRID_RES_LON: Grid hücrelerinin çözünürlüğü.

    DATA_YEARS: Geçmişten kaç yıllık verinin çekileceği.

    API_MIN_MAGNITUDE: API'lardan çekilecek minimum büyüklük (daha düşük değerler veri hacmini artırır).

    FEATURE_WINDOWS_DAYS: Özelliklerin hesaplanacağı zaman pencerelerinin listesi (gün cinsinden, örn. 90 günlük özellikler için [90]).

    PREDICTION_WINDOW_DAYS: Tahminin geleceğe yönelik süresi (gün cinsinden).

    TIME_STEP_DAYS: Geçmiş verilerde ne sıklıkla bir özellik/hedef anlık görüntüsü oluşturulacağı (eğitim verisi boyutunu etkiler).

    MAGNITUDE_THRESHOLD: Tahmin için hedeflenen deprem büyüklüğü (örn. M5.0+ için 5.0).

    ENABLE_NEIGHBOR_FEATURES: Komşu özelliklerini hesaplamak için True olarak ayarlayın (hız için scipy gerektirir).

    ENABLE_OPTUNA: Hiperparametre ayarını etkinleştirmek için True olarak ayarlayın (optuna gerektirir).

    FUTURE_PREDICTION_PROB_THRESHOLD: Bir hücrenin çıktı .txt dosyasına ve haritaya dahil edilmesi için gereken minimum olasılık.


## Sorumluluk Reddi

Bu yazılım "olduğu gibi" sağlanmaktadır ve herhangi bir garanti verilmemektedir. Yazılımın kullanımından doğabilecek doğrudan veya dolaylı hiçbir zarardan geliştirici(ler) sorumlu tutulamaz. Deprem tahmini oldukça karmaşık ve belirsiz bir alandır. Bu aracı kullanarak elde edilen bilgilere dayanarak önemli kararlar vermeyin. Her zaman resmi kurumların (AFAD, KOERI vb.) uyarı ve bilgilerini takip edin.

## Lisans

Bu proje [MIT Lisansı](https://opensource.org/licenses/MIT) altında lisanslanmıştır.
