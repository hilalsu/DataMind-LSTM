# **DataMind-LSTM — Proje Özeti & Çalıştırma Kılavuzu**

Bu doküman, projenin mevcut durumunu başlangıçtan çalıştırmaya kadar tüm yönleriyle açıklar. Kurulum adımları, beklenen çıktılar, bilinen kısıtlar ve öneriler tek bir yerde toplanmıştır.


## **1. Projenin Genel Özeti**

DataMind-LSTM; veri ön işleme, görselleştirme, geleneksel makine öğrenmesi modelleri ve derin öğrenme tabanlı **LSTM & Bidirectional LSTM** modellerini içeren uçtan uca bir modelleme pipeline’ıdır.

Mevcut çalışma kapsamında:

* Veri temizlenmiş ve işlenmiş 
* Görseller üretilmiş 
* LSTM/BiLSTM altyapısı projeye tamamen dahil edilmiştir 
* GUI ile gösterim planlanmıştır (Python GUI)

İşlenen veri `results/processed_data.csv` olarak kaydedilmiştir.


## **2. Görsel Analiz Özeti**

### Yaş Dağılımı

* Normalize değerler -1.0 ile +1.0 arasında.
* Ortalama değerler merkezde yoğunlaşmış.

### Şehir Dağılımı

* İstanbul en yüksek frekansa sahip (~600).
* Dağılım kuyruklu yapıdadır (long-tailed).

### Korelasyon Matrisi

* Orta düzey pozitif korelasyonlar tespit edilmiştir:

  * `day` ↔ `city_encoded` (0.32)
  * `reason_encoded` ↔ `killer_encoded` (0.27)

### Fail Durumu Dağılımı

* "Tutuklu" baskın (%53.8)
* Dağılım aşırı dengesiz → Model performansını etkileyebilir.

### Öldürülme Şekli

* “Ateşli Silah” en yaygın yöntem (~1750)

### Yıl Dağılımı

* Yıllar arasında belirgin dalgalanmalar mevcut.

### Yaş Box-Plot

* Yayılım normalize sınırlar (-1.0 / +1.0) arasında.
* Uç değerler sınır uçlarında yoğun.

---

## **3. Model Yapısı**

### Geleneksel Modeller

* Logistic Regression
* Random Forest
* SVM

### Hiperparametre Optimizasyonu

* RandomizedSearchCV uygulanmıştır.

### LSTM & Bidirectional LSTM

* Tamamen projeye dahil 
* Veri (samples, 1, features) formatında LSTM’e dönüştürülür.
* Dropout + Dense + Softmax mimarisi uygulanır.
* Modeller `results/models/` klasörüne kaydedilir.

**Uyarı:** Hedef sınıf sayısı yüksekse (51 sınıf), eğitim zorlaşır.


## **4. Kurulum**

**Gereksinim**

* Python 3.9+ (3.11 ile test edildi)

**Yükleme**

```powershell
python -m pip install -r "c:\Users\hilal\Desktop\DataMind-LSTM\requirements.txt"
```

TensorFlow ilk yüklemede zaman alabilir.

---

## **5. Çalıştırma Komutları**

**Sadece Veri Ön İşleme**

```powershell
$env:PYTHONIOENCODING='utf-8'; python -c "from data_preprocessing import main; main()"
```

**Tam Pipeline (LSTM Dahil)**

```powershell
$env:PYTHONIOENCODING='utf-8'; python "c:\Users\hilal\Desktop\DataMind-LSTM\main.py"
```

**Log Kaydetme**

```powershell
$env:PYTHONIOENCODING='utf-8'; python "c:\Users\hilal\Desktop\DataMind-LSTM\main.py" 2>&1 | Tee-Object "c:\Users\hilal\Desktop\DataMind-LSTM\full_run_raw.log"; Get-Content "c:\Users\hilal\Desktop\DataMind-LSTM\full_run_raw.log" | Out-File "c:\Users\hilal\Desktop\DataMind-LSTM\full_run.log" -Encoding utf8
```

**Hızlı Baseline (RandomForest)**

```powershell
$env:PYTHONIOENCODING='utf-8'; python - <<'PY'
from data_preprocessing import main as preprocess_main
from modeling import ModelTrainer
res = preprocess_main()
trainer = ModelTrainer()
trainer.train_traditional_models(res['X_train'], res['X_val'], res['y_train'], res['y_val'])
PY
```


## **6. Bilinen Kısıtlar**

* **Sınıf Dengesizliği**:
  Bazı hedef sınıflarda çok düşük örnek sayısı (1–2 adet)

* **Multi-Label Belirsizliği**:
  Bazı kayıtlar birden fazla etikete sahip olabilir.

* **LSTM Avantajı**:
  Tabular veride sınırlı olabilir—yorumlama dikkat gerektirir.

* **Zaman/Maliyet**:
  LSTM ve RandomSearchCV eğitimleri uzun sürebilir.


## **Önemli Dosya Yolları**

* İşlenmiş Veri: `results/processed_data.csv`
* Görseller: `results/*.png`
* Loglar: `run_output.log`, `full_run.log`
* Modeller: `results/models/`



### Proje: **DataMind-LSTM**

