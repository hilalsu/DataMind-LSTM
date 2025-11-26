# DataMind-LSTM — Güncel Proje Özeti ve Çalıştırma Kılavuzu

Bu döküman projeyi mevcut çalışma durumuna göre baştan sona açıklar; nasıl kurup çalıştıracağınız, hangi çıktıları bekleyeceğiniz, bilinen kısıtlar ve öneriler burada yer almaktadır.

**Kısa Özet:** proje veri ön işleme, görselleştirme ve modelleme (Geleneksel ML + LSTM ve Bidirectional LSTM kesin olarak dahil) adımlarını içerir. Veri ön işleme başarıyla çalıştırılmış, işlenmiş veriler `results/processed_data.csv` olarak kaydedildi ve birçok görsel üretildi. Model eğitimi (tam pipeline) istenirse başlatılabilir fakat LSTM/hiperparam optimizasyon ağır ve zaman alıcı olabilir.

**Önemli dosyalar:** `data_preprocessing.py`, `modeling.py`, `evaluation.py`, `main.py`, `requirements.txt` ve `results/` klasörü.

**Not:** README güncellemesi sırasında `data_preprocessing.py` içinde iki değişiklik yapıldı:
- `create_train_test_split()` fonksiyonuna: sınıf başına örnek sayısı çok azsa `stratify=None` kullanacak güvenli fallback eklendi.
- `create_visualizations()` içinde `killer_status` grafiği iyileştirildi ve `results/killer_status_distribution_improved.png` olarak kaydediliyor (küçük kategoriler "Diğer" altında toplanıyor, yatay bar, adet ve yüzde etiketleri).

**Dosya yolları (önemli):**
- İşlenmiş veri: `results/processed_data.csv`
- Tüm görseller: `results/*.png` (ör. `correlation_matrix.png`, `age_distribution.png`, `killer_status_distribution_improved.png`)
- Log (konsol çıktısı): `run_output.log`
- Modeller (eğitildiğinde): `results/models/`

**İçindekiler:**
- **Kurulum**
- **Hızlı Başlangıç (preprocessing / full run / baseline)**
- **Model detayları (özellikle LSTM & Bidirectional LSTM)**
- **Mevcut durum, bilinen sorunlar ve öneriler**
- **Sonraki adımlar / öneriler**

---

**Kurulum**

- Python 3.9+ (3.11 ile test edildi) önerilir.
- Gerekli paketler `requirements.txt` içinde listeli. Kurmak için PowerShell'de:

```powershell
python -m pip install -r "c:\Users\hilal\Desktop\DataMind-LSTM\requirements.txt"
```

Not: `tensorflow` gibi paketler sistemde büyük yer kaplar ve yükleme/ilk import sırasında zaman alabilir.

---

**Hızlı Başlangıç - Örnek Komutlar**

- Sadece veri ön işlemeyi çalıştır (hızlı smoke-test):

```powershell

$env:PYTHONIOENCODING='utf-8'; python -c "from data_preprocessing import main; main()"
```

- Tam pipeline (tüm adımlar: ön işleme -> geleneksel modeller -> hiperparam optimizasyon -> LSTM & BiLSTM -> değerlendirme -> rapor). Uyarı: LSTM ve RandomSearchCV ağırdır.

```powershell
$env:PYTHONIOENCODING='utf-8'; python "c:\Users\hilal\Desktop\DataMind-LSTM\main.py"
```

Bu komut çıktısını dosyaya kaydetmek istersen (önemli loglar için):

```powershell
$env:PYTHONIOENCODING='utf-8'; python "c:\Users\hilal\Desktop\DataMind-LSTM\main.py" 2>&1 | Tee-Object "c:\Users\hilal\Desktop\DataMind-LSTM\full_run_raw.log"; Get-Content "c:\Users\hilal\Desktop\DataMind-LSTM\full_run_raw.log" | Out-File "c:\Users\hilal\Desktop\DataMind-LSTM\full_run.log" -Encoding utf8
```

- Hızlı baseline: sadece işlenmiş veriyi kullanıp RandomForest ile tek adımlık bir baseline çalıştırmak istersen (örnek):

```powershell
$env:PYTHONIOENCODING='utf-8'; python - <<'PY'
from data_preprocessing import main as preprocess_main
from modeling import ModelTrainer
res = preprocess_main()
trainer = ModelTrainer()
trainer.train_traditional_models(res['X_train'], res['X_val'], res['y_train'], res['y_val'])
PY
```

---

**Model Detayları (özellikle LSTM & Bidirectional LSTM)**

- Geleneksel modeller: `Logistic Regression`, `Random Forest`, `SVM` — bunlar `modeling.py` içindeki `train_traditional_models()` fonksiyonunda hazırlanır.
- Hiperparametre optimizasyonu: `RandomizedSearchCV` ile Random Forest için uygulanır (`hyperparameter_optimization()` içinde).
- LSTM ve Bidirectional LSTM: kesinlikle projeye dahil edilmiştir (istenildiği gibi). Özellikler:
  - Veri tabanlı LSTM yaklaşımı: tabular veride her örnek tek bir timestep olarak ele alınıyor (shape → (samples, 1, features)). Bu nedenle LSTM, zaman serisi dışı tabular veride kullanılacaksa dikkatli yorumlanmalıdır.
  - Yapı (özet): iki katmanlı LSTM (128 -> 64), Dropout katmanları, Dense katman ve softmax çıkış.
  - Kayıt: eğitim geçmişleri ve model ağırlıkları `results/models/` içine kaydedilir (Keras `.h5`).

Önemli: LSTM/BiLSTM eğitimi için `n_classes` çok büyükse (burada `killer_status` için 51 sınıf) one-hot encoding ve softmax çıktısı zorlaşır; sınıf sayısını düşürmeyi veya problem tanımını değiştirmeyi düşünün.

---

**Mevcut Durum — Özet (2025-11-26)**

- Veri ön işleme: başarıyla tamamlandı. İşlenmiş veri: `results/processed_data.csv` (≈7169 satır).
- Görselleştirmeler üretildi: `results/` içine kaydedildi. `killer_status` görseli iyileştirildi ve `results/killer_status_distribution_improved.png` adıyla kaydedildi.
- `run_output.log` içinde veri ön işleme çıktısı kaydedildi (terminal çıktıları). Eğer tam pipeline çalıştırıldıysa `full_run.log` veya `full_run_raw.log` dosyalarını kontrol edin.
- Model eğitimi: henüz tam pipeline (tüm modeller + LSTM) otomatik olarak çalıştırıldıysa sonuçlar `results/model_comparison.csv`, `results/evaluation_metrics.csv` gibi dosyalarda bulunur; aksi takdirde modelleri çalıştırmak için `main.py` başlatılmalıdır.

---

**Bilinen Sorunlar / Uyarılar**

- Sınıf dengesizliği: `killer_status` hedefi yüksek kardinalitelidir (51 sınıf). Bazı sınıflarda çok az örnek (1-2) mevcut. Bu, stratify ile split hatalarına ve düşük model güvenilirliğine yol açar. Bu yüzden `create_train_test_split()` içine bir fallback (stratify=None) eklendi. Ancak stratify kapatılması dengesiz dağılımlara neden olabilir — öneriler aşağıda.
- Multi-label benzeri girdiler: bazı `killer_status` kayıtlarında birden fazla etiket ("Tutuklu, Aranıyor") veya karışık format var. Mevcut pipeline tek-etiket (`LabelEncoder`) varsayar. Eğer bu gerçekten multi-label ise, etiketleme stratejisi değişmeli.
- LSTM kullanımı: veri tabanlı tek-timestep yaklaşımla LSTM kullanmak mümkündür ancak genellikle tablo veride LSTM avantajı sınırlıdır. LSTM/BiLSTM yine de mevcut ve çalıştırılabilir; sonuçları dikkatle yorumlayın.
- Zaman ve kaynak: LSTM eğitimi ve RandomizedSearchCV CPU üzerinde çok uzun sürebilir; mümkünse GPU (CUDA) kullanın.

---

**Öneriler / Next Steps**

1. Sınıf azaltma veya yeniden dağıtma:
    - Nadir sınıfları `Diğer` olarak birleştirin veya yalnızca en sık görülen N sınıfa odaklanın.
    - Alternatif: sınıf ağırlığı (`class_weight`) veya oversampling (SMOTE vb.) uygulayın.
2. Multi-label kontrolü:
    - `killer_status` içindeki birden fazla etiket varsa temizleyin veya multi-label pipeline uygulayın.
3. Baseline değerlendirmesi:
    - Önce hızlı bir RandomForest baseline çalıştırıp `macro-F1` ve per-class rapor alın.
4. Tekrarlanabilirlik:
    - Seed ayarlarını (`numpy.random.seed`, `random.seed`, `tensorflow.random.set_seed`) sabitleyin ve kullanılan encoder/Scaler nesnelerini diske kaydedin.
5. GPU kullanımı:
    - Eğer GPU varsa TensorFlow GPU sürümünü ve CUDA uyumunu kurun; LSTM eğitim süresini önemli ölçüde azaltır.

---

**Hızlı Referans — Önemli Komutlar (PowerShell)**

- Paketleri yükle:

```powershell
python -m pip install -r "c:\Users\hilal\Desktop\DataMind-LSTM\requirements.txt"
```

- Sadece veri ön işleme (log terminalde gösterilir):

```powershell
$env:PYTHONIOENCODING='utf-8'; python -c "from data_preprocessing import main; main()"
```

- Tam pipeline (uzun, LSTM dahil):

```powershell
$env:PYTHONIOENCODING='utf-8'; python "c:\Users\hilal\Desktop\DataMind-LSTM\main.py"
```

- Tam pipeline çıktısını log dosyasına kaydet (UTF-8):

```powershell
$env:PYTHONIOENCODING='utf-8'; python "c:\Users\hilal\Desktop\DataMind-LSTM\main.py" 2>&1 | Tee-Object "c:\Users\hilal\Desktop\DataMind-LSTM\full_run_raw.log"; Get-Content "c:\Users\hilal\Desktop\DataMind-LSTM\full_run_raw.log" | Out-File "c:\Users\hilal\Desktop\DataMind-LSTM\full_run.log" -Encoding utf8
```

- Sadece hızlı baseline (RandomForest) çalıştırma örneği:

```powershell
$env:PYTHONIOENCODING='utf-8'; python - <<'PY'
from data_preprocessing import main as preprocess_main
from modeling import ModelTrainer
res = preprocess_main()
trainer = ModelTrainer()
trainer.train_traditional_models(res['X_train'], res['X_val'], res['y_train'], res['y_val'])
PY
```

---

**Sorular / Yardım**

- İstersen şu adımlardan başlayabilirim:
  - (A) Hemen tam pipeline'ı çalıştırıp sonuçları `results/` içine kaydetmemi ister misin? (uzun sürer)
  - (B) Önce `killer_status` sınıf dağılımını birlikte inceleyip nasıl gruplandıracağımıza karar verelim; sonra baseline ve LSTM adımlarına geçelim. (önerilen)
  - (C) Hemen hızlı RandomForest baseline çalıştırıp macro-F1, per-class raporu hazırlayayım.

Lütfen hangi adımı istediğini söyle; ben seçimine göre devam edip logları, grafikleri ve önerileri teslim edeceğim.

#### 2.1.2. Random Forest

**Amaç**: Ensemble yöntemi ile güçlü tahmin yapmak.

**Özellikler**:
- N_estimators: 100
- Random state: 42
- Parallel processing (n_jobs=-1)

**Kullanım Alanı**: Karmaşık ilişkileri yakalayan, robust model.

**Kod Konumu**: `modeling.py` → `train_traditional_models()` metodu

**Değerlendirme Metrikleri**:
- Accuracy
- F1-Score (weighted)

---

#### 2.1.3. Support Vector Machine (SVM)

**Amaç**: Kernel trick ile non-linear sınıflandırma yapmak.

**Özellikler**:
- Kernel: RBF (Radial Basis Function)
- Probability: True (olasılık tahminleri için)
- Sample size: 5000 (büyük veri setleri için optimizasyon)

**Not**: Büyük veri setleri için yavaş olduğundan örneklem kullanılmıştır.

**Kod Konumu**: `modeling.py` → `train_traditional_models()` metodu

**Değerlendirme Metrikleri**:
- Accuracy
- F1-Score (weighted)

---

### 2.2. Derin Öğrenme Modelleri

#### 2.2.1. LSTM (Long Short-Term Memory)

**Amaç**: Zaman serisi ve sequence verileri için derin öğrenme modeli.

**Model Mimarisi**:
```
Input Layer (Sequence)
    ↓
LSTM Layer 1 (128 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
LSTM Layer 2 (64 units, return_sequences=False)
    ↓
Dropout (0.3)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Output Layer (n_classes, Softmax)
```

**Hiperparametreler**:
- Sequence Length: 10
- Learning Rate: 0.001
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 50
- Batch Size: 32

**Callbacks**:
- **Early Stopping**: Validation loss'u izleyerek overfitting'i önler (patience=10)
- **ReduceLROnPlateau**: Learning rate'i dinamik olarak azaltır

**Kod Konumu**: `modeling.py` → `train_lstm_model()` metodu

**Çıktı**:
- Eğitilmiş model
- Eğitim geçmişi (history)
- `results/LSTM_training_history.png`: Eğitim grafikleri

---

#### 2.2.2. Bidirectional LSTM

**Amaç**: Geçmiş ve gelecek bilgilerini birlikte kullanan gelişmiş LSTM modeli.

**Model Mimarisi**:
```
Input Layer (Sequence)
    ↓
Bidirectional LSTM Layer 1 (128 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM Layer 2 (64 units, return_sequences=False)
    ↓
Dropout (0.3)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Output Layer (n_classes, Softmax)
```

**Avantajlar**:
- İleri ve geri yönlü bilgi akışı
- Daha iyi özellik öğrenme
- Genellikle tek yönlü LSTM'den daha iyi performans

**Hiperparametreler**: LSTM ile aynı

**Kod Konumu**: `modeling.py` → `train_bidirectional_lstm_model()` metodu

**Çıktı**:
- Eğitilmiş model
- Eğitim geçmişi
- `results/Bidirectional_LSTM_training_history.png`: Eğitim grafikleri

---

### 2.3. Hiperparametre Optimizasyonu

**Amaç**: En iyi model performansını bulmak için hiperparametreleri optimize etmek.

**Yöntem**: RandomizedSearchCV

**Model**: Random Forest

**Optimize Edilen Hiperparametreler**:

| Parametre | Değerler |
|-----------|----------|
| `n_estimators` | [50, 100, 200] |
| `max_depth` | [10, 20, 30, None] |
| `min_samples_split` | [2, 5, 10] |
| `min_samples_leaf` | [1, 2, 4] |
| `max_features` | ['sqrt', 'log2'] |

**Arama Stratejisi**:
- Iterations: 20
- Cross-Validation: 3-fold
- Scoring: F1-Score (weighted)
- Random State: 42

**İşlemler**:
1. Parametre grid'inin tanımlanması
2. RandomizedSearchCV ile arama
3. En iyi parametrelerin bulunması
4. En iyi modelin eğitilmesi ve değerlendirilmesi

**Kod Konumu**: `modeling.py` → `hyperparameter_optimization()` metodu

**Çıktı**:
- En iyi parametreler
- Optimize edilmiş model
- Performans metrikleri

---

### 2.4. Model Karşılaştırması

**Amaç**: Tüm modellerin performanslarını karşılaştırmak.

**Karşılaştırılan Modeller**:
1. Logistic Regression
2. Random Forest
3. SVM
4. Random Forest (Optimized)
5. LSTM
6. Bidirectional LSTM

**Karşılaştırma Metrikleri**:
- **Accuracy**: Doğru tahmin yüzdesi
- **F1-Score**: Precision ve Recall'un harmonik ortalaması (weighted)

**İşlemler**:
1. Tüm modellerin validation seti üzerinde değerlendirilmesi
2. Metriklerin toplanması
3. Karşılaştırma tablosunun oluşturulması
4. Görselleştirme (bar charts)
5. CSV olarak kaydetme

**Kod Konumu**: `modeling.py` → `compare_models()` metodu

**Çıktı**:
- `results/model_comparison.png`: Görsel karşılaştırma
- `results/model_comparison.csv`: Detaylı sonuçlar

---

## 🚀 Kullanım

### Yöntem 1: Tüm İşlemleri Tek Seferde Çalıştırma (Önerilen)

```bash
python main.py
```

Bu komut tüm işlemleri sırayla çalıştırır:
1. Veri ön işleme
2. Modelleme
3. Değerlendirme (confusion matrix, metrikler, overfitting analizi)
4. Rapor oluşturma

### Yöntem 2: Adım Adım Çalıştırma

#### Adım 1: Veri Ön İşleme

```bash
python data_preprocessing.py
```

Bu komut:
- Verileri yükler ve birleştirir
- Veriyi temizler
- Encoding ve normalizasyon yapar
- Korelasyon analizi yapar
- Görselleştirmeleri oluşturur
- Train/validation/test split oluşturur
- İşlenmiş veriyi kaydeder

#### Adım 2: Modelleme

```bash
python modeling.py
```

Bu komut:
- Geleneksel ML modellerini eğitir
- Hiperparametre optimizasyonu yapar
- LSTM ve Bidirectional LSTM modellerini eğitir
- Modelleri karşılaştırır
- Modelleri kaydeder

#### Adım 3: Değerlendirme

```bash
python evaluation.py
```

Veya `main.py` içinde otomatik olarak çalışır.

#### Adım 4: Rapor Oluşturma

```bash
python reporting.py
```

Veya `main.py` içinde otomatik olarak çalışır.

### Yöntem 3: GUI Uygulaması (Streamlit)

```bash
streamlit run gui_app.py
```

Bu komut web tabanlı bir arayüz açar ve şunları yapabilirsiniz:
- Veri yükleme
- Veri ön işleme
- Model eğitimi
- Sonuçları görüntüleme
- Grafikleri inceleme

### Sonuçları İnceleme

Tüm sonuçlar `results/` klasöründe bulunur:

- **Görselleştirmeler**: `*.png` dosyaları
- **İşlenmiş Veri**: `processed_data.csv`
- **Model Karşılaştırması**: `model_comparison.csv` ve `model_comparison.png`
- **Değerlendirme Metrikleri**: `evaluation_metrics.csv` ve `metrics_comparison.png`
- **Confusion Matrix'ler**: `confusion_matrix_*.png`
- **Overfitting Analizi**: `overfitting_analysis.json` ve `overfitting_analysis_*.png`
- **Eğitilmiş Modeller**: `models/` klasörü
- **En İyi Model**: `models/best_model.*` ve `models/best_model_info.json`
- **Proje Raporu**: `project_report.html`

---

## 📈 Sonuçlar ve Değerlendirme

### Model Performans Metrikleri

Modeller aşağıdaki metriklerle değerlendirilir:

1. **Accuracy**: Genel doğruluk oranı
2. **F1-Score**: Precision ve Recall'un dengeli ölçüsü

### En İyi Model Seçimi

En iyi model, validation seti üzerindeki performansa göre seçilir. Genellikle:
- **Derin Öğrenme Modelleri** (LSTM, Bidirectional LSTM): Karmaşık pattern'leri yakalama
- **Random Forest (Optimized)**: Robust ve yorumlanabilir
- **SVM**: Non-linear ilişkileri yakalama

---

## 🔍 Detaylı Açıklamalar

### Veri Ön İşleme Adımları

#### Eksik Veri İşleme Stratejisi

- **Kategorik Değişkenler**: "Unknown" ile doldurulur
- **Sayısal Değişkenler**: 0 ile doldurulur (normalizasyon sonrası)
- **Tarih**: Geçersiz tarihler NaN olarak bırakılır

#### Aykırı Değer İşleme Stratejisi

- **Yöntem**: IQR (Interquartile Range)
- **İşlem**: Silme yerine sınır değerleriyle değiştirme
- **Neden**: Veri kaybını önlemek

#### Encoding Stratejisi

- **Label Encoding**: Kategorik değişkenler için
- **One-Hot Encoding**: LSTM modelleri için (categorical crossentropy loss)

#### Normalizasyon Stratejisi

- **StandardScaler**: Z-score normalizasyonu
- **Neden**: Farklı ölçeklerdeki değişkenleri aynı ölçeğe getirmek

### Modelleme Stratejileri

#### Sequence Preparation (LSTM için)

- **Sequence Length**: 10 (ayarlanabilir)
- **Padding**: Kısa sequence'ler için
- **One-Hot Encoding**: Çok sınıflı sınıflandırma için

#### Overfitting Önleme

- **Dropout Layers**: %20-30 dropout
- **Early Stopping**: Validation loss izleme
- **Learning Rate Reduction**: Dinamik öğrenme oranı

#### Model Kaydetme

- **Keras Modelleri**: `.h5` formatında
- **Scikit-learn Modelleri**: `.pkl` formatında (pickle)

---

## 📝 Notlar

1. **Veri Gizliliği**: Bu proje hassas veriler içermektedir. Verilerin kullanımında etik kurallara uyulmalıdır.

2. **Performans**: Büyük veri setleri için model eğitim süreleri uzun olabilir. GPU kullanımı önerilir.

3. **Hiperparametreler**: Tüm hiperparametreler ayarlanabilir. Proje dosyalarındaki ilgili bölümlerden değiştirilebilir.

4. **Encoding**: Label encoder'lar model kaydetme sırasında saklanmalıdır. Test verisi için aynı encoder'lar kullanılmalıdır.

---

## 🛠️ Sorun Giderme

### Yaygın Hatalar

1. **Memory Error**: 
   - Batch size'ı küçültün
   - Sequence length'i azaltın

2. **CUDA/GPU Hatası**:
   - CPU moduna geçin: TensorFlow otomatik olarak CPU kullanır

3. **Encoding Hatası**:
   - Label encoder'ların doğru yüklendiğinden emin olun

---

## 📚 Referanslar

- Scikit-learn Documentation: https://scikit-learn.org/
- TensorFlow Documentation: https://www.tensorflow.org/
- Pandas Documentation: https://pandas.pydata.org/
- Matplotlib Documentation: https://matplotlib.org/

---

## 👥 Katkıda Bulunanlar

Bu proje DataMind-LSTM ekibi tarafından geliştirilmiştir.

---

## 📄 Lisans

Bu proje eğitim amaçlıdır.

---

## 📞 İletişim

Sorularınız için issue açabilirsiniz.

---

**Son Güncelleme**: 2024

