# 🚀 Hızlı Başlangıç Kılavuzu

## Kurulum

```bash
# 1. Gerekli kütüphaneleri yükle
pip install -r requirements.txt
```

## Kullanım

### Yöntem 1: Tüm İşlemleri Tek Seferde (Önerilen)

```bash
python main.py
```

Bu komut şunları yapar:
- ✅ Veri ön işleme
- ✅ Modelleme
- ✅ Değerlendirme (confusion matrix, metrikler, overfitting)
- ✅ Rapor oluşturma

### Yöntem 2: GUI Uygulaması

```bash
streamlit run gui_app.py
```

Web tarayıcınızda otomatik olarak açılır.

## Sonuçlar

Tüm sonuçlar `results/` klasöründe:
- 📊 Grafikler: `*.png`
- 📈 Metrikler: `evaluation_metrics.csv`
- 🎯 En iyi model: `models/best_model.*`
- 📄 Rapor: `project_report.html`

## Dosya Yapısı

```
DataMind-LSTM/
├── dataset/              # Veri dosyaları
├── results/              # Sonuçlar (otomatik oluşturulur)
│   ├── models/          # Eğitilmiş modeller
│   └── *.png, *.csv    # Grafikler ve metrikler
├── data_preprocessing.py # Veri ön işleme
├── modeling.py          # Modelleme
├── evaluation.py        # Değerlendirme
├── reporting.py         # Rapor oluşturma
├── gui_app.py          # GUI uygulaması
└── main.py             # Ana script
```

## Sorun mu yaşıyorsunuz?

1. **Memory Error**: Batch size'ı küçültün
2. **Import Error**: `pip install -r requirements.txt` çalıştırın
3. **Veri Bulunamadı**: `dataset/` klasöründe CSV dosyalarının olduğundan emin olun

## Daha Fazla Bilgi

Detaylı bilgi için `README.md` dosyasına bakın.

