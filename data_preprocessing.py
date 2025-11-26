"""
Veri Ön İşleme Modülü
Bu modül veri temizleme, normalizasyon, encoding ve korelasyon analizi işlemlerini gerçekleştirir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Türkçe karakter desteği için matplotlib ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

class DataPreprocessor:
    """
    Veri ön işleme sınıfı
    """
    
    def __init__(self, data_dir='dataset', results_dir='results'):
        """
        Args:
            data_dir: Veri klasörü yolu
            results_dir: Sonuçların kaydedileceği klasör
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.df = None
        self.df_cleaned = None
        self.scaler = None
        self.label_encoders = {}
        
        # Results klasörünü oluştur
        os.makedirs(results_dir, exist_ok=True)
    
    def load_data(self):
        """
        Veri dosyalarını yükler ve birleştirir
        """
        print("=" * 60)
        print("1. VERİ YÜKLEME")
        print("=" * 60)
        
        # İlk dosya (Türkçe kolon isimleri)
        file1_path = os.path.join(self.data_dir, '55677c66-5743-453b-83f0-d5153a92ade1.csv')
        df1 = pd.read_csv(file1_path, encoding='utf-8')
        print(f"✓ Dosya 1 yüklendi: {df1.shape[0]} satır, {df1.shape[1]} kolon")
        
        # İkinci dosya (İngilizce kolon isimleri)
        file2_path = os.path.join(self.data_dir, '8fbe5a82-b16e-485e-9e7c-7145596828e6.csv')
        df2 = pd.read_csv(file2_path, encoding='utf-8')
        print(f"✓ Dosya 2 yüklendi: {df2.shape[0]} satır, {df2.shape[1]} kolon")
        
        # Kolon isimlerini standardize et
        # Dosya 1 kolonları: Ad Soyad, Maktülün yaşı, İl/ilçe, Tarih, Neden öldürüldü, 
        #                     Kim tarafından öldürüldü, Öldürülme şekli, Failin durumu
        # Dosya 2 kolonları: city, age, date, protectionorder, why1, killer1, killingway1, statusofkiller, year
        
        # Dosya 1'i standardize et
        df1_renamed = df1.rename(columns={
            'Ad Soyad': 'name',
            'Maktülün yaşı': 'age',
            'İl/ilçe': 'city',
            'Tarih': 'date',
            'Neden öldürüldü': 'reason',
            'Kim tarafından öldürüldü': 'killer',
            'Öldürülme şekli': 'killing_method',
            'Failin durumu': 'killer_status'
        })
        
        # Dosya 2'yi standardize et
        df2_renamed = df2.rename(columns={
            'why1': 'reason',
            'killer1': 'killer',
            'killingway1': 'killing_method',
            'statusofkiller': 'killer_status'
        })
        
        # Ortak kolonları seç
        common_cols = ['age', 'city', 'date', 'reason', 'killer', 'killing_method', 'killer_status']
        
        # Dosya 1'den ortak kolonları al
        df1_selected = df1_renamed[common_cols].copy()
        df1_selected['source'] = 'file1'
        
        # Dosya 2'den ortak kolonları al
        df2_selected = df2_renamed[common_cols].copy()
        df2_selected['source'] = 'file2'
        
        # Birleştir
        self.df = pd.concat([df1_selected, df2_selected], ignore_index=True)
        print(f"✓ Veriler birleştirildi: Toplam {self.df.shape[0]} satır, {self.df.shape[1]} kolon")
        
        # İlk 5 satırı göster
        print("\nİlk 5 satır:")
        print(self.df.head())
        
        # Veri tiplerini göster
        print("\nVeri tipleri:")
        print(self.df.dtypes)
        
        return self.df
    
    def clean_data(self):
        """
        Veri temizleme işlemleri
        """
        print("\n" + "=" * 60)
        print("2. VERİ TEMİZLEME")
        print("=" * 60)
        
        self.df_cleaned = self.df.copy()
        initial_rows = len(self.df_cleaned)
        
        # 1. Eksik veri analizi
        print("\n2.1. Eksik Veri Analizi:")
        missing_data = self.df_cleaned.isnull().sum()
        missing_percent = (missing_data / len(self.df_cleaned)) * 100
        missing_df = pd.DataFrame({
            'Eksik Sayı': missing_data,
            'Yüzde': missing_percent
        })
        print(missing_df[missing_df['Eksik Sayı'] > 0])
        
        # 2. Boşlukları temizle
        print("\n2.2. Boşluk Temizleme:")
        for col in self.df_cleaned.select_dtypes(include=['object']).columns:
            self.df_cleaned[col] = self.df_cleaned[col].astype(str).str.strip()
            # Boş string'leri NaN'a çevir
            self.df_cleaned[col] = self.df_cleaned[col].replace(['', 'nan', 'None', 'nan'], np.nan)
        
        # 3. Tarih kolonunu işle
        print("\n2.3. Tarih İşleme:")
        # Birden fazla tarih formatını dene
        date_formats = ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d.%m.%Y']
        self.df_cleaned['date'] = pd.to_datetime(self.df_cleaned['date'], errors='coerce', format='%d/%m/%Y', dayfirst=True)
        # Eğer parse edilemediyse, diğer formatları dene
        if self.df_cleaned['date'].isna().sum() > 0:
            for fmt in date_formats[1:]:
                mask = self.df_cleaned['date'].isna()
                if mask.sum() > 0:
                    self.df_cleaned.loc[mask, 'date'] = pd.to_datetime(
                        self.df_cleaned.loc[mask, 'date'], errors='coerce', format=fmt
                    )
        
        self.df_cleaned['year'] = self.df_cleaned['date'].dt.year
        self.df_cleaned['month'] = self.df_cleaned['date'].dt.month
        self.df_cleaned['day'] = self.df_cleaned['date'].dt.day
        
        # Parse edilemeyen tarih sayısını raporla
        invalid_dates = self.df_cleaned['date'].isna().sum()
        if invalid_dates > 0:
            print(f"  ⚠ {invalid_dates} geçersiz tarih bulundu (NaN olarak işaretlendi)")
        else:
            print(f"  ✓ Tüm tarihler başarıyla parse edildi")
        
        # 4. Yaş kolonunu işle
        print("\n2.4. Yaş İşleme:")
        # "Reşit", "Reşit Değil", "Bilinmiyor" gibi değerleri işle
        def process_age(age_str):
            if pd.isna(age_str):
                return np.nan
            age_str = str(age_str).strip()
            if age_str == 'Reşit':
                return 18  # Reşit yaşını 18 olarak kodla
            elif age_str == 'Reşit Değil':
                return 17  # Reşit değil yaşını 17 olarak kodla
            elif age_str == 'Bilinmiyor' or age_str == 'Tespit Edilemeyen':
                return np.nan
            else:
                try:
                    return int(float(age_str))
                except:
                    return np.nan
        
        self.df_cleaned['age_numeric'] = self.df_cleaned['age'].apply(process_age)
        
        # 5. Aykırı değer tespiti (yaş için)
        print("\n2.5. Aykırı Değer Analizi (Yaş):")
        if 'age_numeric' in self.df_cleaned.columns:
            Q1 = self.df_cleaned['age_numeric'].quantile(0.25)
            Q3 = self.df_cleaned['age_numeric'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df_cleaned[(self.df_cleaned['age_numeric'] < lower_bound) | 
                                      (self.df_cleaned['age_numeric'] > upper_bound)]
            print(f"✓ Tespit edilen aykırı değer sayısı: {len(outliers)}")
            print(f"  Alt sınır: {lower_bound:.2f}, Üst sınır: {upper_bound:.2f}")
            
            # Aykırı değerleri sınırla (silme yerine)
            self.df_cleaned.loc[self.df_cleaned['age_numeric'] < lower_bound, 'age_numeric'] = lower_bound
            self.df_cleaned.loc[self.df_cleaned['age_numeric'] > upper_bound, 'age_numeric'] = upper_bound
        
        # 6. Tekrarlanan satırları kaldır
        print("\n2.6. Tekrarlanan Satırları Kaldırma:")
        duplicates = self.df_cleaned.duplicated().sum()
        print(f"✓ Tekrarlanan satır sayısı: {duplicates}")
        self.df_cleaned = self.df_cleaned.drop_duplicates()
        
        print(f"\n✓ Temizleme tamamlandı: {initial_rows} -> {len(self.df_cleaned)} satır")
        
        return self.df_cleaned
    
    def encode_categorical(self):
        """
        Kategorik değişkenleri encode et
        """
        print("\n" + "=" * 60)
        print("3. ENCODING İŞLEMLERİ")
        print("=" * 60)
        
        categorical_cols = ['city', 'reason', 'killer', 'killing_method', 'killer_status']
        
        for col in categorical_cols:
            if col in self.df_cleaned.columns:
                le = LabelEncoder()
                # NaN değerleri 'Unknown' ile doldur
                self.df_cleaned[col + '_encoded'] = self.df_cleaned[col].fillna('Unknown')
                self.df_cleaned[col + '_encoded'] = le.fit_transform(self.df_cleaned[col + '_encoded'])
                self.label_encoders[col] = le
                print(f"✓ {col} kolonu encode edildi: {len(le.classes_)} kategori")
        
        return self.df_cleaned
    
    def normalize_data(self):
        """
        Veriyi normalize et/ölçekle
        """
        print("\n" + "=" * 60)
        print("4. NORMALİZASYON/ÖLÇEKLEME")
        print("=" * 60)
        
        # Sayısal kolonları seç
        numeric_cols = ['year', 'month', 'day', 'age_numeric']
        numeric_cols = [col for col in numeric_cols if col in self.df_cleaned.columns]
        
        # StandardScaler kullan
        self.scaler = StandardScaler()
        self.df_cleaned[numeric_cols] = self.scaler.fit_transform(self.df_cleaned[numeric_cols].fillna(0))
        
        print(f"✓ {len(numeric_cols)} sayısal kolon normalize edildi")
        print(f"  Normalize edilen kolonlar: {numeric_cols}")
        
        return self.df_cleaned
    
    def correlation_analysis(self):
        """
        Korelasyon analizi yap
        """
        print("\n" + "=" * 60)
        print("5. KORELASYON ANALİZİ")
        print("=" * 60)
        
        # Sayısal ve encode edilmiş kolonları seç
        numeric_cols = ['year', 'month', 'day', 'age_numeric']
        encoded_cols = [col for col in self.df_cleaned.columns if col.endswith('_encoded')]
        
        corr_cols = [col for col in numeric_cols + encoded_cols if col in self.df_cleaned.columns]
        
        correlation_matrix = self.df_cleaned[corr_cols].corr()
        
        # Korelasyon matrisini görselleştir
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
        plt.title('Korelasyon Matrisi (Correlation Matrix)', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Korelasyon matrisi kaydedildi: results/correlation_matrix.png")
        
        # Yüksek korelasyonlu değişkenleri bul
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            print("\nYüksek korelasyonlu değişkenler (>0.7):")
            for pair in high_corr_pairs:
                print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
        else:
            print("\n✓ Yüksek korelasyonlu değişken bulunamadı")
        
        return correlation_matrix
    
    def create_visualizations(self):
        """
        Veri görselleştirme grafikleri oluştur
        """
        print("\n" + "=" * 60)
        print("6. VERİ GÖRSELLEŞTİRME")
        print("=" * 60)
        
        # 1. Yaş dağılımı
        if 'age_numeric' in self.df_cleaned.columns:
            plt.figure(figsize=(10, 6))
            self.df_cleaned['age_numeric'].hist(bins=30, edgecolor='black')
            plt.xlabel('Yaş (Age)')
            plt.ylabel('Frekans (Frequency)')
            plt.title('Yaş Dağılımı (Age Distribution)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'age_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Yaş dağılımı grafiği kaydedildi")
        
        # 2. Yıllara göre dağılım
        if 'year' in self.df_cleaned.columns:
            plt.figure(figsize=(12, 6))
            year_counts = self.df_cleaned['year'].value_counts().sort_index()
            plt.bar(year_counts.index, year_counts.values, edgecolor='black')
            plt.xlabel('Yıl (Year)')
            plt.ylabel('Sayı (Count)')
            plt.title('Yıllara Göre Dağılım (Distribution by Year)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'year_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Yıl dağılımı grafiği kaydedildi")
        
        # 3. Şehir dağılımı (Top 15)
        if 'city' in self.df_cleaned.columns:
            plt.figure(figsize=(12, 8))
            city_counts = self.df_cleaned['city'].value_counts().head(15)
            plt.barh(range(len(city_counts)), city_counts.values, edgecolor='black')
            plt.yticks(range(len(city_counts)), city_counts.index)
            plt.xlabel('Sayı (Count)')
            plt.title('En Çok Olay Görülen Şehirler (Top 15 Cities)')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'city_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Şehir dağılımı grafiği kaydedildi")
        
        # 4. Öldürülme şekli dağılımı
        if 'killing_method' in self.df_cleaned.columns:
            plt.figure(figsize=(12, 8))
            method_counts = self.df_cleaned['killing_method'].value_counts().head(10)
            plt.barh(range(len(method_counts)), method_counts.values, edgecolor='black')
            plt.yticks(range(len(method_counts)), method_counts.index)
            plt.xlabel('Sayı (Count)')
            plt.title('Öldürülme Şekli Dağılımı (Killing Method Distribution)')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'killing_method_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Öldürülme şekli dağılımı grafiği kaydedildi")
        
        # 5. Fail durumu dağılımı (geliştirilmiş)
        if 'killer_status' in self.df_cleaned.columns:
            # Yatay bar chart ile göster; küçük kategorileri 'Diğer' altında birleştir
            status_counts = self.df_cleaned['killer_status'].value_counts()
            total = int(status_counts.sum())

            # Eşik: yüzde bazında 1%'den az olanları 'Diğer' altında topla
            pct = status_counts / total
            threshold = 0.01
            major = status_counts[pct >= threshold].copy()
            minor_sum = int(status_counts[pct < threshold].sum())
            if minor_sum > 0:
                major['Diğer'] = minor_sum

            major = major.sort_values(ascending=True)

            plt.figure(figsize=(10, max(4, 0.4 * len(major))))
            sns.barplot(x=major.values, y=major.index, palette='viridis')

            # Annotate counts and percentages
            max_val = major.values.max() if len(major) > 0 else 0
            for i, v in enumerate(major.values):
                percent = v / total * 100
                plt.text(v + max(1, max_val * 0.01), i, f"{v} ({percent:.1f}%)", va='center')

            plt.xlabel('Sayı (Count)')
            plt.title('Fail Durumu Dağılımı (Killer Status Distribution)')
            plt.tight_layout()
            out_path = os.path.join(self.results_dir, 'killer_status_distribution_improved.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Fail durumu dağılımı grafiği kaydedildi (improved): {out_path}")
        
        # 6. Box plot - Yaş dağılımı
        if 'age_numeric' in self.df_cleaned.columns:
            plt.figure(figsize=(8, 6))
            self.df_cleaned.boxplot(column='age_numeric')
            plt.ylabel('Yaş (Age)')
            plt.title('Yaş Dağılımı Box Plot')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'age_boxplot.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Yaş box plot grafiği kaydedildi")
        
        print(f"\n✓ Tüm görseller {self.results_dir} klasörüne kaydedildi")
    
    def create_train_test_split(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Train/validation/test split oluştur
        """
        print("\n" + "=" * 60)
        print("7. TRAIN/VALIDATION/TEST SPLIT")
        print("=" * 60)
        
        # Özellik kolonlarını seç
        feature_cols = []
        
        # Sayısal kolonlar
        numeric_cols = ['year', 'month', 'day', 'age_numeric']
        feature_cols.extend([col for col in numeric_cols if col in self.df_cleaned.columns])
        
        # Encode edilmiş kolonlar
        encoded_cols = [col for col in self.df_cleaned.columns if col.endswith('_encoded')]
        feature_cols.extend(encoded_cols)
        
        # Eksik değerleri doldur
        X = self.df_cleaned[feature_cols].fillna(0)
        
        # Hedef değişken: killer_status (fail durumu) - en önemli sınıflandırma hedefi
        if 'killer_status_encoded' in self.df_cleaned.columns:
            y = self.df_cleaned['killer_status_encoded']
        else:
            # Eğer encode edilmemişse, encode et
            le = LabelEncoder()
            y = le.fit_transform(self.df_cleaned['killer_status'].fillna('Unknown'))
            self.label_encoders['killer_status'] = le
        
        # Eğer bazı sınıflarda çok az örnek varsa stratify kullanmak hata verir.
        # Bu durumda uyarı verip stratify=None ile bölme yapıyoruz.
        try:
            class_counts = pd.Series(y).value_counts()
            if class_counts.min() < 2:
                print("  ⚠ Dikkat: Bazı sınıflarda çok az örnek var; stratify kapatılıyor (stratify=None).")
                stratify_first = None
                stratify_second = None
            else:
                stratify_first = y
                stratify_second = None  # belirlenecek sonrası için

        except Exception:
            # Eğer y üzerinde sayımda bir hata olursa, güvenli tarafta kal
            stratify_first = None
            stratify_second = None

        # İlk olarak train+val ve test'e ayır
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_first
        )

        # Sonra train ve validation'a ayır
        val_size_adjusted = val_size / (1 - test_size)
        # Stratify için y_temp kullanmak istiyorsak ve ilk stratify kullanıldıysa
        if stratify_first is not None:
            stratify_second = y_temp
        else:
            stratify_second = None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=stratify_second
        )
        
        print(f"✓ Train seti: {X_train.shape[0]} örnek, {X_train.shape[1]} özellik")
        print(f"✓ Validation seti: {X_val.shape[0]} örnek, {X_val.shape[1]} özellik")
        print(f"✓ Test seti: {X_test.shape[0]} örnek, {X_test.shape[1]} özellik")
        print(f"✓ Toplam özellik sayısı: {len(feature_cols)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
    
    def save_processed_data(self):
        """
        İşlenmiş veriyi kaydet
        """
        output_path = os.path.join(self.results_dir, 'processed_data.csv')
        self.df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✓ İşlenmiş veri kaydedildi: {output_path}")
        return output_path


def main():
    """
    Ana işlem fonksiyonu
    """
    preprocessor = DataPreprocessor()
    
    # 1. Veri yükleme
    preprocessor.load_data()
    
    # 2. Veri temizleme
    preprocessor.clean_data()
    
    # 3. Encoding
    preprocessor.encode_categorical()
    
    # 4. Normalizasyon
    preprocessor.normalize_data()
    
    # 5. Korelasyon analizi
    preprocessor.correlation_analysis()
    
    # 6. Görselleştirme
    preprocessor.create_visualizations()
    
    # 7. Train/test split
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = preprocessor.create_train_test_split()
    
    # 8. İşlenmiş veriyi kaydet
    preprocessor.save_processed_data()
    
    # Sonuçları döndür
    return {
        'preprocessor': preprocessor,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_cols': feature_cols
    }


if __name__ == "__main__":
    results = main()
    print("\n" + "=" * 60)
    print("VERİ ÖN İŞLEME TAMAMLANDI!")
    print("=" * 60)

