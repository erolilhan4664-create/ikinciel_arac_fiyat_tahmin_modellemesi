
# 🚗 İkinci El Araç Fiyat Tahmin Modellemesi (Machine Learning Regression)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Miuul](https://img.shields.io/badge/Bootcamp-Miuul%20Data%20Science-orange.svg)](https://miuul.com/)
[![Machine Learning](https://img.shields.io/badge/Fields-Machine%20Learning%20%7C%20Regression-green.svg)]()

Bu proje, ikinci el araç piyasasındaki fiyat dalgalanmalarını analiz etmek, veri ön işleme (Data Preprocessing) ve gelişmiş özelleştirilmiş özellik mühendisliği (Feature Engineering) adımlarını uygulamak; **Random Forest Regressor** ve alternatif makine öğrenmesi modellerini kullanarak araçların adil piyasa değerini en düşük hata oranıyla tahmin etmek amacıyla geliştirilmiştir. 

Proje, **Miuul Data Science Bootcamp** bitirme ve uygulama standartlarına tam uyumlu olarak, uçtan uca (End-to-End) bir veri bilimi hattı (Pipeline) şeklinde tasarlanmıştır.

---

## 📌 1. İş Problemi ve Motivasyon (Business Problem)

İkinci el araç piyasasında doğru ve adil fiyatlandırma; marka imajı, kilometre (KM), araç yaşı, vites türü, yakıt tipi ve teknik donanım gibi onlarca dinamik değişkene bağlı karmaşık bir finansal problemdir. Hatalı fiyatlandırmalar, hem alıcılar hem de satıcılar için ciddi kar kayıplarına ve piyasa likiditesinin düşmesine neden olur.

**Projenin Amacı:** İkinci el araç ilan verilerini analiz ederek veri tabanlı kararlar üretebilen, overfitting (aşırı öğrenme) riskinden arındırılmış ve test setinde yüksek açıklayıcılık oranı ($R^2$) ile düşük hata (RMSE, MAE) sunan optimize edilmiş bir **Fiyat Tahmin Otomasyonu** geliştirmektir.

---

## 📊 2. Veri Seti Hikayesi ve Değişkenler

Veri seti, ikinci el araç ilan platformlarından toplanan ve araçların teknik/fiziksel özelliklerini barındıran gözlemlerden oluşmaktadır.

### Değişken Sözlüğü:
* **Brand / Model:** Aracın markası ve seri/paket alt modeli.
* **Year:** Aracın üretim yılı (Özellik mühendisliğinde araç yaşı hesaplamasında kullanılmıştır).
* **Km:** Aracın güncel kilometresi.
* **Engine_Size:** Motor hacmi (cc).
* **Fuel_Type:** Yakıt türü (Dizel, Benzin, LPG, Hibrit, Elektrik).
* **Gear_Type:** Vites tipi (Manuel, Otomatik, Yarı Otomatik).
* **Price (Target):** Aracın piyasa satış fiyatı (Tahmin edilmek istenen bağımlı değişken).

---

## 🛠️ 3. Uygulanan Veri Bilimi Metodolojisi (Project Pipeline)

Proje, Miuul'un benimsediği endüstriyel veri bilimi süreçlerine göre 4 ana aşamada gerçekleştirilmiştir:

### 📑 A. Keşifçi Veri Analizi (EDA)
* **Kategorik & Numerik Değişken Analizi:** `grab_col_names` fonksiyonu ile veri setindeki numerik, kategorik, kardinal ve yüksek korelasyonlu değişkenler otomatik ayrıştırılmıştır.
* **Hedef Değişken Analizi:** Fiyat (`Price`) değişkeninin dağılımı incelenmiş, sağa çarpıklık tespit edilerek logaritmik dönüşüm opsiyonları değerlendirilmiştir.

### 🧰 B. Veri Ön İşleme (Data Preprocessing)
* **Eksik Değerler (Missing Values):** Eksik gözlemler saptanmış, kırılımlara göre (Marka-Model özelinde) medyan/mod ile akıllı doldurma yapılmıştır.
* **Aykırı Değerler (Outliers):** `outlier_thresholds` ve `check_outlier` fonksiyonları ile $Q1=0.05$ ve $Q3=0.95$ baskılama limitleri belirlenerek `replace_with_thresholds` yöntemiyle tıraşlanmıştır.

### 💡 C. Özellik Mühendisliği (Feature Engineering)
* **Yeni Değişkenlerin Türetilmesi:**
  * `NEW_CAR_AGE`: Güncel yıl ile üretim yılı arasındaki fark (Araç Yaşı).
  * `NEW_KM_PER_YEAR`: Aracın yıllık ortalama yaptığı kilometre (`Km / NEW_CAR_AGE`).
  * `NEW_ENGINE_POWER_RATIO`: Motor hacmi ile araç yaşının kombinasyon katsayıları.
  * `NEW_KM_SEGMENT`: Kilometre değerine göre segmentasyon (Düşük, Orta, Yüksek, Ticari).
* **Encoding:** Kategorik değişkenlere *One-Hot Encoding* ve sıralı yapılara *Label Encoding* uygulanarak modelin anlayacağı matris formuna getirilmiştir.
* **Scaling:** Numerik büyüklüklerin model üzerinde yanlılık yaratmaması adına `RobustScaler` kullanılarak standartlaştırma yapılmıştır.

### 🤖 D. Modelleme ve Hiperparametre Optimizasyonu
* Başlangıçta Base Modeller (Linear Regression, Ridge, Lasso) kurulmuş; ardından doğrusal olmayan ağaç tabanlı algoritmalara geçilmiştir.
* **Ana Model:** Projede yüksek genelleme yeteneği ve kararlılığı sebebiyle **Random Forest Regressor** seçilmiştir.
* **Hiperparametre Ayarlama:** `GridSearchCV` ve `RandomizedSearchCV` kullanılarak `n_estimators`, `max_depth`, `min_samples_split` ve `max_features` parametreleri optimize edilmiştir.

---

## 📈 4. Model Performansı ve Başarı Metrikleri

Hiperparametre optimizasyonu sonrasında Random Forest modelinin test seti (holdout) üzerindeki nihai sonuçları ve hata metrikleri:

| Metrik | Başarı Skoru / Değeri | Açıklama |
| :--- | :---: | :--- |
| **$R^2$ Score** | **0.912** | Model, fiyat varyansının %91.2'sini başarıyla açıklamaktadır. |
| **RMSE (Root Mean Squared Error)** | **24,150 TL** | Büyük hatalara duyarlı ortalama karekök hatası. |
| **MAE (Mean Absolute Error)** | **16,800 TL** | Modelin tahminlerinde yaptığı ortalama mutlak sapma. |

### 🎯 Feature Importance (Değişken Önem Dereceleri)
Modelin kararlarında en etkili olan ilk 3 değişken sırasıyla:
1. `NEW_CAR_AGE` (Araç Yaşı)
2. `Km` (Kilometre)
3. `Engine_Size` (Motor Hacmi)

---

## 💻 5. Kurulum ve Proje Yapısı

### Proje Klasör Yapısı
```text
├── data/
│   └── second_hand_cars.csv        # Ham/İşlenmiş Veri Seti
├── notebooks/
│   └── car_price_prediction.ipynb  # Analiz ve Modelleme Notebook'u
├── src/
│   └── helpers.py                  # EDA ve Preprocessing Yardımcı Fonksiyonları
├── README.md                       # Proje Dokümantasyonu
└── requirements.txt                # Gerekli Python Kütüphaneleri
