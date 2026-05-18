# 🚗 İkinci El Araç Fiyat Tahmin Otomasyonu

Bu proje, **Miuul Data Science Bootcamp** kapsamında geliştirilmiş, uçtan uca (End-to-End) çalışan makine öğrenmesi tabanlı bir araç fiyat tahmin uygulamasıdır. Projede nesne yönelimli ve modüler bir pipeline mimarisi kullanılmış, ön yüz entegrasyonu ise **Streamlit** ile sağlanmıştır.

---

## 📊 Proje Klasör Yapısı ve Modüller

Proje, kodun sürdürülebilirliği ve üretim (production) standartlarına uygunluğu açısından 4 ana parçaya bölünmüştür:

```text
arac_projesi/
│
├── veri_tanima_temizleme.py  # Veri üretimi, eksik/aykırı değer yönetimi
├── ozellik_muhendisligi.py   # NEW_ değişkenlerinin türetilmesi
├── model_egitim.py          # Encoding, RF Eğitimi ve Karşılaştırma
└── app.py                   # Streamlit Frontend arayüzü


