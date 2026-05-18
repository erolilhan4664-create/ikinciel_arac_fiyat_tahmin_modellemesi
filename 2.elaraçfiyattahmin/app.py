import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


from model_egitim import model_hazirlik_ve_egitim
from ozellik_muhendisligi import (
    ozellik_muhendisligi_ekle,
    tekli_veri_ozellik_turet,
)
from veri_tanima_temizleme import (
    veri_temizleme_isleme,
    veri_ureti_ve_havuz_getir,
)


st.set_page_config(
    page_title="Miuul İkinci El Araç Fiyat Tahmin Projesi",
    layout="wide",
    page_icon="🚗",
)


@st.cache_resource
def uctan_uca_pipeline_akis():
    df_ham, ARAC_HAVUZU, YAKIT_TURU, VITES_TURU, RENKLER, SEHIRLER = (
        veri_ureti_ve_havuz_getir()
    )
    df_temiz = veri_temizleme_isleme(df_ham)
    df_muhendislik = ozellik_muhendisligi_ekle(df_temiz)
    model, model_sutunlari, metrikler = model_hazirlik_ve_egitim(df_muhendislik)
    return (
        model,
        model_sutunlari,
        metrikler,
        ARAC_HAVUZU,
        YAKIT_TURU,
        VITES_TURU,
        RENKLER,
        SEHIRLER,
        df_muhendislik,
    )



(
    model,
    model_sutunlari,
    metrikler,
    ARAC_HAVUZU,
    YAKIT_TURU,
    VITES_TURU,
    RENKLER,
    SEHIRLER,
    ana_df,
) = uctan_uca_pipeline_akis()


st.title("🚗 İkinci El Araç Fiyat Tahmin Otomasyonu")
st.write(
    "Miuul Data Science Bootcamp Projesi kapsamında Nesne Yönelimli ve Modüler Pipeline mimarisi kullanılarak optimize edilmiştir."
)
st.markdown("---")

sekme1, sekme2 = st.tabs(
    ["🔮 Fiyat Tahmini Yap", "📊 Proje Veri Analizi & Model Karşılaştırma"]
)


with sekme1:
    col1, col2 = st.columns(2)

    with col1:
        secilen_marka = st.selectbox("Araç Markası", list(ARAC_HAVUZU.keys()))
        secilen_model = st.selectbox("Araç Modeli", ARAC_HAVUZU[secilen_marka])
        secilen_yil = st.slider("Araç Yılı", 2010, 2025, 2018)
        secilen_km = st.number_input(
            "Kilometre (KM)", min_value=0, max_value=500000, value=100000, step=5000
        )

    with col2:
        secilen_vites = st.selectbox("Vites Türü", VITES_TURU)
        secilen_yakit = st.selectbox("Yakıt Türü", YAKIT_TURU)
        secilen_renk = st.selectbox("Renk", RENKLER)
        secilen_sehir = st.selectbox("Şehir", SEHIRLER)

    st.markdown("---")

    if st.button("💰 Araç Fiyatını Tahmin Et", use_container_width=True):

        yeni_ozellikler = tekli_veri_ozellik_turet(
            secilen_marka, secilen_yil, secilen_km
        )

        kullanici_verisi = pd.DataFrame(
            [
                {
                    "Marka": secilen_marka,
                    "Model": secilen_model,
                    "Yil": secilen_yil,
                    "KM": secilen_km,
                    "Vites": secilen_vites,
                    "Yakit": secilen_yakit,
                    "Renk": secilen_renk,
                    "Sehir": secilen_sehir,
                    **yeni_ozellikler,
                }
            ]
        )

        kullanici_encoded = pd.get_dummies(kullanici_verisi)
        kullanici_encoded = kullanici_encoded.reindex(
            columns=model_sutunlari, fill_value=0
        )

        tahmin_fiyat = model.predict(kullanici_encoded)[0]
        tahmin_fiyat = max(360750, tahmin_fiyat)

        st.success(f"### 🎯 Tahmini Araç Değeri: **{tahmin_fiyat:,.2f} TL**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Minimum Piyasa Değeri", f"{tahmin_fiyat * 0.95:,.0f} TL")
        m2.metric("Önerilen İlan Fiyatı", f"{tahmin_fiyat:,.0f} TL")
        m3.metric("Maksimum Piyasa Değeri", f"{tahmin_fiyat * 1.05:,.0f} TL")
        st.balloons()


with sekme2:
    st.subheader("📊 Otomatik Keşifçi Veri Analizi (EDA)")
    c_grafik1, c_grafik2 = st.columns(2)

    with c_grafik1:
        fig_hist = px.histogram(
            ana_df,
            x="Fiyat_TL",
            title="Veri Setindeki Araç Fiyat Dağılımı",
            color_discrete_sequence=["#228B22"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c_grafik2:
        fig_scatter = px.scatter(
            ana_df,
            x="Yil",
            y="Fiyat_TL",
            color="Marka",
            title="Yıllara Göre Fiyat Değişimi ve Marka Kırılımı",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    st.subheader("🤖 Model Karşılaştırma Başarımı")
    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Random Forest $R^2$", f"{metrikler['RF_R2']:.4f}")
    met2.metric("Random Forest RMSE", f"{metrikler['RF_RMSE']:,.2f} TL")
    met3.metric("Linear Reg. $R^2$", f"{metrikler['LR_R2']:.4f}")
    met4.metric(
        "Linear Reg. RMSE",
        f"{metrikler['LR_RMSE']:,.2f} TL",
        delta=f"{metrikler['LR_RMSE'] - metrikler['RF_RMSE']:,.0f} Hata",
        delta_color="inverse",
    )

    st.markdown("---")
    st.subheader("⚡ Model Karv Mekanizması (Feature Importance)")

    onem_dereceleri = model.feature_importances_
    onem_df = pd.DataFrame(
        {"Ozellik": model_sutunlari, "Onem Skoru": onem_dereceleri}
    )
    onem_df = onem_df.sort_values(by="Onem Skoru", ascending=False).head(8)

    fig_bar = px.bar(
        onem_df,
        x="Onem Skoru",
        y="Ozellik",
        orientation="h",
        title="Modelin En Çok Önem Verdiği Değişkenler (Türetilenler Dahil)",
        color="Onem Skoru",
        color_continuous_scale="Viridis",
    )
    fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_bar, use_container_width=True)