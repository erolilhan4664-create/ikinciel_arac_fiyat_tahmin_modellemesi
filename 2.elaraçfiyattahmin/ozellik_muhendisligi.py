import pandas as pd


def ozellik_muhendisligi_ekle(df):
    """Veri setine en az 5 yeni analitik değişken ekler."""

    df["NEW_Yas"] = 2026 - df["Yil"]

    df["NEW_Yillik_KM"] = df["KM"] / (df["NEW_Yas"] + 1)

    df["NEW_Luks_Segment"] = df["Marka"].apply(
        lambda x: 1 if x in ["BMW", "Mercedes-Benz"] else 0
    )

    df["NEW_KM_Durumu"] = pd.cut(
        df["KM"],
        bins=[-1, 50000, 150000, 9999999],
        labels=["Dusuk", "Orta", "Yuksek"],
    )

    df["NEW_Yas_KM_Etkisi"] = df["NEW_Yas"] * df["KM"]

    return df


def tekli_veri_ozellik_turet(secilen_marka, secilen_yil, secilen_km):
    kullanici_yas = 2026 - secilen_yil
    kullanici_yillik_km = secilen_km / (kullanici_yas + 1)
    kullanici_luks = 1 if secilen_marka in ["BMW", "Mercedes-Benz"] else 0

    if secilen_km <= 50000:
        kullanici_km_durumu = "Dusuk"
    elif secilen_km <= 150000:
        kullanici_km_durumu = "Orta"
    else:
        kullanici_km_durumu = "Yuksek"

    kullanici_yas_km = kullanici_yas * secilen_km

    return {
        "NEW_Yas": kullanici_yas,
        "NEW_Yillik_KM": kullanici_yillik_km,
        "NEW_Luks_Segment": kullanici_luks,
        "NEW_KM_Durumu": kullanici_km_durumu,
        "NEW_Yas_KM_Etkisi": kullanici_yas_km,
    }