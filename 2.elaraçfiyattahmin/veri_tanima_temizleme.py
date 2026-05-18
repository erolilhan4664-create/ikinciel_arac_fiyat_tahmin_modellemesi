import numpy as np
import pandas as pd
import random
from faker import Faker


def veri_ureti_ve_havuz_getir():
    fake = Faker("tr_TR")

    ARAC_HAVUZU = {
        "Volkswagen": ["Passat", "Golf", "Polo", "Tiguan"],
        "Renault": ["Clio", "Megane", "Fluence", "Symbol"],
        "Fiat": ["Egea", "Linea", "Punto", "Fiorino"],
        "Toyota": ["Corolla", "Yaris", "Aurais"],
        "BMW": ["3 Serisi", "5 Serisi", "1 Serisi"],
        "Mercedes-Benz": ["C Serisi", "E Serisi", "A Serisi"],
        "Hyundai": ["i20", "i30", "Accent Blue", "Tucson"],
    }
    YAKIT_TURU = ["Benzin", "Dizel", "LPG & Benzin", "Hibrit", "Elektrik"]
    VITES_TURU = ["Manuel", "Otomatik", "Yarı Otomatik"]
    RENKLER = [
        "Beyaz",
        "Siyah",
        "Gri (Gümüş)",
        "Füme",
        "Kırmızı",
        "Mavi",
        "Metalik Gri",
    ]
    SEHIRLER = [
        "İstanbul",
        "Ankara",
        "İzmir",
        "Bursa",
        "Antalya",
        "Kocaeli",
        "Adana",
        "Gaziantep",
    ]

    veri_listesi = []
    for i in range(3000):
        marka = random.choice(list(ARAC_HAVUZU.keys()))
        model = random.choice(ARAC_HAVUZU[marka])
        yil = random.randint(2010, 2025)
        yas = 2026 - yil
        km = max(
            random.randint(5000, 35000),
            (yas * random.randint(10000, 20000))
            + random.randint(-10000, 20000),
        )

        baz_fiyat = 527250
        if marka in ["BMW", "Mercedes-Benz"]:
            baz_fiyat = 1776000
        elif marka in ["Volkswagen", "Toyota"]:
            baz_fiyat = 971250

        if model in ["Passat", "5 Serisi", "E Serisi"]:
            baz_fiyat += 416250
        elif model in ["Tiguan", "Tucson"]:
            baz_fiyat += 333000
        elif model in ["Golf", "Megane", "Corolla", "3 Serisi", "C Serisi"]:
            baz_fiyat += 194250

        fiyat = baz_fiyat + (yil - 2010) * 52725 - (km * 1.8)
        fiyat = max(360750, int(fiyat))

        veri_listesi.append(
            {
                "Marka": marka,
                "Model": model,
                "Yil": yil,
                "KM": km,
                "Vites": random.choice(VITES_TURU),
                "Yakit": random.choice(YAKIT_TURU),
                "Renk": random.choice(RENKLER),
                "Sehir": random.choice(SEHIRLER),
                "Fiyat_TL": fiyat,
            }
        )

    df = pd.DataFrame(veri_listesi)
    return df, ARAC_HAVUZU, YAKIT_TURU, VITES_TURU, RENKLER, SEHIRLER


def veri_temizleme_isleme(df):

    df["KM"] = df["KM"].fillna(df["KM"].median())


    q1 = df["KM"].quantile(0.05)
    q3 = df["KM"].quantile(0.95)
    df["KM"] = np.clip(df["KM"], q1, q3)

    return df