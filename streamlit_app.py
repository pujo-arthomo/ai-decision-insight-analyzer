import streamlit as st
import pandas as pd
import re
import os
from groq import Groq
from datetime import datetime

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="AI Decision Insight Analyzer",
    page_icon="🧠",
    layout="wide"
)

NEGATIVE_KEYWORDS = [
    "pecah", "rusak", "lama", "kecewa", "retak",
    "tidak sesuai", "jelek", "parah", "buruk"
]

MIN_LOW_INFO_LEN = 10


# ======================================================
# DATA QUALITY CORE (INTERNAL)
# ======================================================
def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def analyze_data(df):
    df["review_clean"] = df["review"].apply(clean_text)

    total = len(df)

    low_info = df[df["review_clean"].str.len() < MIN_LOW_INFO_LEN]

    mismatch = df[
        (df["rating"] >= 4) &
        (df["review_clean"].str.contains("|".join(NEGATIVE_KEYWORDS)))
    ]

    repetitive_pct = (
        df["review_clean"].value_counts()
        .gt(1)
        .sum() / total * 100
    )

    return {
        "total_reviews": total,
        "low_info_pct": round(len(low_info) / total * 100, 2),
        "rating_mismatch_pct": round(len(mismatch) / total * 100, 2),
        "repetitive_pct": round(repetitive_pct, 2),
    }


# ======================================================
# DECISION INSIGHT (GROQ)
# ======================================================
def generate_decision_insight(stats):
    api_key = st.secrets["GROQ_API_KEY"]
    if not api_key:
        return (
            "⚠️ **AI Insight tidak dapat dihasilkan** karena API key tidak tersedia.\n\n"
            "Namun, secara umum:\n"
            "- Rating dan isi ulasan sering tidak sejalan\n"
            "- Data berisiko menyesatkan keputusan produk dan AI\n"
            "- Perlu pemisahan feedback produk vs layanan"
        )

    client = Groq(api_key=api_key)

    prompt = f"""
Anda adalah AI Decision & Product Insight Advisor.

Ringkasan temuan dari analisis ulasan produk e-commerce:

- Total ulasan dianalisis: {stats['total_reviews']}
- Rating–text mismatch: {stats['rating_mismatch_pct']}%
- Ulasan minim informasi: {stats['low_info_pct']}%
- Pola ulasan berulang: {stats['repetitive_pct']}%

Konteks penggunaan:
Data ini dipakai untuk pengambilan keputusan bisnis dan pelatihan AI.

Tugas Anda:
1. Jelaskan makna temuan ini untuk pengambil keputusan non-teknis.
2. Sebutkan 3 risiko utama jika data digunakan tanpa perbaikan.
3. Berikan 3 rekomendasi keputusan praktis (1–2 minggu).
4. Tentukan satu prioritas utama yang paling masuk akal.

Gunakan bahasa profesional, ringkas, dan berbentuk narasi.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content


# ======================================================
# STREAMLIT UI (DECISION-MAKER MODE)
# ======================================================
st.title("🧠 AI Decision Insight Analyzer")
st.write(
    "Unggah file CSV ulasan produk untuk mendapatkan **insight naratif** "
    "yang siap digunakan pengambil keputusan."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # --- Auto-detect kolom ---
        REVIEW_COLS = ["review", "review_text", "content", "ulasan", "comment"]
        RATING_COLS = ["rating", "star", "score", "overall_rating"]

        review_col = next((c for c in REVIEW_COLS if c in df.columns), None)
        rating_col = next((c for c in RATING_COLS if c in df.columns), None)

        if not review_col or not rating_col:
            st.error(
                "CSV tidak sesuai.\n\n"
                "Diperlukan kolom ulasan dan rating.\n"
                f"Kolom ditemukan: {list(df.columns)}"
            )
            st.stop()

        df = df.rename(columns={
            review_col: "review",
            rating_col: "rating"
        })

        st.success(f"File berhasil dimuat ({len(df)} baris)")

        if st.button("Generate Decision Insight"):
            with st.spinner("Menganalisis dan menyusun insight keputusan..."):
                stats = analyze_data(df)
                insight = generate_decision_insight(stats)

            st.markdown("---")
            st.markdown(
                f"### 📊 Decision Insight Report\n"
                f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n"
            )
            st.markdown(insight)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
