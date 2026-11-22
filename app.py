import streamlit as st, pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime

XLSX = Path(__file__).parent / "Data_Uji_Kelompok6.xlsx"
ASPECTS = ['comfort', 'performance', 'fuel_economy', 'features', 'design_interior', 'price_value', 'reliability_maintenance']
LABEL = {'comfort': 'Kenyamanan', 'performance': 'Performa', 'fuel_economy': 'Efisiensi BBM', 'features': 'Fitur', 'design_interior': 'Desain/Interior', 'price_value': 'Harga/Value', 'reliability_maintenance': 'Keandalan/Perawatan'}
PRESETS = {'Seimbang': {'comfort': 1, 'performance': 1, 'fuel_economy': 1, 'features': 1, 'design_interior': 1, 'price_value': 1, 'reliability_maintenance': 1}, 'Keluarga hemat': {'comfort': 0.25, 'fuel_economy': 0.25, 'price_value': 0.2, 'reliability_maintenance': 0.15, 'features': 0.05, 'performance': 0.05, 'design_interior': 0.05}, 'Performa': {'performance': 0.4, 'comfort': 0.15, 'features': 0.15, 'design_interior': 0.1, 'fuel_economy': 0.05, 'price_value': 0.05, 'reliability_maintenance': 0.1}, 'Default v1': {'comfort': 0.25, 'performance': 0.15, 'fuel_economy': 0.2, 'features': 0.1, 'design_interior': 0.1, 'price_value': 0.1, 'reliability_maintenance': 0.1}}
SOURCE_NOTE = 'Data ulasan berasal dari komentar YouTube melalui YouTube Data API.'
MIN_OPINION = 5
PRICE_OPTS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]   # 1000 = "Rp1 miliar+" (tanpa batas atas)

st.set_page_config(page_title="Rekomendasi Mobil", page_icon="🚗", layout="wide")
st.title("🚗 Rekomendasi Mobil dari Ulasan Konsumen")
st.caption(SOURCE_NOTE + " Skor aspek berasal dari sentimen komentar berbahasa Indonesia; mobil diurutkan dengan TOPSIS (MCDM).")

if not XLSX.exists():
    st.error(f"File data tidak ditemukan: {XLSX}. Jalankan notebook sampai bagian ekspor XLSX.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(path, stamp):
    raw = pd.read_excel(path, sheet_name="1_data_mentah", usecols=["video_id", "source"])
    op = pd.read_excel(path, sheet_name="3_aspek_sentimen")
    sc = pd.read_excel(path, sheet_name="5_skor_aspek")
    cat = pd.read_excel(path, sheet_name="7_catalog_mobil")
    return raw, op, sc, cat

try:
    raw, op, sc, cat = load_data(str(XLSX), XLSX.stat().st_mtime)
except Exception as e:
    st.error(f"Gagal membaca {XLSX.name}: {e}")
    st.stop()

sc = sc.set_index("car_model").reindex(columns=ASPECTS)
cat = cat.copy()
cat["price_max"] = cat["price_max"].fillna(np.inf)

def fmt_price(v):
    return "Rp1 miliar+" if v >= 1000 else f"Rp{v} juta"

def normalize(w):
    s = pd.Series(w, dtype=float).reindex(ASPECTS).fillna(0)
    return s / s.sum()

def topsis(X, w):
    w = pd.Series(w, dtype=float).reindex(X.columns).fillna(0); w = w / w.sum()
    V = X / np.sqrt((X ** 2).sum()).replace(0, 1) * w
    db = np.sqrt(((V - V.max()) ** 2).sum(axis=1)); dw = np.sqrt(((V - V.min()) ** 2).sum(axis=1))
    return (dw / (db + dw).replace(0, np.nan)).fillna(.5)

# ---- jumlah opini (positif / negatif / netral) per mobil dan per aspek
def per_car(d):
    return d.groupby("car_model").size().reindex(cat["car_model"]).fillna(0).astype(int)

def per_aspect(d):
    if d.empty:
        return pd.DataFrame(0, index=cat["car_model"], columns=ASPECTS)
    return (d.groupby(["car_model", "aspect"]).size().unstack(fill_value=0)
             .reindex(index=cat["car_model"], columns=ASPECTS, fill_value=0).astype(int))

pos, neg, neu = op[op["sentiment"] == 1], op[op["sentiment"] == -1], op[op["sentiment"] == 0]
cnt = pd.DataFrame({"positif": per_car(pos), "negatif": per_car(neg), "netral": per_car(neu)})
cnt["jumlah_opini"] = cnt["positif"] + cnt["negatif"]
asp_pos, asp_neg = per_aspect(pos), per_aspect(neg)

# ---- sidebar: filter + preferensi
with st.sidebar:
    st.header("Filter kandidat")
    body_opts = sorted(cat["body"].unique())
    body = st.multiselect("Tipe bodi", body_opts, default=[b for b in ["SUV"] if b in body_opts])
    ice_only = st.checkbox("Hanya konvensional (ICE) — kecualikan hybrid/EV", value=True)
    fuel_opts = sorted(cat["fuel"].unique())
    fuel = st.multiselect("Bahan bakar", fuel_opts, default=fuel_opts)
    p_lo, p_hi = st.select_slider("Rentang harga", options=PRICE_OPTS, value=(500, 1000), format_func=fmt_price)
    seats = st.radio("Minimal kursi", [5, 7], horizontal=True)
    st.caption("Pilihan kosong pada tipe bodi/bahan bakar = tidak difilter.")

    st.header("Preferensi aspek")
    preset = st.selectbox("Preset bobot", list(PRESETS))
    base = normalize(PRESETS[preset])
    w = {a: st.slider(LABEL[a], 0, 100, int(round(base[a] * 100)), 1, key=f"w_{preset}_{a}") for a in ASPECTS}
    total_w = sum(w.values())
    if total_w == 0:
        st.error("Semua bobot 0. Naikkan minimal satu bobot.")
        st.stop()
    wn = {a: w[a] / total_w for a in ASPECTS}
    st.caption("Bobot ternormalisasi yang dipakai TOPSIS")
    st.dataframe(pd.DataFrame({"Bobot (%)": [round(wn[a] * 100, 1) for a in ASPECTS]},
                              index=[LABEL[a] for a in ASPECTS]))
    st.caption(f"Total bobot: {sum(wn.values()) * 100:.0f}%")

# ---- filter kandidat
cand = cat.copy()
if body: cand = cand[cand["body"].isin(body)]
if ice_only: cand = cand[cand["powertrain"].eq("ICE")]
if fuel: cand = cand[cand["fuel"].isin(fuel)]
cand = cand[(cand["seats"] >= seats) & (cand["price_max"] >= p_lo)]
if p_hi < 1000: cand = cand[cand["price_min"] <= p_hi]

if len(cand) < 2:
    st.warning("Kandidat kurang dari 2. Longgarkan filter di sidebar.")
    st.stop()

# ---- TOPSIS
X = sc.reindex(cand["car_model"])
imp = X.isna()
X = X.fillna(sc.mean())
s = topsis(X, wn).sort_values(ascending=False)
res = cand.set_index("car_model").loc[s.index]
disp = pd.DataFrame({
    "Peringkat": list(range(1, len(res) + 1)),
    "Skor TOPSIS": s.round(3),
    "Harga": res["price_band"],
    "Body": res["body"],
    "Bahan bakar": res["fuel"],
    "Powertrain": res["powertrain"],
    "Kursi": res["seats"],
    "Jumlah opini": cnt["jumlah_opini"].reindex(res.index),
    "Opini positif": cnt["positif"].reindex(res.index),
    "Opini negatif": cnt["negatif"].reindex(res.index),
    "Aspek data kurang": [", ".join(LABEL[a] for a in ASPECTS if imp.loc[m, a]) for m in res.index],
}, index=res.index)

top = s.index[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rekomendasi teratas", top)
c2.metric("Skor TOPSIS", f"{s.iloc[0]:.3f}")
c3.metric("Harga", str(res.loc[top, "price_band"]))
c4.metric("Kandidat sesuai filter", len(cand))
contrib = (X.loc[top] * pd.Series(wn)).sort_values(ascending=False)
st.success(f"**{top}** unggul terutama pada aspek berbobot tinggi: " +
           ", ".join(f"{LABEL[a]} ({X.loc[top, a]:.2f})" for a in contrib.index[:3]) + ".")

st.subheader("Ranking")
st.dataframe(disp)
st.bar_chart(disp["Skor TOPSIS"])
st.caption("Jumlah opini = opini berpolaritas (positif + negatif) yang dihitung ke skor aspek; opini netral tidak dihitung.")

# ---- profil aspek
st.subheader("Profil aspek")
pick = st.selectbox("Pilih mobil", list(s.index))
prof = pd.DataFrame({"Skor": sc.loc[pick], "Jumlah opini": asp_pos.loc[pick] + asp_neg.loc[pick],
                     "Positif": asp_pos.loc[pick], "Negatif": asp_neg.loc[pick]})
prof.index = [LABEL[a] for a in prof.index]
st.bar_chart(prof["Skor"])
st.dataframe(prof.round(3))
st.caption(f"Skor kosong = opini berpolaritas < {MIN_OPINION}; nilainya diimputasi rata-rata aspek saat TOPSIS.")

# ---- contoh opini
st.subheader("Contoh opini konsumen")
asp = st.selectbox("Aspek", ASPECTS, format_func=lambda a: LABEL[a])
sub = op[(op["car_model"] == pick) & (op["aspect"] == asp)]
if "like_count" not in sub.columns:
    sub = sub.assign(like_count=0)
cp, cn = st.columns(2)
cp.markdown("**Positif**"); cn.markdown("**Negatif**")
def show(col, d):
    for t, l in d.sort_values("like_count", ascending=False)[["clause", "like_count"]].head(5).itertuples(index=False):
        col.write(f"• {t} (👍 {0 if pd.isna(l) else int(l)})")
show(cp, sub[sub["sentiment"] == 1]); show(cn, sub[sub["sentiment"] == -1])

# ---- status data & sumber
with st.expander("Status data dan sumber"):
    st.write({
        "File data": XLSX.name,
        "Terakhir diubah": datetime.fromtimestamp(XLSX.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "Komentar mentah": int(len(raw)),
        "Video unik": int(raw["video_id"].nunique()),
        "Mobil di katalog": int(len(cat)),
        "Klausa-aspek terklasifikasi": int(len(op)),
        "Sumber": sorted(raw["source"].dropna().unique().tolist()),
    })
st.info(SOURCE_NOTE)
