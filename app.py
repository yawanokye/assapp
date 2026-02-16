# app.py
# Streamlit app (v2): Universal Financial Structure Similarity & Convergence (Decomposition-Based)
#
# Features (v2):
# - Supports BOTH wide and long data formats
# - Multiple decomposition options (EEMD / CEEMDAN / EMD, Wavelet(DWT), STL)
# - Structural core selector (Residue/Trend only, Low-frequency core, Variance-top core)
# - Similarity matrix (SSI baseline) + rolling SSI
# - Convergence tests + rolling convergence
# - Clustering (hierarchical dendrogram) + simple network graph (thresholded edges)
# - Exports: decomposed components, feature tables, similarity matrices, convergence results, and a simple PDF report
#
# Install:
#   pip install -r requirements.txt
# Run:
#   streamlit run app.py

import io
import json
import math
import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

import networkx as nx

# Optional deps (graceful fallback)
HAS_PYEMD = True
HAS_PYWAVELETS = True
HAS_STL = True
HAS_REPORTLAB = True

try:
    from PyEMD import EMD, EEMD, CEEMDAN
except Exception:
    HAS_PYEMD = False

try:
    import pywt
except Exception:
    HAS_PYWAVELETS = False

try:
    from statsmodels.tsa.seasonal import STL
except Exception:
    HAS_STL = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
except Exception:
    HAS_REPORTLAB = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# -----------------------------
# Core types
# -----------------------------
@dataclass
class DecompResult:
    components: pd.DataFrame           # columns: c1..cK + residue/trend where applicable
    component_meta: pd.DataFrame       # per-component features
    method: str
    params: dict


# -----------------------------
# Helpers
# -----------------------------
def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 8:
        return np.nan
    if np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _variance_share(comp: np.ndarray, x: np.ndarray) -> float:
    vx = np.nanvar(x)
    if not np.isfinite(vx) or vx == 0:
        return np.nan
    return float(np.nanvar(comp) / vx)


def _mean_period_from_peaks(comp: np.ndarray) -> float:
    # Mean period: T / number_of_peaks (simple, stable)
    c = comp.copy()
    if np.sum(np.isfinite(c)) < 20:
        return np.nan
    c = np.nan_to_num(c, nan=np.nanmedian(c))
    peaks = 0
    for i in range(1, len(c) - 1):
        if c[i] > c[i - 1] and c[i] > c[i + 1]:
            peaks += 1
    if peaks <= 0:
        return np.nan
    return float(len(c) / peaks)


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return x
    return (x - m) / s


def _hash_params(obj: dict) -> str:
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


def _build_component_meta(x: np.ndarray, comps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x_arr = np.asarray(x, dtype=float)
    for col in comps.columns:
        c = comps[col].to_numpy(dtype=float)
        rows.append({
            "component": col,
            "mean_period": _mean_period_from_peaks(c),
            "corr_with_original": _safe_corr(c, x_arr),
            "variance_share": _variance_share(c, x_arr),
        })
    df = pd.DataFrame(rows)

    # Sort: c1..cK then residue/trend at end
    def _order(name: str) -> int:
        if name in ("residue", "trend"):
            return 10_000
        if name.startswith("c"):
            try:
                return int(name.replace("c", "").split("_")[0])
            except Exception:
                return 9_999
        return 9_999

    df["__order"] = df["component"].apply(_order)
    df = df.sort_values("__order").drop(columns=["__order"]).reset_index(drop=True)
    return df


# -----------------------------
# Decomposition methods
# -----------------------------
def decompose_emd_family(x: np.ndarray, method: str, params: dict) -> DecompResult:
    if not HAS_PYEMD:
        raise RuntimeError("PyEMD not installed. Install: pip install EMD-signal")

    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x))

    if method == "EMD":
        emd = EMD()
        imfs = emd.emd(x)
    elif method == "EEMD":
        eemd = EEMD(trials=int(params["trials"]), noise_width=float(params["noise_width"]))
        imfs = eemd.eemd(x)
    elif method == "CEEMDAN":
        ce = CEEMDAN(trials=int(params["trials"]), noise_width=float(params["noise_width"]))
        imfs = ce.ceemdan(x)
    else:
        raise ValueError("Unknown EMD-family method")

    if imfs.ndim == 1:
        imfs = imfs.reshape(1, -1)

    comps = {f"c{k+1}": imfs[k, :] for k in range(imfs.shape[0])}
    comps_df = pd.DataFrame(comps)
    residue = x - comps_df.sum(axis=1).to_numpy()
    comps_df["residue"] = residue

    meta = _build_component_meta(x, comps_df)
    return DecompResult(components=comps_df, component_meta=meta, method=method, params=params)


def decompose_wavelet(x: np.ndarray, params: dict) -> DecompResult:
    if not HAS_PYWAVELETS:
        raise RuntimeError("PyWavelets not installed. Install: pip install pywavelets")

    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x))

    wavelet = params["wavelet"]
    level = int(params["level"])

    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode="symmetric")
    # coeffs: [cA_n, cD_n, cD_{n-1}, ..., cD_1]

    comps = {}
    # Detail components as modes
    for idx in range(1, len(coeffs)):
        c = [np.zeros_like(cc) for cc in coeffs]
        c[idx] = coeffs[idx]
        rec = pywt.waverec(c, wavelet=wavelet, mode="symmetric")[: len(x)]
        comps[f"c{idx}"] = rec

    # Approximation as trend
    c_tr = [np.zeros_like(cc) for cc in coeffs]
    c_tr[0] = coeffs[0]
    trend = pywt.waverec(c_tr, wavelet=wavelet, mode="symmetric")[: len(x)]

    comps_df = pd.DataFrame(comps)
    comps_df["trend"] = trend

    meta = _build_component_meta(x, comps_df)
    return DecompResult(components=comps_df, component_meta=meta, method="Wavelet(DWT)", params=params)


def decompose_stl(x: np.ndarray, params: dict) -> DecompResult:
    if not HAS_STL:
        raise RuntimeError("statsmodels not installed. Install: pip install statsmodels")

    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x))

    period = int(params["period"])
    robust = bool(params["robust"])

    stl = STL(x, period=period, robust=robust)
    res = stl.fit()

    comps_df = pd.DataFrame({
        "c1_seasonal": res.seasonal,
        "c2_cycle": res.resid,
        "trend": res.trend
    })

    meta = _build_component_meta(x, comps_df)
    return DecompResult(components=comps_df, component_meta=meta, method="STL", params=params)


def run_decomposition(x: np.ndarray, method: str, params: dict) -> DecompResult:
    if method in ("EMD", "EEMD", "CEEMDAN"):
        return decompose_emd_family(x, method, params)
    if method == "Wavelet(DWT)":
        return decompose_wavelet(x, params)
    if method == "STL":
        return decompose_stl(x, params)
    raise ValueError("Unsupported method")


# -----------------------------
# Structure selectors
# -----------------------------
def select_structural_core(
    dr: DecompResult,
    structure_mode: str,
    low_k: int,
    top_var_share: float
) -> pd.Series:
    cols = list(dr.components.columns)

    # 1) Residue/Trend only
    if structure_mode == "Residue/Trend only":
        for name in ("residue", "trend"):
            if name in cols:
                return dr.components[name]
        return dr.components.iloc[:, -1]

    # 2) Low-frequency core: last low_k + residue/trend if exists
    if structure_mode.startswith("Low-frequency"):
        base_cols = [c for c in cols if c not in ("residue", "trend")]
        selected = []
        if base_cols:
            selected.extend(base_cols[-low_k:])
        if "residue" in cols:
            selected.append("residue")
        elif "trend" in cols:
            selected.append("trend")
        if not selected:
            selected = [cols[-1]]
        return dr.components[selected].sum(axis=1)

    # 3) Variance-top core: include largest-variance components until reaching threshold
    if structure_mode.startswith("Variance-top"):
        meta = dr.component_meta.copy()
        meta = meta.dropna(subset=["variance_share"])
        if meta.empty:
            return dr.components.iloc[:, -1]
        meta = meta.sort_values("variance_share", ascending=False)

        selected = []
        running = 0.0
        for _, row in meta.iterrows():
            comp = row["component"]
            if comp not in dr.components.columns:
                continue
            selected.append(comp)
            running += float(row["variance_share"])
            if running >= top_var_share:
                break
        if not selected:
            selected = [dr.components.columns[-1]]
        return dr.components[selected].sum(axis=1)

    return dr.components.iloc[:, -1]


# -----------------------------
# Similarity & convergence
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_decomposition(series_values: Tuple[float, ...], method: str, params_json: str) -> DecompResult:
    # Cache-friendly wrapper: pass tuple + json string
    x = np.asarray(series_values, dtype=float)
    params = json.loads(params_json)
    return run_decomposition(x, method, params)


def compute_similarity_matrix(
    series: Dict[str, np.ndarray],
    method: str,
    params: dict,
    structure_mode: str,
    low_k: int,
    top_var_share: float
) -> Tuple[pd.DataFrame, Dict[str, DecompResult], Dict[str, np.ndarray]]:
    names = list(series.keys())
    results: Dict[str, DecompResult] = {}
    struct: Dict[str, np.ndarray] = {}

    params_json = json.dumps(params, sort_keys=True)

    for name in names:
        dr = cached_decomposition(tuple(series[name].tolist()), method, params_json)
        results[name] = dr
        s = select_structural_core(dr, structure_mode, low_k, top_var_share).to_numpy(dtype=float)
        struct[name] = s

    sim = np.full((len(names), len(names)), np.nan, dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                sim[i, j] = 1.0
            else:
                sim[i, j] = _safe_corr(struct[a], struct[b])

    sim_df = pd.DataFrame(sim, index=names, columns=names)
    return sim_df, results, struct


def rolling_similarity(
    struct: Dict[str, np.ndarray],
    dates: np.ndarray,
    window: int
) -> pd.DataFrame:
    # Rolling average similarity (mean of off-diagonal correlations) per time window
    names = list(struct.keys())
    T = len(dates)
    out = []
    out_dates = []

    if window < 20 or window >= T:
        return pd.DataFrame()

    for end in range(window, T + 1):
        start = end - window
        mat = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = struct[names[i]][start:end]
                b = struct[names[j]][start:end]
                mat.append(_safe_corr(a, b))
        out.append(np.nanmean(mat) if len(mat) else np.nan)
        out_dates.append(dates[end - 1])

    return pd.DataFrame({"date": out_dates, "rolling_mean_similarity": out})


def convergence_table(
    struct: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str]]
) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        x = np.asarray(struct[a], dtype=float)
        y = np.asarray(struct[b], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 30:
            rows.append({"pair": f"{a} vs {b}", "slope": np.nan, "p_value": np.nan, "r2": np.nan})
            continue

        d = np.abs(x[mask] - y[mask])
        t = np.arange(len(d), dtype=float)
        lr = linregress(t, d)
        rows.append({
            "pair": f"{a} vs {b}",
            "slope": lr.slope,         # negative means distance decreasing (convergence)
            "p_value": lr.pvalue,
            "r2": lr.rvalue**2
        })
    return pd.DataFrame(rows)


def rolling_distance(
    struct: Dict[str, np.ndarray],
    dates: np.ndarray,
    pair: Tuple[str, str],
    window: int
) -> pd.DataFrame:
    a, b = pair
    x = np.asarray(struct[a], dtype=float)
    y = np.asarray(struct[b], dtype=float)
    T = len(dates)

    if window < 20 or window >= T:
        return pd.DataFrame()

    out_dates, out_vals = [], []
    for end in range(window, T + 1):
        start = end - window
        xa = x[start:end]
        yb = y[start:end]
        mask = np.isfinite(xa) & np.isfinite(yb)
        if mask.sum() < 10:
            out_vals.append(np.nan)
        else:
            out_vals.append(float(np.nanmean(np.abs(xa[mask] - yb[mask]))))
        out_dates.append(dates[end - 1])

    return pd.DataFrame({"date": out_dates, "rolling_mean_distance": out_vals})


# -----------------------------
# Export: PDF report
# -----------------------------
def build_pdf_report(
    title: str,
    settings: dict,
    sim_df: pd.DataFrame,
    conv_df: pd.DataFrame,
) -> Optional[bytes]:
    if not HAS_REPORTLAB:
        return None

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 2 * cm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, y, title)
    y -= 1 * cm

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, "Settings")
    y -= 0.6 * cm

    c.setFont("Helvetica", 9)
    for k, v in settings.items():
        txt = f"{k}: {v}"
        c.drawString(2 * cm, y, txt[:110])
        y -= 0.45 * cm
        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm
            c.setFont("Helvetica", 9)

    # Similarity summary
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2 * cm, y, "Structural Similarity (matrix summary)")
    y -= 0.6 * cm
    c.setFont("Helvetica", 9)

    # Write top correlations
    pairs = []
    cols = sim_df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = sim_df.iloc[i, j]
            if np.isfinite(val):
                pairs.append((cols[i], cols[j], float(val)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for a, b, v in pairs[:10]:
        c.drawString(2 * cm, y, f"{a} vs {b}: {v:.3f}")
        y -= 0.45 * cm
        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm
            c.setFont("Helvetica", 9)

    # Convergence summary
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2 * cm, y, "Structural Convergence (distance slope test, negative slope = convergence)")
    y -= 0.6 * cm
    c.setFont("Helvetica", 9)

    if conv_df is not None and not conv_df.empty:
        for _, row in conv_df.head(12).iterrows():
            c.drawString(
                2 * cm, y,
                f"{row['pair']}: slope={row['slope']:.6f}, p={row['p_value']:.4f}, r2={row['r2']:.3f}"
            )
            y -= 0.45 * cm
            if y < 3 * cm:
                c.showPage()
                y = height - 2 * cm
                c.setFont("Helvetica", 9)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


# -----------------------------
# Data loading (wide + long)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_file(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    raise ValueError("Unsupported file type")


def long_to_wide(df: pd.DataFrame, date_col: str, var_col: str, value_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[value_col] = _to_numeric_series(tmp[value_col])
    tmp = tmp.dropna(subset=[date_col, var_col, value_col])
    wide = tmp.pivot_table(index=date_col, columns=var_col, values=value_col, aggfunc="mean")
    wide = wide.reset_index()
    wide.columns = [str(c) for c in wide.columns]
    return wide


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Financial Structure Similarity & Convergence", layout="wide")
st.title("Financial Structure Similarity & Convergence (Decomposition App)")

with st.expander("Quick guide", expanded=False):
    st.write(
        "1) Upload data (wide or long). 2) Select decomposition method. 3) Choose structural core. "
        "4) View decomposition, similarity, clustering, and convergence. 5) Export results."
    )

# Sidebar: input
st.sidebar.header("A) Data input")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
use_demo = st.sidebar.checkbox("Use demo dataset", value=(uploaded is None))

data_format = st.sidebar.radio("Data format", ["Wide", "Long"], index=0)

if use_demo:
    np.random.seed(7)
    T = 720
    t = np.arange(T)
    dates = pd.date_range("2020-01-01", periods=T, freq="D")
    base = 0.002 * t + 0.25 * np.sin(2 * np.pi * t / 200)

    A = base + 0.50 * np.sin(2 * np.pi * t / 30) + 0.15 * np.random.randn(T)
    B = base + 0.35 * np.sin(2 * np.pi * t / 30) + 0.18 * np.random.randn(T)
    C = 0.001 * t + 0.60 * np.sin(2 * np.pi * t / 60) + 0.25 * np.random.randn(T)
    D = base + 0.10 * np.sin(2 * np.pi * t / 14) + 0.20 * np.random.randn(T)

    df_raw = pd.DataFrame({"date": dates, "Series_A": A, "Series_B": B, "Series_C": C, "Series_D": D})
    data_format = "Wide"
else:
    if uploaded is None:
        st.info("Upload a file or tick the demo option.")
        st.stop()
    try:
        df_raw = load_file(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

st.sidebar.header("B) Column mapping")
cols = df_raw.columns.tolist()
if len(cols) < 2:
    st.error("Dataset needs at least 2 columns.")
    st.stop()

if data_format == "Wide":
    date_col = st.sidebar.selectbox("Date/Time column", options=cols, index=0)
    numeric_candidates = [c for c in cols if c != date_col]
    value_cols = st.sidebar.multiselect("Series columns", options=numeric_candidates, default=numeric_candidates[:4])
    if not value_cols:
        st.error("Select at least one series column.")
        st.stop()
    df_wide = df_raw[[date_col] + value_cols].copy()

else:
    date_col = st.sidebar.selectbox("Date/Time column", options=cols, index=0)
    var_col = st.sidebar.selectbox("Variable column (names)", options=[c for c in cols if c != date_col], index=0)
    value_col = st.sidebar.selectbox("Value column", options=[c for c in cols if c not in (date_col, var_col)], index=0)
    df_wide = long_to_wide(df_raw, date_col, var_col, value_col)
    # after pivot
    cols2 = df_wide.columns.tolist()
    date_col = cols2[0]
    numeric_candidates = cols2[1:]
    value_cols = st.sidebar.multiselect("Series columns", options=numeric_candidates, default=numeric_candidates[:4])

    if not value_cols:
        st.error("Select at least one series column.")
        st.stop()

st.sidebar.header("C) Cleaning & scaling")
dropna_rows = st.sidebar.checkbox("Drop rows with missing values in selected series", value=True)
standardize = st.sidebar.checkbox("Standardize (z-score) before decomposition", value=False)

df_wide[date_col] = pd.to_datetime(df_wide[date_col], errors="coerce")
df_wide = df_wide.sort_values(date_col)
for c in value_cols:
    df_wide[c] = _to_numeric_series(df_wide[c])

if dropna_rows:
    df_wide = df_wide.dropna(subset=[date_col] + value_cols)

if len(df_wide) < 80:
    st.error("Not enough rows after cleaning. Try disabling dropna or choose different series.")
    st.stop()

dates_np = df_wide[date_col].to_numpy()

series_dict: Dict[str, np.ndarray] = {}
for c in value_cols:
    x = df_wide[c].to_numpy(dtype=float)
    if standardize:
        x = _standardize(x)
    series_dict[c] = x

# Sidebar: decomposition
st.sidebar.header("D) Decomposition method")
available_methods = []
if HAS_PYEMD:
    available_methods.extend(["EEMD", "CEEMDAN", "EMD"])
if HAS_PYWAVELETS:
    available_methods.append("Wavelet(DWT)")
if HAS_STL:
    available_methods.append("STL")

if not available_methods:
    st.error("No decomposition backends found. Install at least one listed in requirements.txt.")
    st.stop()

method = st.sidebar.selectbox("Method", options=available_methods, index=0)

params = {}
if method in ("EEMD", "CEEMDAN"):
    params["trials"] = st.sidebar.slider("Ensemble trials", 10, 400, 120, 10)
    params["noise_width"] = st.sidebar.slider("Noise width", 0.01, 0.80, 0.20, 0.01)
elif method == "EMD":
    params = {}
elif method == "Wavelet(DWT)":
    params["wavelet"] = st.sidebar.selectbox("Wavelet family", ["db4", "db6", "sym4", "coif1", "haar"], index=0)
    params["level"] = st.sidebar.slider("Decomposition level", 2, 10, 5, 1)
elif method == "STL":
    params["period"] = st.sidebar.number_input("STL period", min_value=2, max_value=365, value=30, step=1)
    params["robust"] = st.sidebar.checkbox("Robust STL", value=True)

# Sidebar: structure
st.sidebar.header("E) Structural core")
low_k = st.sidebar.slider("Low-frequency modes count", 1, 8, 2, 1)
top_var_share = st.sidebar.slider("Variance-top threshold", 0.10, 0.95, 0.70, 0.05)

structure_mode = st.sidebar.selectbox(
    "Structural core definition",
    options=[
        "Residue/Trend only",
        f"Low-frequency (last {low_k} modes + residue/trend)",
        f"Variance-top (top components until {top_var_share:.2f} variance share)",
    ],
    index=0
)

# Sidebar: rolling + clustering
st.sidebar.header("F) Rolling & graphs")
rolling_window = st.sidebar.slider("Rolling window (observations)", 30, min(360, len(df_wide) - 1), 120, 10)
net_threshold = st.sidebar.slider("Network edge threshold (similarity)", -0.20, 0.95, 0.60, 0.05)

# Tabs
tab_data, tab_decomp, tab_similarity, tab_convergence, tab_export = st.tabs(
    ["Data", "Decomposition", "Similarity & Clustering", "Convergence", "Export"]
)

with tab_data:
    st.subheader("Data preview")
    st.dataframe(df_wide[[date_col] + value_cols].head(25), use_container_width=True)
    st.write(f"Rows: {len(df_wide):,} | Series: {len(value_cols)}")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for c in value_cols[: min(6, len(value_cols))]:
        ax.plot(dates_np, series_dict[c], label=c, linewidth=1.0)
    ax.set_title("Selected series (first up to 6)")
    ax.legend(loc="best", fontsize=8)
    st.pyplot(fig, use_container_width=True)

with tab_decomp:
    st.subheader("Decomposition view")
    target = st.selectbox("Select series", options=value_cols, index=0)

    try:
        dr = cached_decomposition(tuple(series_dict[target].tolist()), method, json.dumps(params, sort_keys=True))
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.write(f"Method: {dr.method}")
    st.dataframe(dr.component_meta, use_container_width=True)

    # Structural core plot
    core = select_structural_core(
        dr,
        structure_mode if not structure_mode.startswith("Variance-top") else "Variance-top",
        low_k,
        top_var_share
    ).to_numpy(dtype=float)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(dates_np, series_dict[target], label="Original", linewidth=1.1)
    ax.plot(dates_np, core, label="Structural core", linewidth=1.1)
    ax.set_title(f"{target}: Original vs Structural core")
    ax.legend(loc="best", fontsize=9)
    st.pyplot(fig, use_container_width=True)

    # Component plots (subset)
    st.write("Component plots (subset)")
    comp_cols = dr.components.columns.tolist()
    show_cols = comp_cols[: min(6, len(comp_cols))]
    if len(comp_cols) > 9:
        show_cols += comp_cols[-2:]
    show_cols = list(dict.fromkeys(show_cols))

    for col in show_cols:
        fig = plt.figure(figsize=(10, 2.4))
        ax = fig.add_subplot(111)
        ax.plot(dates_np, dr.components[col].to_numpy(dtype=float), linewidth=1.0)
        ax.set_title(col)
        st.pyplot(fig, use_container_width=True)

with tab_similarity:
    st.subheader("Structural similarity (SSI baseline) + clustering + network")

    # compute similarity and structural cores for all series
    try:
        sim_df, results, struct = compute_similarity_matrix(
            series_dict,
            method,
            params,
            structure_mode if not structure_mode.startswith("Variance-top") else "Variance-top",
            low_k,
            top_var_share
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.write("Similarity is computed as correlation between chosen structural cores.")
    st.dataframe(sim_df.round(3), use_container_width=True)

    # Heatmap
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(sim_df.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(range(len(sim_df.columns)))
    ax.set_xticklabels(sim_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(sim_df.index)))
    ax.set_yticklabels(sim_df.index)
    ax.set_title("Structural Similarity Matrix (correlation)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, use_container_width=True)

    # Rolling similarity (mean off-diagonal)
    if len(value_cols) >= 2:
        roll_df = rolling_similarity(struct, dates_np, rolling_window)
        if not roll_df.empty:
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            ax.plot(roll_df["date"], roll_df["rolling_mean_similarity"], linewidth=1.1)
            ax.set_title(f"Rolling mean structural similarity (window={rolling_window})")
            st.pyplot(fig, use_container_width=True)

    # Hierarchical clustering (distance = 1 - similarity)
    st.write("Hierarchical clustering (distance = 1 - similarity)")
    sim = sim_df.to_numpy(dtype=float)
    # Make valid distance matrix (handle NaNs)
    sim2 = np.where(np.isfinite(sim), sim, 0.0)
    dist = 1.0 - sim2
    np.fill_diagonal(dist, 0.0)

    try:
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        dendrogram(Z, labels=sim_df.index.tolist(), leaf_rotation=45)
        ax.set_title("Dendrogram (average linkage)")
        st.pyplot(fig, use_container_width=True)
    except Exception:
        st.info("Clustering skipped (distance matrix issue).")

    # Network graph (threshold edges)
    st.write(f"Network graph (edges where similarity ≥ {net_threshold:.2f})")
    G = nx.Graph()
    for n in sim_df.index:
        G.add_node(n)
    for i, a in enumerate(sim_df.index):
        for j, b in enumerate(sim_df.columns):
            if j <= i:
                continue
            v = sim_df.loc[a, b]
            if np.isfinite(v) and v >= net_threshold:
                G.add_edge(a, b, weight=float(v))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.axis("off")
    if G.number_of_edges() == 0:
        ax.text(0.5, 0.5, "No edges at this threshold. Lower the threshold.", ha="center", va="center")
    else:
        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
        nx.draw_networkx_edges(G, pos, ax=ax, width=1.2)
    ax.set_title("Structural Similarity Network")
    st.pyplot(fig, use_container_width=True)

with tab_convergence:
    st.subheader("Structural convergence")

    # Ensure similarity computed and struct available
    try:
        _, _, struct = compute_similarity_matrix(
            series_dict,
            method,
            params,
            structure_mode if not structure_mode.startswith("Variance-top") else "Variance-top",
            low_k,
            top_var_share
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Pairs
    all_pairs = []
    for i in range(len(value_cols)):
        for j in range(i + 1, len(value_cols)):
            all_pairs.append((value_cols[i], value_cols[j]))
    pair_labels = [f"{a} vs {b}" for a, b in all_pairs]

    default_pairs = pair_labels[: min(4, len(pair_labels))]
    selected = st.multiselect("Pairs to test", options=pair_labels, default=default_pairs)

    pairs = []
    for lab in selected:
        a, b = lab.split(" vs ")
        pairs.append((a.strip(), b.strip()))

    conv_df = convergence_table(struct, pairs) if pairs else pd.DataFrame()
    if conv_df.empty:
        st.info("Select at least one pair.")
    else:
        st.write("Distance slope test on structural core: negative slope suggests convergence.")
        st.dataframe(conv_df, use_container_width=True)

        # Rolling distance plot for the first selected pair
        a0, b0 = pairs[0]
        rd = rolling_distance(struct, dates_np, (a0, b0), rolling_window)
        if not rd.empty:
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            ax.plot(rd["date"], rd["rolling_mean_distance"], linewidth=1.1)
            ax.set_title(f"Rolling mean structural distance: {a0} vs {b0} (window={rolling_window})")
            st.pyplot(fig, use_container_width=True)

with tab_export:
    st.subheader("Export results")

    # Recompute for export to ensure consistency
    sim_df, results, struct = compute_similarity_matrix(
        series_dict,
        method,
        params,
        structure_mode if not structure_mode.startswith("Variance-top") else "Variance-top",
        low_k,
        top_var_share
    )

    # Export similarity matrix
    st.download_button(
        "Download similarity matrix (CSV)",
        data=sim_df.to_csv().encode("utf-8"),
        file_name="structural_similarity_matrix.csv"
    )

    # Export structural cores
    cores_df = pd.DataFrame({"date": dates_np})
    for name, s in struct.items():
        cores_df[name] = s
    st.download_button(
        "Download structural cores (CSV)",
        data=cores_df.to_csv(index=False).encode("utf-8"),
        file_name="structural_cores.csv"
    )

    # Export decomposition components + meta for each series (zip-like workaround: separate downloads)
    pick = st.selectbox("Select a series to export its decomposition", options=value_cols, index=0)
    dr_pick = cached_decomposition(tuple(series_dict[pick].tolist()), method, json.dumps(params, sort_keys=True))

    comp_df = dr_pick.components.copy()
    comp_df.insert(0, "date", dates_np)
    st.download_button(
        f"Download decomposition components for {pick} (CSV)",
        data=comp_df.to_csv(index=False).encode("utf-8"),
        file_name=f"decomposition_{pick}.csv"
    )
    st.download_button(
        f"Download component features for {pick} (CSV)",
        data=dr_pick.component_meta.to_csv(index=False).encode("utf-8"),
        file_name=f"features_{pick}.csv"
    )

    # Convergence export
    all_pairs = []
    for i in range(len(value_cols)):
        for j in range(i + 1, len(value_cols)):
            all_pairs.append((value_cols[i], value_cols[j]))
    conv_df = convergence_table(struct, all_pairs) if len(all_pairs) else pd.DataFrame()
    if not conv_df.empty:
        st.download_button(
            "Download convergence results (CSV)",
            data=conv_df.to_csv(index=False).encode("utf-8"),
            file_name="structural_convergence_results.csv"
        )

    # PDF report
    st.write("PDF report (summary)")
    settings = {
        "method": method,
        "params": params,
        "structure_mode": structure_mode,
        "low_k": low_k,
        "top_var_share": top_var_share,
        "standardize": standardize,
        "dropna_rows": dropna_rows,
        "rolling_window": rolling_window,
        "network_threshold": net_threshold
    }

    pdf_bytes = build_pdf_report(
        title="Financial Structure Similarity & Convergence Report",
        settings=settings,
        sim_df=sim_df,
        conv_df=conv_df
    )
    if pdf_bytes is None:
        st.info("PDF export needs reportlab. Install it, or keep CSV exports.")
    else:
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name="structure_similarity_convergence_report.pdf"
        )

st.caption(
    "Backends used when available: PyEMD (EMD/EEMD/CEEMDAN), PyWavelets (Wavelet), statsmodels (STL). "
    "Tip: EEMD/CEEMDAN can be slower on long series, reduce trials or use STL/Wavelet for quick checks."
)
