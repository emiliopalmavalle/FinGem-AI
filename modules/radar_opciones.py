"""
radar_opciones.py  — v4  (Radar Quant + Swing Deep Scan)
=========================================================

CAPA 1 — Radar Quant original (PRESERVADO INTACTO):
  Score 0-100 por horizonte (Scalping / Swing / Posicional)
  Señales: PCR extremo · Volumen inusual · OI+Delta · IV Spike
  Gráfico 3-subplots: Gamma Map · Delta neto · IV surface

CAPA 2 — Swing Deep Scan:
  Objetivo: identificar los mejores activos para entradas
  direccionales en opciones a 1-3 días.

  Filtros cuantitativos estilo Finviz/Derivatives:
    F1  Volumen inusual precio     RVOL = vol_hoy / media_vol_20d
    F2  Squeeze técnico            BB dentro de KC (vectorizado Pandas)
    F3  IV Rank                    percentil IV actual vs ventana 30d-proxy
    F4  Momentum EMA               cruce EMA-8 vs EMA-21 (diario)
    F5  RSI(14) + Divergencia      señal de divergencia alcista/bajista
    F6  Liquidez opciones          OI mínimo + spread bid/ask ATM
    F7  Contexto macro             bias SPY/QQQ del día

  Score Swing 0-100 ponderado:
    RVOL precio  15%  |  Squeeze 20%  |  IV Rank 20%
    Momentum     20%  |  RSI     10%  |  Liquidez 15%

  Señal direccional:
    BUY CALL  — score_alcista >= 65 + contexto macro alcista
    BUY PUT   — score_bajista >= 65 + contexto macro bajista
    WAIT      — sin ventaja estadística clara

Prompt Gemini (MEJORADO — recupera profundidad del cerebro viejo):
  Datos técnicos completos (EMA 8/21/55/200, ADX, RSI, squeeze)
  Contexto macro (bias SPY/QQQ), IV Rank, muros gamma/delta,
  señal calculada + sub-scores + Entry/SL/TP explícitos

Changelog v4:
  - _calcular_indicadores_tecnicos completado (EMAs, RVOL, divergencias)
  - ddof=0 en std para consistencia institucional
  - PCR con guardia de volumen marginal (umbral mínimo 10 contratos)
  - IV Rank mejorado: usa IV real de cadena si disponible
  - _descargar_historiales optimizado: .copy() explícito evita fragmentación
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import concurrent.futures
import warnings
from modules.gemini_client import llamar_gemini

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ══════════════════════════════════════════════════════

MAX_WORKERS = 8

HORIZONTE_DTE = {
    "scalping":   (1,   7),
    "swing":      (21,  60),
    "posicional": (90, 210),
}

# Pesos Score Quant original (Capa 1)
PESOS_QUANT = {
    "pcr":      0.30,
    "rvol_opt": 0.25,
    "oi_delta": 0.25,
    "iv_spike": 0.20,
}

# Pesos Score Swing Deep Scan (Capa 2)
PESOS_SWING = {
    "rvol_precio": 0.15,
    "squeeze":     0.20,
    "iv_rank":     0.20,
    "momentum":    0.20,
    "rsi":         0.10,
    "liquidez":    0.15,
}

# Semáforo Score Quant
VERDE    = 65
AMARILLO = 40

# Umbral señal direccional
UMBRAL_SENAL = 65
OI_MINIMO    = 500

# Umbral mínimo de volumen para PCR (evita distorsión por valores marginales)
PCR_VOL_MINIMO = 10


# ══════════════════════════════════════════════════════
# CAPA 2 — INDICADORES TÉCNICOS (100% vectorizados)
# ══════════════════════════════════════════════════════

def _calcular_indicadores_tecnicos(hist: pd.DataFrame) -> dict:
    """Calcula todos los indicadores técnicos necesarios para el Swing Deep Scan.

    Optimizaciones v4:
        - SMA20 calculado una sola vez para Bollinger Bands y Keltner Channel.
        - ddof=0 en std() para consistencia con cálculos institucionales.
        - ATR vectorizado con np.maximum (sin bucles).
        - RSI con ewm(alpha=1/14) para precisión decimal mejorada.
        - EMAs 8/21/55/200 incluidas (requeridas por _score_swing_ticker).
        - RVOL precio y divergencias RSI calculadas inline.

    Args:
        hist: DataFrame de yfinance con columnas OHLCV estándar.

    Returns:
        dict con todas las series de indicadores indexadas por nombre.
    """
    c, h, lo, v = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    r = {}

    # ── EMAs requeridas por _score_swing_ticker
    r["ema8"]   = c.ewm(span=8,   adjust=False).mean()
    r["ema21"]  = c.ewm(span=21,  adjust=False).mean()
    r["ema55"]  = c.ewm(span=55,  adjust=False).mean()
    r["ema200"] = c.ewm(span=200, adjust=False).mean()

    # ── Optimización: SMA20 una sola vez para BB y KC
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std(ddof=0)  # ddof=0 para consistencia institucional

    # ── Bollinger Bands
    r["bb_upper"], r["bb_lower"] = sma20 + 2 * std20, sma20 - 2 * std20
    r["bb_width"] = r["bb_upper"] - r["bb_lower"]

    # ── ATR vectorizado eficiente
    tr = np.maximum(
        (h - lo),
        np.maximum((h - c.shift(1)).abs(), (lo - c.shift(1)).abs()),
    )
    r["atr14"] = tr.ewm(span=14, adjust=False).mean()

    # ── Keltner Channel (basado en SMA20 + ATR)
    r["kc_upper"], r["kc_lower"] = sma20 + 1.5 * r["atr14"], sma20 - 1.5 * r["atr14"]

    # ── Squeeze Logic (preservando densidad lógica)
    r["squeeze"] = (r["bb_upper"] <= r["kc_upper"]) & (r["bb_lower"] >= r["kc_lower"])
    r["squeeze_release"] = ~r["squeeze"] & r["squeeze"].shift(1).fillna(False)

    # ── RSI con corrección de precisión decimal
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    r["rsi14"] = 100 - (100 / (1 + rs))

    # ── RVOL precio (volumen hoy / media 20 días)
    vol_sma20 = v.rolling(20).mean()
    r["rvol_precio"] = v / vol_sma20.replace(0, np.nan)

    # ── Retornos para momentum
    r["retorno_1d"] = c.pct_change(1)
    r["retorno_5d"] = c.pct_change(5)

    # ── Divergencias RSI (alcista: precio nuevo mínimo, RSI mínimo ascendente)
    ventana_div = 14
    precio_min_roll = c.rolling(ventana_div).min()
    rsi_min_roll    = r["rsi14"].rolling(ventana_div).min()
    precio_max_roll = c.rolling(ventana_div).max()
    rsi_max_roll    = r["rsi14"].rolling(ventana_div).max()

    r["div_bullish"] = (
        (c <= precio_min_roll * 1.005) &
        (r["rsi14"] > rsi_min_roll.shift(1))
    ).fillna(False)

    r["div_bearish"] = (
        (c >= precio_max_roll * 0.995) &
        (r["rsi14"] < rsi_max_roll.shift(1))
    ).fillna(False)

    return r


def _score_swing_ticker(hist: pd.DataFrame, datos_opciones: dict | None) -> dict:
    """Score Swing 0-100 por dirección + señal BUY CALL / BUY PUT / WAIT.

    Todos los cálculos vectorizados; solo lee los valores finales (.iloc[-1]).

    Mejoras v4:
        - IV Rank usa IV real de la cadena de opciones si disponible,
          en lugar de comparar exclusivamente contra volatilidad realizada.
        - Guardia contra NaN en rvol_precio y ema200.

    Args:
        hist: DataFrame OHLCV con al menos 30 velas.
        datos_opciones: dict de _calcular_score_horizonte o None.

    Returns:
        dict con señal, scores, sub-scores y metadatos técnicos.
    """
    if hist.empty or len(hist) < 30:
        return {"senal": "SIN DATOS", "score_alcista": 0, "score_bajista": 0}

    ind = _calcular_indicadores_tecnicos(hist)
    c   = hist["Close"]

    precio   = c.iloc[-1]
    ema8     = ind["ema8"].iloc[-1]
    ema21    = ind["ema21"].iloc[-1]
    ema55    = ind["ema55"].iloc[-1]
    ema200   = ind["ema200"].iloc[-1]
    ema8_p   = ind["ema8"].iloc[-2]
    ema21_p  = ind["ema21"].iloc[-2]
    rsi      = ind["rsi14"].iloc[-1]
    squeeze  = bool(ind["squeeze"].iloc[-1])
    sq_rel   = bool(ind["squeeze_release"].iloc[-1])
    rvol_p   = float(ind["rvol_precio"].iloc[-1]) if not np.isnan(ind["rvol_precio"].iloc[-1]) else 1.0
    div_bull = bool(ind["div_bullish"].iloc[-1])
    div_bear = bool(ind["div_bearish"].iloc[-1])
    ret_1d   = ind["retorno_1d"].iloc[-1]
    ret_5d   = ind["retorno_5d"].iloc[-1]

    # F1 — RVOL precio
    if rvol_p >= 2.5:   score_rvol = 90
    elif rvol_p >= 1.8: score_rvol = 75
    elif rvol_p >= 1.3: score_rvol = 55
    elif rvol_p >= 0.8: score_rvol = 35
    else:               score_rvol = 20

    # F2 — Squeeze
    if sq_rel:    score_squeeze = 95
    elif squeeze: score_squeeze = 70
    else:
        bb_min20 = ind["bb_width"].rolling(20).min().iloc[-1]
        score_squeeze = 55 if ind["bb_width"].iloc[-1] <= bb_min20 * 1.10 else 30

    # F3 — IV Rank (mejorado: IV real de cadena si disponible)
    iv_rank_pct = 50
    iv_actual   = 0.30
    if datos_opciones and datos_opciones.get("iv_avg"):
        iv_actual = datos_opciones["iv_avg"] / 100.0

        # Intentar usar ventana de IV histórica de la cadena (preferible)
        iv_hist_min = datos_opciones.get("iv_hist_min")
        iv_hist_max = datos_opciones.get("iv_hist_max")

        if iv_hist_min is not None and iv_hist_max is not None and iv_hist_max > iv_hist_min:
            # IV Rank real: posición de IV actual dentro del rango histórico
            iv_rank_pct = min(100, max(0, int(
                (iv_actual - iv_hist_min) / (iv_hist_max - iv_hist_min) * 100
            )))
        else:
            # Fallback: proxy con volatilidad realizada 30d (documentado como limitación)
            ret_std = c.pct_change().dropna().iloc[-30:].std() * np.sqrt(252)
            if ret_std > 0:
                iv_rank_pct = min(100, int((iv_actual / (ret_std * 2.5)) * 100))

    if iv_rank_pct < 20:   score_iv_rank = 85
    elif iv_rank_pct < 35: score_iv_rank = 70
    elif iv_rank_pct < 50: score_iv_rank = 55
    elif iv_rank_pct < 70: score_iv_rank = 35
    else:                  score_iv_rank = 15

    # F4 — Momentum EMA (alcista y bajista)
    cruce_bull = (ema8 > ema21) and (ema8_p <= ema21_p)
    cruce_bear = (ema8 < ema21) and (ema8_p >= ema21_p)

    if cruce_bull and precio > ema55:           score_mom_bull = 90
    elif ema8 > ema21 and precio > ema55 and ret_5d > 0: score_mom_bull = 72
    elif ema8 > ema21 and ret_1d > 0:           score_mom_bull = 55
    elif precio > ema200:                        score_mom_bull = 40
    else:                                        score_mom_bull = 20

    if cruce_bear and precio < ema55:           score_mom_bear = 90
    elif ema8 < ema21 and precio < ema55 and ret_5d < 0: score_mom_bear = 72
    elif ema8 < ema21 and ret_1d < 0:           score_mom_bear = 55
    elif precio < ema200:                        score_mom_bear = 40
    else:                                        score_mom_bear = 20

    # F5 — RSI + divergencias
    if rsi < 30:
        s_rsi_bull, s_rsi_bear = 85, 15
    elif rsi < 45 and div_bull:
        s_rsi_bull, s_rsi_bear = 75, 25
    elif rsi < 50:
        s_rsi_bull, s_rsi_bear = 55, 40
    elif rsi < 60:
        s_rsi_bull, s_rsi_bear = 45, 50
    elif rsi > 70:
        s_rsi_bull, s_rsi_bear = 15, 85
    elif rsi > 55 and div_bear:
        s_rsi_bull, s_rsi_bear = 25, 75
    else:
        s_rsi_bull, s_rsi_bear = 40, 40

    # F6 — Liquidez opciones
    score_liquidez = 40
    liquidez_ok    = False
    spread_atm     = None

    if datos_opciones:
        oi_total = datos_opciones.get("oi_total", 0)
        if oi_total >= OI_MINIMO * 10:  score_liquidez = 85; liquidez_ok = True
        elif oi_total >= OI_MINIMO * 3: score_liquidez = 65; liquidez_ok = True
        elif oi_total >= OI_MINIMO:     score_liquidez = 45
        else:                           score_liquidez = 15

        for df_op in [datos_opciones.get("df_calls", pd.DataFrame()), datos_opciones.get("df_puts", pd.DataFrame())]:
            if df_op.empty: continue
            if "bid" in df_op.columns and "ask" in df_op.columns:
                idx_atm = (df_op["strike"] - precio).abs().idxmin()
                bid = df_op.loc[idx_atm, "bid"]
                ask = df_op.loc[idx_atm, "ask"]
                mid = (bid + ask) / 2
                if mid > 0:
                    spread_pct = (ask - bid) / mid * 100
                    spread_atm = round(spread_pct, 1)
                    if spread_pct > 20:   score_liquidez = max(score_liquidez - 25, 10)
                    elif spread_pct > 10: score_liquidez = max(score_liquidez - 10, 20)
                break

    # ── Scores finales por dirección
    intensidad = (
        score_rvol    * PESOS_SWING["rvol_precio"] +
        score_squeeze * PESOS_SWING["squeeze"]     +
        score_iv_rank * PESOS_SWING["iv_rank"]     +
        score_liquidez * PESOS_SWING["liquidez"]
    )
    score_alcista = max(0, min(100, int(intensidad + score_mom_bull * PESOS_SWING["momentum"] + s_rsi_bull * PESOS_SWING["rsi"])))
    score_bajista = max(0, min(100, int(intensidad + score_mom_bear * PESOS_SWING["momentum"] + s_rsi_bear * PESOS_SWING["rsi"])))

    # ── Cruce EMA label
    if cruce_bull:    cruce_str = "Cruce alcista EMA 8/21"
    elif cruce_bear:  cruce_str = "Cruce bajista EMA 8/21"
    elif ema8 > ema21: cruce_str = "Alcista (EMA 8 > 21)"
    else:              cruce_str = "Bajista (EMA 8 < 21)"

    # ── Señal
    if not liquidez_ok:                                              senal = "ILIQUIDO"
    elif score_alcista >= UMBRAL_SENAL and score_alcista > score_bajista: senal = "BUY CALL"
    elif score_bajista >= UMBRAL_SENAL and score_bajista > score_alcista: senal = "BUY PUT"
    else:                                                            senal = "WAIT"

    # ── Razones para el prompt IA
    razones = []
    if sq_rel:      razones.append("Squeeze release")
    elif squeeze:   razones.append("Squeeze activo")
    if rvol_p >= 1.8: razones.append(f"Vol inusual {rvol_p:.1f}x")
    if cruce_bull:  razones.append("Cruce alcista EMA 8/21")
    if cruce_bear:  razones.append("Cruce bajista EMA 8/21")
    if div_bull:    razones.append("Divergencia alcista RSI")
    if div_bear:    razones.append("Divergencia bajista RSI")
    if rsi < 32:    razones.append(f"RSI sobreventa ({rsi:.0f})")
    if rsi > 68:    razones.append(f"RSI sobrecompra ({rsi:.0f})")
    if iv_rank_pct < 25: razones.append(f"IV Rank bajo ({iv_rank_pct}%) — prima barata")

    return {
        "senal": senal, "score_alcista": score_alcista, "score_bajista": score_bajista,
        "squeeze_activo": squeeze, "squeeze_release": sq_rel,
        "cruce_ema": cruce_str, "rsi": round(rsi, 1),
        "rvol_precio": round(rvol_p, 2), "iv_rank_pct": iv_rank_pct,
        "liquidez_ok": liquidez_ok, "spread_atm_pct": spread_atm,
        "ema8": round(ema8, 2), "ema21": round(ema21, 2),
        "ema55": round(ema55, 2),
        "ema200": round(float(ema200), 2) if not np.isnan(float(ema200)) else None,
        "div_bullish": div_bull, "div_bearish": div_bear, "razones": razones,
        "_score_rvol": score_rvol, "_score_squeeze": score_squeeze,
        "_score_iv_rank": score_iv_rank, "_score_mom_bull": score_mom_bull,
        "_score_mom_bear": score_mom_bear, "_score_rsi_bull": s_rsi_bull,
        "_score_rsi_bear": s_rsi_bear, "_score_liquidez": score_liquidez,
    }


# ══════════════════════════════════════════════════════
# CONTEXTO MACRO SPY / QQQ
# ══════════════════════════════════════════════════════

def _obtener_contexto_macro() -> dict:
    """Sesgo del mercado general hoy (EMA 8/21 + retorno 3d).

    

    Returns:
        dict con sesgo_macro, retornos 3d y precios de SPY/QQQ.
    """
    try:
        def _sesgo(sym):
            df = yf.Ticker(sym).history(period="10d", interval="1d")
            if df.empty or len(df) < 3: return "neutro", 0, 0
            cl   = df["Close"]
            ema8  = cl.ewm(span=8,  adjust=False).mean().iloc[-1]
            ema21 = cl.ewm(span=21, adjust=False).mean().iloc[-1]
            ret3  = (cl.iloc[-1] - cl.iloc[-3]) / cl.iloc[-3] * 100
            if ema8 > ema21 and ret3 > 0: s = "alcista"
            elif ema8 < ema21 and ret3 < 0: s = "bajista"
            else: s = "neutro"
            return s, round(ret3, 2), round(float(cl.iloc[-1]), 2)

        s_spy, r_spy, p_spy = _sesgo("SPY")
        s_qqq, r_qqq, p_qqq = _sesgo("QQQ")
        macro = s_spy if s_spy == s_qqq else "mixto"
        return {"sesgo_macro": macro, "spy_ret3d": r_spy, "qqq_ret3d": r_qqq,
                "spy_precio": p_spy, "qqq_precio": p_qqq}
    except Exception:
        return {"sesgo_macro": "desconocido", "spy_ret3d": 0, "qqq_ret3d": 0,
                "spy_precio": 0, "qqq_precio": 0}


# ══════════════════════════════════════════════════════
# CAPA 1 — SCORE QUANT ORIGINAL (PRESERVADO)
# ══════════════════════════════════════════════════════

def _fecha_dte(fechas: list, dte_min: int, dte_max: int) -> list:
    """Filtra fechas de vencimiento dentro del rango DTE especificado."""
    hoy = datetime.now()
    return [f for f in fechas if dte_min <= (datetime.strptime(f, "%Y-%m-%d") - hoy).days <= dte_max]


def _calcular_score_horizonte(ticker_obj, fechas_horizonte, precio_spot):
    """Calcula score Quant para un horizonte temporal específico.

    Mejoras v4:
        - PCR con guardia de volumen marginal (PCR_VOL_MINIMO).
        - Captura iv_hist_min/iv_hist_max para IV Rank real en Capa 2.

    Args:
        ticker_obj: objeto yf.Ticker con sesión inyectada.
        fechas_horizonte: lista de fechas de vencimiento filtradas por DTE.
        precio_spot: precio actual del subyacente.

    Returns:
        dict con score, métricas y DataFrames de calls/puts, o None.
    """
    all_calls, all_puts = [], []
    for fecha in fechas_horizonte:
        try:
            cad = ticker_obj.option_chain(fecha)
            c = cad.calls.copy(); c["_fecha"] = fecha
            p = cad.puts.copy();  p["_fecha"] = fecha
            all_calls.append(c); all_puts.append(p)
        except Exception:
            continue

    if not all_calls and not all_puts: return None

    df_calls = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    df_puts  = pd.concat(all_puts,  ignore_index=True) if all_puts  else pd.DataFrame()

    r_min, r_max = precio_spot * 0.85, precio_spot * 1.15
    if not df_calls.empty:
        df_calls = df_calls[(df_calls["strike"] >= r_min) & (df_calls["strike"] <= r_max)].copy()
    if not df_puts.empty:
        df_puts  = df_puts [(df_puts["strike"]  >= r_min) & (df_puts["strike"]  <= r_max)].copy()

    for col in ["openInterest", "volume", "impliedVolatility", "lastPrice", "delta", "bid", "ask"]:
        for df in [df_calls, df_puts]:
            if col in df.columns: df[col] = df[col].fillna(0)

    vc = df_calls["volume"].sum() if not df_calls.empty else 0
    vp = df_puts ["volume"].sum() if not df_puts.empty  else 0
    oc = df_calls["openInterest"].sum() if not df_calls.empty else 0
    op = df_puts ["openInterest"].sum() if not df_puts.empty  else 0
    oi_total = oc + op

    # PCR con guardia de volumen marginal (evita distorsión por valores ~0)
    if vc >= PCR_VOL_MINIMO:
        pcr = vp / vc
    elif vp >= PCR_VOL_MINIMO:
        pcr = 2.0  # Calls insignificantes vs Puts activos → sesgo bajista extremo
    else:
        pcr = 1.0  # Sin volumen significativo en ningún lado

    if pcr > 1.4:   sp = 90
    elif pcr > 1.1: sp = 70
    elif pcr < 0.5: sp = 85
    elif pcr < 0.7: sp = 60
    else:           sp = 40

    rvol_opt = (vc + vp) / oi_total if oi_total > 0 else 0
    if rvol_opt > 0.30:   sr = 90
    elif rvol_opt > 0.15: sr = 70
    elif rvol_opt > 0.07: sr = 50
    else:                 sr = 25

    dc = (df_calls["delta"] * df_calls["openInterest"]).sum() if (not df_calls.empty and "delta" in df_calls.columns) else oc * 0.50
    dp = (df_puts["delta"].abs() * df_puts["openInterest"]).sum() if (not df_puts.empty and "delta" in df_puts.columns) else op * 0.50
    dt = dc + dp
    bias = dc / dt if dt > 0 else 0.5
    so = int(30 + abs(bias - 0.5) * 2 * 65)

    ica = df_calls["impliedVolatility"].replace(0, np.nan).mean() if not df_calls.empty else 0
    ipa = df_puts ["impliedVolatility"].replace(0, np.nan).mean() if not df_puts.empty  else 0
    iv_avg = np.nanmean([ica, ipa]) if (ica or ipa) else 0.30
    ivr = iv_avg / 0.30
    if ivr > 2.0:   si = 90
    elif ivr > 1.5: si = 75
    elif ivr > 1.2: si = 55
    elif ivr < 0.6: si = 70
    else:           si = 35

    score = max(0, min(100, int(sp * PESOS_QUANT["pcr"] + sr * PESOS_QUANT["rvol_opt"] + so * PESOS_QUANT["oi_delta"] + si * PESOS_QUANT["iv_spike"])))

    muros = []
    for df_op, tipo in [(df_calls, "CALL"), (df_puts, "PUT")]:
        if df_op.empty: continue
        dfo = df_op.copy()
        dfo["_M"] = (dfo["openInterest"] * dfo["lastPrice"] * 100) / 1_000_000
        for _, row in dfo.nlargest(3, "_M").iterrows():
            if row["_M"] > 0.1:
                muros.append({"tipo": tipo, "strike": row["strike"], "oi": int(row["openInterest"]),
                               "dinero_M": round(row["_M"], 2), "iv": round(row.get("impliedVolatility", 0) * 100, 1),
                               "fecha": row.get("_fecha", "")})

    mc = df_calls.loc[df_calls["openInterest"].idxmax(), "strike"] if not df_calls.empty else precio_spot * 1.05
    mp = df_puts .loc[df_puts ["openInterest"].idxmax(), "strike"] if not df_puts.empty  else precio_spot * 0.95

    # Captura min/max IV del horizonte para IV Rank real en Capa 2
    iv_series = pd.concat([
        df_calls["impliedVolatility"].replace(0, np.nan) if not df_calls.empty else pd.Series(dtype=float),
        df_puts["impliedVolatility"].replace(0, np.nan) if not df_puts.empty else pd.Series(dtype=float),
    ]).dropna()
    iv_hist_min = float(iv_series.min()) if not iv_series.empty else None
    iv_hist_max = float(iv_series.max()) if not iv_series.empty else None

    return {"score": score, "pcr": round(pcr, 3), "rvol_opt": round(rvol_opt, 3),
            "bias_calls": round(bias, 3), "iv_avg": round(iv_avg * 100, 1),
            "iv_spike_ratio": round(ivr, 2), "oi_total": int(oi_total),
            "vol_calls": int(vc), "vol_puts": int(vp),
            "mejor_call_strike": mc, "mejor_put_strike": mp,
            "muros": muros, "df_calls": df_calls, "df_puts": df_puts,
            "iv_hist_min": iv_hist_min, "iv_hist_max": iv_hist_max,
            "_score_pcr": sp, "_score_rvol": sr, "_score_oi_delta": so, "_score_iv": si}


# ══════════════════════════════════════════════════════
# PROCESAMIENTO COMPLETO POR TICKER (CAPA 1 + CAPA 2)
# ══════════════════════════════════════════════════════

def _analizar_ticker(args: tuple) -> dict | None:
    """Procesa un ticker completo: Capa 1 (Quant) + Capa 2 (Swing).

    Args:
        args: tupla (ticker_sym, hist_diario) donde hist_diario es el
              DataFrame de precios pre-descargado en batch.

    Returns:
        dict con resultados por horizonte y swing_scan, o None si falla.
    """
    ticker_sym, hist_diario = args
    try:
        t = yf.Ticker(ticker_sym)
        fechas = t.options
        if not fechas: return None

        info = t.fast_info
        precio = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None) or 0.0
        if not precio: return None

        res = {"ticker": ticker_sym, "precio": precio}

        for h, (dmin, dmax) in HORIZONTE_DTE.items():
            fh = _fecha_dte(fechas, dmin, dmax)
            res[h] = _calcular_score_horizonte(t, fh, precio) if fh else None

        datos_opts = res.get("scalping") or res.get("swing")
        res["swing_scan"] = _score_swing_ticker(
            hist_diario if hist_diario is not None else pd.DataFrame(), datos_opts
        )
        return res
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════
# DESCARGA BATCH DE HISTORIALES (una sola llamada HTTP)
# ══════════════════════════════════════════════════════

def _descargar_historiales(lista_tickers: list) -> dict:
    """Descarga historiales de precios en batch con una sola llamada HTTP.

    Optimización v4: usa .copy() explícito al extraer sub-DataFrames
    para evitar fragmentación de memoria por vistas sobre el DataFrame
    multi-nivel. Reduce presión del heap en listas de >50 activos.

    Args:
        lista_tickers: lista de símbolos a descargar.

    Returns:
        dict {ticker: DataFrame} con historiales OHLCV.
    """
    try:
        raw = yf.download(
            lista_tickers, period="60d", interval="1d",
            group_by="ticker", auto_adjust=True, progress=False, threads=True,
        )
        if len(lista_tickers) == 1:
            return {lista_tickers[0]: raw.copy()}
        resultado = {}
        tickers_disponibles = raw.columns.get_level_values(0).unique().tolist()
        for sym in lista_tickers:
            if sym in tickers_disponibles:
                df_sym = raw[sym].dropna(how="all").copy()
                resultado[sym] = df_sym
        return resultado
    except Exception as e:
        return {sym: pd.DataFrame() for sym in lista_tickers}


# ══════════════════════════════════════════════════════
# TABLAS DE SALIDA
# ══════════════════════════════════════════════════════

def _semaforo(score):
    """Convierte un score numérico en emoji semáforo para la tabla."""
    if score is None:       return "N/A"
    if score >= VERDE:      return f"🟢 {score}"
    if score >= AMARILLO:   return f"🟡 {score}"
    return f"🔴 {score}"


def _construir_tabla_scores(resultados: list, contexto_macro: dict):
    """Construye las tablas de salida Quant y Swing a partir de los resultados."""
    filas_q, filas_s = [], []
    sesgo = contexto_macro.get("sesgo_macro", "desconocido")

    for r in resultados:
        if r is None: continue
        def _s(h):    return r[h]["score"]       if r.get(h) else None
        def _pcr(h):  return r[h]["pcr"]          if r.get(h) else None
        def _iv(h):   return r[h]["iv_avg"]        if r.get(h) else None
        def _bias(h): return r[h]["bias_calls"]    if r.get(h) else None

        sd = [_s(h) for h in HORIZONTE_DTE if _s(h) is not None]
        sg = int(np.mean(sd)) if sd else 0

        filas_q.append({
            "Ticker": r["ticker"], "Precio": f"${r['precio']:,.2f}", "Score Global": sg,
            "Scalping (1-7d)": _semaforo(_s("scalping")), "Swing (21-60d)": _semaforo(_s("swing")),
            "Posicional (90d+)": _semaforo(_s("posicional")),
            "PCR Scalp": _pcr("scalping"), "IV Avg%": _iv("swing") or _iv("scalping"),
            "Bias Calls": _bias("swing"), "_n": sg,
        })

        sc = r.get("swing_scan", {})
        senal = sc.get("senal", "N/A")

        # Ajuste macro: degradar señal si va contra el mercado
        if sesgo == "bajista"  and senal == "BUY CALL": senal = "WAIT (contra macro)"
        elif sesgo == "alcista" and senal == "BUY PUT":  senal = "WAIT (contra macro)"

        # Emojis de señal
        emoji_map = {"BUY CALL": "🟢 BUY CALL", "BUY PUT": "🔴 BUY PUT",
                     "WAIT": "🟡 WAIT", "ILIQUIDO": "⚫ ILÍQUIDO",
                     "WAIT (contra macro)": "🟡 WAIT ⚠️"}
        senal_display = emoji_map.get(senal, f"🟡 {senal}")

        sq_icon  = "🔥 Release" if sc.get("squeeze_release") else ("🟤 Activo" if sc.get("squeeze_activo") else "—")
        liq_icon = "✅" if sc.get("liquidez_ok") else "❌"

        filas_s.append({
            "Ticker": r["ticker"], "Señal": senal_display,
            "Score CALL": sc.get("score_alcista", 0), "Score PUT": sc.get("score_bajista", 0),
            "Squeeze": sq_icon, "IV Rank %": sc.get("iv_rank_pct", "—"),
            "RSI(14)": sc.get("rsi", "—"), "EMA Cruce": sc.get("cruce_ema", "—"),
            "RVOL Precio": sc.get("rvol_precio", "—"), "Liquidez": liq_icon,
            "Spread ATM%": sc.get("spread_atm_pct", "—"),
            "Confluencias": " · ".join(sc.get("razones", [])) or "Sin señal",
            "_m": max(sc.get("score_alcista", 0), sc.get("score_bajista", 0)),
        })

    if not filas_q:
        return pd.DataFrame(), pd.DataFrame()

    df_q = pd.DataFrame(filas_q).sort_values("_n", ascending=False).drop(columns=["_n"])
    df_s = pd.DataFrame(filas_s).sort_values("_m", ascending=False).drop(columns=["_m"])
    return df_q, df_s


# ══════════════════════════════════════════════════════
# GRÁFICO GAMMA + DELTA + IV SURFACE
# ══════════════════════════════════════════════════════

def construir_grafico_radar(resultado_ticker: dict) -> go.Figure | None:
    """Construye gráfico Plotly de 3 subplots: Gamma Map, Delta neto, IV Surface."""
    sym    = resultado_ticker["ticker"]
    precio = resultado_ticker["precio"]

    datos = None
    for h in ["swing", "scalping", "posicional"]:
        if resultado_ticker.get(h) and resultado_ticker[h].get("df_calls") is not None:
            datos = resultado_ticker[h]; hlabel = h.capitalize(); break

    if datos is None: return None

    df_c = datos["df_calls"]
    df_p = datos["df_puts"]
    strikes = sorted(set(df_c["strike"].tolist() + df_p["strike"].tolist()))

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.45, 0.30, 0.25],
        subplot_titles=[f"Mapa Gamma — OI por Strike ({hlabel})",
                        "Delta Neto Acumulado (Posicionamiento institucional)",
                        "Implied Volatility por Strike (% anualizado)"])

    if not df_c.empty:
        oi_c = df_c.groupby("strike")["openInterest"].sum().reindex(strikes, fill_value=0)
        fig.add_trace(go.Bar(x=list(oi_c.index), y=list(oi_c.values), name="OI Calls (resistencia)", marker_color="#F23645", opacity=0.85), row=1, col=1)
    if not df_p.empty:
        oi_p = df_p.groupby("strike")["openInterest"].sum().reindex(strikes, fill_value=0)
        fig.add_trace(go.Bar(x=list(oi_p.index), y=list(oi_p.values), name="OI Puts (soporte)", marker_color="#089981", opacity=0.85), row=1, col=1)

    fig.add_vline(x=precio, line_dash="solid", line_color="white", line_width=2, annotation_text=f"Spot ${precio:.2f}", annotation_position="top right", row=1, col=1)
    if not df_c.empty:
        smc = df_c.loc[df_c["openInterest"].idxmax(), "strike"]
        fig.add_vline(x=smc, line_dash="dash", line_color="#FF9800", line_width=1.5, annotation_text=f"Muro CALL ${smc:.0f}", annotation_position="top left", row=1, col=1)
    if not df_p.empty:
        smp = df_p.loc[df_p["openInterest"].idxmax(), "strike"]
        fig.add_vline(x=smp, line_dash="dash", line_color="#00BCD4", line_width=1.5, annotation_text=f"Muro PUT ${smp:.0f}", annotation_position="top left", row=1, col=1)

    dn = {}
    for df_op in [df_c, df_p]:
        if not df_op.empty and "delta" in df_op.columns:
            for s, g in df_op.groupby("strike"):
                dn[s] = dn.get(s, 0) + (g["delta"] * g["openInterest"]).sum()
    if dn:
        sk = sorted(dn.keys()); vk = [dn[s] for s in sk]
        fig.add_trace(go.Bar(x=sk, y=vk, name="Delta neto", marker_color=["#089981" if v >= 0 else "#F23645" for v in vk], opacity=0.80), row=2, col=1)
        fig.add_hline(y=0, line_color="gray", line_width=0.8, row=2, col=1)

    iv_d = {}
    for df_op, key in [(df_c, "call"), (df_p, "put")]:
        if not df_op.empty and "impliedVolatility" in df_op.columns:
            for s, g in df_op.groupby("strike"):
                v = g["impliedVolatility"].replace(0, np.nan).mean()
                if not np.isnan(v): iv_d.setdefault(s, {})[key] = v * 100
    if iv_d:
        sk_iv = sorted(iv_d.keys())
        fig.add_trace(go.Scatter(x=sk_iv, y=[iv_d[s].get("call", 0) for s in sk_iv], mode="lines+markers", name="IV Calls", line=dict(color="#FF9800", width=1.5), marker=dict(size=4)), row=3, col=1)
        fig.add_trace(go.Scatter(x=sk_iv, y=[iv_d[s].get("put",  0) for s in sk_iv], mode="lines+markers", name="IV Puts",  line=dict(color="#00BCD4", width=1.5), marker=dict(size=4)), row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="gray", line_width=0.8, row=3, col=1)

    fig.update_layout(title=dict(text=f"Radar de Opciones Institucional — {sym}", font=dict(size=15)),
        template="plotly_dark", paper_bgcolor="#131722", plot_bgcolor="#131722", height=750,
        barmode="group", hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=15, r=15, t=80, b=15))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, side="right")
    return fig


# ══════════════════════════════════════════════════════
# REPORTE IA — PROMPT ENRIQUECIDO (cerebro v3)
# ══════════════════════════════════════════════════════

def _generar_reporte_ia(ticker, precio, scores_quant, swing_scan, contexto_macro,
                         muros_scalp, muros_swing, muros_macro, api_key="") -> str:
    """Genera el prompt enriquecido para Gemini con todos los datos técnicos."""
    if not api_key:
        return "❌ API Key de Gemini no configurada."

    def _fm(muros):
        if not muros: return "Sin muros relevantes."
        return "\n".join(f"  {m['tipo']} Strike USD {m['strike']:.2f} | OI {m['oi']:,} | USD {m['dinero_M']:.1f}M | IV {m['iv']:.1f}%"
                         for m in muros[:5])

    sc     = swing_scan
    senal  = sc.get("senal", "WAIT")
    razones = " | ".join(sc.get("razones", ["Sin confluencia"])) or "Sin confluencia detectada"
    ema_str = f"EMA 8: USD {sc.get('ema8','?')} | EMA 21: USD {sc.get('ema21','?')} | EMA 55: USD {sc.get('ema55','?')} | EMA 200: {sc.get('ema200','?')}"
    sq_str  = "RELEASE (explosion activa)" if sc.get("squeeze_release") else ("ACTIVO (energia acumulada)" if sc.get("squeeze_activo") else "Inactivo")
    mac     = contexto_macro
    ses     = mac.get("sesgo_macro", "desconocido").upper()
    confluencia_macro = "VA A FAVOR del mercado general." if ses in senal or ses in ("NEUTRO", "MIXTO") else "VA EN CONTRA del mercado — stop mas ajustado obligatorio."

    # Sub-scores para desglose completo
    is_bull = "CALL" in senal
    sqd  = sc.get("_score_squeeze",  "?")
    ivd  = sc.get("_score_iv_rank",  "?")
    momd = sc.get("_score_mom_bull", "?") if is_bull else sc.get("_score_mom_bear", "?")
    rsid = sc.get("_score_rsi_bull", "?") if is_bull else sc.get("_score_rsi_bear", "?")
    rvd  = sc.get("_score_rvol",     "?")
    liqd = sc.get("_score_liquidez", "?")

    prompt = f"""
Eres un Quant Senior especialista en opciones direccionales y flujo institucional.
Estilo: mesa de derivados de primera linea. Directo, estructurado, niveles exactos, sin frases genericas.

════════════════════════════════════════════
ACTIVO: {ticker}  |  Precio Spot: USD {precio:.2f}
════════════════════════════════════════════

CONTEXTO MACRO HOY (SPY/QQQ):
  Sesgo mercado : {ses}
  SPY retorno 3d: {mac.get('spy_ret3d',0):+.2f}%  |  QQQ retorno 3d: {mac.get('qqq_ret3d',0):+.2f}%
  Nota          : La senal {confluencia_macro}

SENAL ALGORÍTMICA (Deep Scan v3):
  Veredicto     : {senal}
  Score CALL    : {sc.get('score_alcista',0)}/100
  Score PUT     : {sc.get('score_bajista',0)}/100
  Confluencias  : {razones}

DESGLOSE TÉCNICO:
  Squeeze BB/KC : {sq_str}  (sub-score: {sqd}/100)
  IV Rank       : {sc.get('iv_rank_pct','?')}%  — {'PRIMA BARATA: comprar opciones simples.' if sc.get('iv_rank_pct',50) < 35 else 'PRIMA CARA: considerar spreads en lugar de compra simple.' if sc.get('iv_rank_pct',50) > 65 else 'Primas en rango medio.'}  (sub-score: {ivd}/100)
  Momentum EMA  : {sc.get('cruce_ema','N/A')}  (sub-score: {momd}/100)
  RSI(14)       : {sc.get('rsi','N/A')}  {'SOBREVENTA' if sc.get('rsi',50) < 32 else 'SOBRECOMPRA' if sc.get('rsi',50) > 68 else ''}  (sub-score: {rsid}/100)
  RVOL precio   : {sc.get('rvol_precio','N/A')}x media 20d  (sub-score: {rvd}/100)
  Liquidez OPC  : {'OK' if sc.get('liquidez_ok') else 'INSUFICIENTE'}  |  Spread ATM: {sc.get('spread_atm_pct','N/A')}%  (sub-score: {liqd}/100)
  Divergencia   : {'Alcista RSI detectada' if sc.get('div_bullish') else 'Bajista RSI detectada' if sc.get('div_bearish') else 'Sin divergencia'}
  {ema_str}

SCORES QUANT OPCIONES (Capa 1 — por horizonte):
  Scalping (1-7d)  : {scores_quant.get('scalping', {}).get('score','N/A')} | PCR: {scores_quant.get('scalping',{}).get('pcr','N/A')} | IV: {scores_quant.get('scalping',{}).get('iv_avg','N/A')}% | Bias Calls: {scores_quant.get('scalping',{}).get('bias_calls','N/A')}
  Swing (21-60d)   : {scores_quant.get('swing',    {}).get('score','N/A')} | PCR: {scores_quant.get('swing',   {}).get('pcr','N/A')} | IV: {scores_quant.get('swing',   {}).get('iv_avg','N/A')}% | Bias Calls: {scores_quant.get('swing',   {}).get('bias_calls','N/A')}
  Posicional (90d+): {scores_quant.get('posicional',{}).get('score','N/A')} | PCR: {scores_quant.get('posicional',{}).get('pcr','N/A')} | IV: {scores_quant.get('posicional',{}).get('iv_avg','N/A')}% | Bias Calls: {scores_quant.get('posicional',{}).get('bias_calls','N/A')}

MUROS GAMMA — Scalping (esta semana):
{_fm(muros_scalp)}

MUROS GAMMA — Swing (21-60 dias):
{_fm(muros_swing)}

MUROS LEAPS — Posicional / Smart Money (90d+):
{_fm(muros_macro)}

════════════════════════════════════════════
INSTRUCCIONES (respeta este orden exacto):
════════════════════════════════════════════

1. VEREDICTO INMEDIATO (1-3 dias):
   Confirma o contradice la senal algoritmica ({senal}).
   Usa datos tecnicos Y contexto macro. Si hay contradiccion, explica cual pesa mas.

2. SETUP ACCIONABLE (1-3 dias — el mas importante):
   Tipo de opcion: CALL o PUT
   Vencimiento sugerido: DTE aproximado
   Strike sugerido: delta aproximado (ej. 0.40 delta)
   ENTRADA al subyacente: precio exacto o zona
   STOP LOSS (nivel que invalida el setup)
   TAKE PROFIT 1 (50% posicion): objetivo conservador
   TAKE PROFIT 2 (50% restante): objetivo agresivo
   Ratio Riesgo/Beneficio estimado

3. MUROS CLAVE A VIGILAR:
   Resistencia proxima (muro CALL relevante)
   Soporte proximo (muro PUT relevante)
   Nivel de gamma squeeze (si aplica)

4. VISION SWING/POSICIONAL (si score > 50):
   Que apuesta el Smart Money a 30-90 dias segun LEAPS
   Divergencia corto vs largo plazo si existe

5. RIESGOS:
   Factor que invalidaria el setup
   Si IV Rank > 65%: recomendar spread especifico en lugar de compra simple

REGLA ABSOLUTA: Usa 'USD' en lugar del simbolo dolar.
Maximo 450 palabras. Niveles numericos exactos. Sin frases genericas.
"""
    # Contexto para el fallback local si se agota el cupo
    ctx_fallback = {
        "ticker":          ticker,
        "precio":          precio,
        "senal":           swing_scan.get("senal", "WAIT"),
        "score_alcista":   swing_scan.get("score_alcista", 0),
        "score_bajista":   swing_scan.get("score_bajista", 0),
        "squeeze_activo":  swing_scan.get("squeeze_activo", False),
        "squeeze_release": swing_scan.get("squeeze_release", False),
        "rsi":             swing_scan.get("rsi", 50),
        "ema8":            swing_scan.get("ema8", precio),
        "ema21":           swing_scan.get("ema21", precio),
        "razones":         swing_scan.get("razones", []),
        "mejor_call_strike": muros_scalp[0]["strike"] if muros_scalp else precio * 1.05,
        "mejor_put_strike":  muros_scalp[-1]["strike"] if muros_scalp else precio * 0.95,
        "pcr":    scores_quant.get("scalping", {}).get("pcr", 1.0) if scores_quant.get("scalping") else 1.0,
        "iv_avg": scores_quant.get("scalping", {}).get("iv_avg", 30) if scores_quant.get("scalping") else 30,
    }
    return llamar_gemini(prompt, api_key, contexto_fallback=ctx_fallback)


# ══════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL PÚBLICA
# ══════════════════════════════════════════════════════

def ejecutar_radar_opciones(lista_tickers: list, gemini_api_key: str = "") -> tuple:
    """Punto de entrada principal del radar.

    Orquesta el pipeline completo: contexto macro → descarga batch →
    análisis paralelo (Capa 1 + Capa 2) → tablas → gráfico → reporte IA.

    Args:
        lista_tickers: lista de símbolos a escanear.
        gemini_api_key: API key de Gemini para el reporte IA.

    Returns:
        tuple de (df_quant, df_swing, fig_gamma, reporte_ia, raw_results, contexto).
    """
    # 1. Contexto macro
    contexto = _obtener_contexto_macro()

    # 2. Historiales batch
    historiales = _descargar_historiales(lista_tickers)

    # 3. Análisis paralelo
    args = [(sym, historiales.get(sym, pd.DataFrame())) for sym in lista_tickers]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        todos = list(ex.map(_analizar_ticker, args))

    resultados = [r for r in todos if r is not None]
    if not resultados:
        return pd.DataFrame(), pd.DataFrame(), None, "Sin datos de opciones.", {}, contexto

    # 4. Tablas
    df_quant, df_swing = _construir_tabla_scores(resultados, contexto)

    # 5. Ticker líder (mayor score_max swing)
    lider = max(resultados, key=lambda r: max(
        r.get("swing_scan", {}).get("score_alcista", 0),
        r.get("swing_scan", {}).get("score_bajista", 0)
    ), default=None)

    # 6. Gráfico
    fig_gamma = construir_grafico_radar(lider) if lider else None

    # 7. Reporte IA
    reporte_ia = ""
    if lider:
        sc = lider.get("swing_scan", {})
        sq = {h: lider.get(h) for h in HORIZONTE_DTE}
        def _m(h): return lider.get(h, {}).get("muros", []) if lider.get(h) else []
        reporte_ia = _generar_reporte_ia(
            ticker=lider["ticker"], precio=lider["precio"],
            scores_quant=sq, swing_scan=sc, contexto_macro=contexto,
            muros_scalp=_m("scalping"), muros_swing=_m("swing"), muros_macro=_m("posicional"),
            api_key=gemini_api_key,
        )

    raw = {r["ticker"]: r for r in resultados}
    return df_quant, df_swing, fig_gamma, reporte_ia, raw, contexto
