"""
radar_etfs.py — Radar dedicado de ETFs (NY + BMV)
==================================================

Escanea listas curadas de ETFs y muestra su CONDICIÓN técnica actual:
tendencia (vs SMA200), momentum semanal, fuerza de volumen (RVOL) y
distancia al máximo de 52 semanas.

A diferencia del escáner dual de acciones, NO filtra: muestra TODOS los
ETFs de la lista con su estado, ordenados de más fuerte a más débil, para
que de un vistazo veas qué sectores/regiones están calientes o fríos.

Usa solo history() + fast_info (sin la llamada pesada .info): rápido y
amable con los rate limits de Yahoo. Los nombres salen de un catálogo
curado (más limpios que los de Yahoo, que ahora prefija "State Street").

Todos los tickers están verificados con datos reales en Yahoo Finance.
Los .MX son listados de la Bolsa Mexicana (NAFTRAC/MEXTRAC locales del IPC
+ los SIC más líquidos, cotizados en pesos).
"""

import concurrent.futures
import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

MAX_WORKERS  = 10
PERIODO_HIST = "2mo"
MIN_VELAS    = 15

# Catálogo NY (cotizan en USD en NYSE/NASDAQ)
ETFS_NY = {
    "SPY":  "SPDR S&P 500",
    "QQQ":  "Invesco Nasdaq 100",
    "DIA":  "SPDR Dow Jones",
    "IWM":  "iShares Russell 2000",
    "VOO":  "Vanguard S&P 500",
    "VTI":  "Vanguard Total Market",
    "GLD":  "SPDR Oro",
    "SLV":  "iShares Plata",
    "USO":  "US Oil Fund (Petróleo)",
    "UNG":  "US Natural Gas",
    "XLE":  "Sector Energía",
    "XLF":  "Sector Financiero",
    "XLK":  "Sector Tecnología",
    "XLV":  "Sector Salud",
    "XLI":  "Sector Industrial",
    "XLY":  "Sector Consumo Discrecional",
    "XLP":  "Sector Consumo Básico",
    "XLU":  "Sector Utilities",
    "XLB":  "Sector Materiales",
    "XLRE": "Sector Real Estate",
    "XLC":  "Sector Comunicaciones",
    "SMH":  "VanEck Semiconductores",
    "SOXX": "iShares Semiconductores",
    "ARKK": "ARK Innovation",
    "KRE":  "SPDR Bancos Regionales",
    "XBI":  "SPDR Biotecnología",
    "XOP":  "SPDR Exploración Petrolera",
    "EEM":  "iShares Emergentes",
    "TLT":  "iShares Bonos 20+ años",
    "HYG":  "iShares Bonos High Yield",
    "LQD":  "iShares Bonos Grado Inversión",
}

# Catálogo BMV (cotizan en MXN en la Bolsa Mexicana; sufijo .MX)
ETFS_BMV = {
    "NAFTRAC.MX": "NAFTRAC — IPC México (local)",
    "MEXTRAC.MX": "MEXTRAC — IPC México (local)",
    "IVV.MX":  "iShares S&P 500 (SIC)",
    "SPY.MX":  "SPDR S&P 500 (SIC)",
    "QQQ.MX":  "Nasdaq 100 (SIC)",
    "VOO.MX":  "Vanguard S&P 500 (SIC)",
    "VTI.MX":  "Vanguard Total Market (SIC)",
    "DIA.MX":  "Dow Jones (SIC)",
    "IWM.MX":  "Russell 2000 (SIC)",
    "GLD.MX":  "Oro SPDR (SIC)",
    "SLV.MX":  "Plata iShares (SIC)",
    "TLT.MX":  "Bonos 20+ años (SIC)",
    "EEM.MX":  "Emergentes iShares (SIC)",
    "VWO.MX":  "Vanguard Emergentes (SIC)",
    "VEA.MX":  "Vanguard Desarrollados (SIC)",
    "IEMG.MX": "iShares Core Emergentes (SIC)",
    "EWZ.MX":  "Brasil iShares (SIC)",
    "XLE.MX":  "Sector Energía (SIC)",
    "XLF.MX":  "Sector Financiero (SIC)",
    "XLK.MX":  "Sector Tecnología (SIC)",
    "SMH.MX":  "Semiconductores VanEck (SIC)",
    "SOXX.MX": "Semiconductores iShares (SIC)",
    "AGG.MX":  "Bonos Agregados EE.UU. (SIC)",
    "SHV.MX":  "Bonos Corto Plazo (SIC)",
    "BIL.MX":  "T-Bills 1-3 meses (SIC)",
}


def _analizar_etf(args: tuple) -> dict | None:
    """Calcula la condición técnica de un ETF. Pensado para hilos paralelos.

    Solo usa history() + fast_info (sin .info): mínimo de peticiones a Yahoo.
    """
    ticker, nombre = args
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=PERIODO_HIST)
        if hist.empty or len(hist) < MIN_VELAS:
            return None

        close = hist["Close"]
        precio = close.iloc[-1]
        cambio_sem = ((precio - close.iloc[-6]) / close.iloc[-6]) * 100

        # RVOL robusto a sesión en curso (igual criterio que el radar de
        # acciones): el mayor entre la vela de hoy y la de ayer.
        vol_prom = hist["Volume"].iloc[-21:-1].mean()
        if vol_prom > 0:
            rvol = max(hist["Volume"].iloc[-1], hist["Volume"].iloc[-2]) / vol_prom
        else:
            rvol = 0.0

        fi = tk.fast_info
        sma200 = getattr(fi, "two_hundred_day_average", None) or 0
        max_52 = getattr(fi, "year_high", None) or close.max()
        moneda = getattr(fi, "currency", None) or "USD"

        vs_sma = ((precio - sma200) / sma200) * 100 if sma200 > 0 else None
        dist_max = ((precio - max_52) / max_52) * 100
        sobre_sma = vs_sma is not None and vs_sma >= 0

        # Score de fuerza: momentum pesa más, tendencia y cercanía al máximo
        # aportan contexto. Ordena de ETF más fuerte a más débil.
        score = (
            cambio_sem * 2.0
            + (vs_sma if vs_sma is not None else 0.0) * 0.3
            + dist_max * 0.2
        )

        # Condición: tendencia (SMA200) + momentum semanal
        if sobre_sma and cambio_sem > 0:
            condicion = "🟢 Fuerte"
        elif sobre_sma or cambio_sem > 0:
            condicion = "🟡 Neutral"
        else:
            condicion = "🔴 Débil"

        if vs_sma is None:
            tendencia = "N/A"
        elif sobre_sma:
            tendencia = f"+{vs_sma:.1f}% ✅"
        else:
            tendencia = f"{vs_sma:.1f}% ⚠️"

        return {
            "Ticker":     ticker,
            "Nombre":     nombre,
            "Precio":     f"{moneda} {precio:,.2f}",
            "Semana":     f"{cambio_sem:+.2f}%",
            "RVOL":       f"{rvol:.1f}x",
            "Tendencia":  tendencia,
            "vs Máx 52s": f"{dist_max:.1f}%",
            "Condición":  condicion,
            "_score":     score,
        }
    except Exception:
        return None


def escanear_etfs(catalogo: dict) -> pd.DataFrame:
    """Escanea un catálogo {ticker: nombre} y devuelve la tabla de condición.

    NO filtra: incluye todos los ETFs con datos, ordenados por fuerza
    (más fuerte arriba). Devuelve DataFrame vacío si nada respondió.
    """
    args = list(catalogo.items())
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        resultados = list(ex.map(_analizar_etf, args))

    filas = [r for r in resultados if r is not None]
    if not filas:
        return pd.DataFrame()

    df = pd.DataFrame(filas).sort_values("_score", ascending=False)
    return df.drop(columns=["_score"]).reset_index(drop=True)
