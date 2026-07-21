"""
opciones_cboe.py — Cadena de opciones real (CBOE + Black-Scholes de respaldo)
=============================================================================

MOTIVO: los datos de opciones de yfinance están rotos. Verificado en julio
2026 sobre AAPL/SPY/QQQ/TSLA/NVDA, todos los vencimientos:
    openInterest      → 0 en TODA la cadena
    bid / ask         → 0.0
    impliedVolatility → 0.00001 (relleno basura)
    volume, lastPrice → sí funcionan

Eso rompía en silencio los muros (call/put wall), el Max Pain y — peor —
generaba FALSOS POSITIVOS de "flujo fresco": la condición `volume > openInterest`
es siempre cierta cuando el OI es 0, así que cada strike líquido se reportaba
como posicionamiento institucional nuevo, y eso se inyectaba en el prompt de la IA.

SOLUCIÓN: el endpoint público de CBOE (sin API key, sin límite de peticiones)
devuelve en UNA sola llamada la cadena completa de TODOS los vencimientos con
open_interest, greeks, IV, bid/ask y volumen reales.

Cobertura medida en AAPL (3.476 contratos): OI 78%, delta 98%, IV 92%.
Para el ~20% de contratos sin greeks (los ilíquidos), se calculan con
Black-Scholes local desde el mid del bid/ask.

Convenciones (idénticas a las de CBOE, para poder mezclar ambas fuentes):
    theta → pérdida de valor POR DÍA
    vega  → cambio de precio por cada 1% de movimiento de la IV
"""

import re
import math
import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional

# ==========================================
# ⚙️ CONFIGURACIÓN
# ==========================================
_URL_CBOE = "https://cdn.cboe.com/api/global/delayed_quotes/options/{}.json"
_TIMEOUT  = 25
TASA_LIBRE_DEFECTO = 0.04   # tasa libre de riesgo anual para Black-Scholes

# Símbolo OCC: SUBYACENTE + YYMMDD + (C|P) + strike*1000 en 8 dígitos
_PATRON_OCC = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})")


def _simbolo_cboe(simbolo: str) -> str:
    """Traduce el ticker al formato de CBOE.

    Los índices llevan guion bajo delante (^SPX → _SPX); las acciones y
    ETFs van tal cual.
    """
    s = (simbolo or "").upper().strip()
    if s.startswith("^"):
        return "_" + s[1:]
    return s


# ==========================================
# 🧮 BLACK-SCHOLES (respaldo, sin dependencias nuevas)
# ==========================================
def _norm_cdf(x: float) -> float:
    """N(x) — normal acumulada vía math.erf (evita depender de scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """n(x) — densidad normal estándar."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    vol_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_t
    return d1, d1 - vol_t


def _precio_bs(S: float, K: float, T: float, r: float, sigma: float, es_call: bool) -> float:
    """Precio teórico Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if es_call else (K - S))
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if es_call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def calcular_iv(precio_mercado: float, S: float, K: float, T: float,
                r: float, es_call: bool) -> Optional[float]:
    """Despeja la volatilidad implícita desde el precio real de mercado.

    Newton-Raphson con bisección de respaldo. Devuelve None cuando el
    precio es imposible (por debajo del valor intrínseco) o el contrato
    está muerto de liquidez.
    """
    if precio_mercado <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    intrinseco = max(0.0, (S - K) if es_call else (K - S))
    if precio_mercado < intrinseco - 0.01:
        return None

    sigma = 0.30  # semilla razonable para acciones
    for _ in range(50):
        precio = _precio_bs(S, K, T, r, sigma, es_call)
        d1, _  = _d1_d2(S, K, T, r, sigma)
        vega   = S * _norm_pdf(d1) * math.sqrt(T)
        if vega < 1e-8:
            break
        sigma_nuevo = sigma - (precio - precio_mercado) / vega
        if sigma_nuevo <= 0:
            break
        if abs(sigma_nuevo - sigma) < 1e-6:
            return sigma_nuevo if 0.001 < sigma_nuevo < 5.0 else None
        sigma = sigma_nuevo

    bajo, alto = 0.001, 5.0
    if _precio_bs(S, K, T, r, alto, es_call) < precio_mercado:
        return None
    for _ in range(100):
        medio = (bajo + alto) / 2.0
        if _precio_bs(S, K, T, r, medio, es_call) < precio_mercado:
            bajo = medio
        else:
            alto = medio
        if alto - bajo < 1e-6:
            break
    resultado = (bajo + alto) / 2.0
    return resultado if 0.002 < resultado < 4.99 else None


def calcular_greeks(S: float, K: float, T: float, r: float,
                    sigma: float, es_call: bool) -> dict:
    """Delta, gamma, theta (por día) y vega (por 1% de IV)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan}

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    raiz_t = math.sqrt(T)

    delta = _norm_cdf(d1) if es_call else _norm_cdf(d1) - 1.0
    gamma = _norm_pdf(d1) / (S * sigma * raiz_t)
    vega  = S * _norm_pdf(d1) * raiz_t / 100.0

    theta_comun = -(S * _norm_pdf(d1) * sigma) / (2.0 * raiz_t)
    if es_call:
        theta = theta_comun - r * K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        theta = theta_comun + r * K * math.exp(-r * T) * _norm_cdf(-d2)

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta / 365.0, 4),
        "vega":  round(vega, 4),
    }


# ==========================================
# 📡 DESCARGA DESDE CBOE
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def descargar_cadena_cruda(simbolo: str) -> dict:
    """Baja la cadena COMPLETA (todos los vencimientos) en una sola llamada.

    Cacheada 5 minutos: el mismo activo se consulta desde varios módulos
    y el feed es retardado 15 min, así que no tiene sentido re-pedirlo.

    Returns:
        dict con 'spot' (float) y 'contratos' (list[dict]); vacío si falla.
    """
    try:
        url = _URL_CBOE.format(_simbolo_cboe(simbolo))
        resp = requests.get(url, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return {}
        datos = resp.json().get("data", {}) or {}
        return {
            "spot":      float(datos.get("current_price") or 0),
            "contratos": datos.get("options", []) or [],
        }
    except Exception:
        return {}


def salud_datos(simbolo: str) -> dict:
    """Cobertura real de la cadena descargada — detector de degradación.

    Los datos de opciones no fallan con un error: fallan devolviendo ceros
    con apariencia de normalidad (fue justo lo que pasó con yfinance). Medir
    la cobertura convierte esa avería silenciosa en algo visible.

    No cuesta peticiones: recorre la cadena que ya está en caché.

    Returns:
        dict con fuente, nº de contratos, % con OI, % con greeks, % con IV
        y 'nivel' ('ok' | 'degradado' | 'caido') para pintar la UI.
    """
    cruda = descargar_cadena_cruda(simbolo)
    contratos = cruda.get("contratos", [])
    n = len(contratos)

    if n == 0:
        return {"fuente": "CBOE", "contratos": 0, "pct_oi": 0.0,
                "pct_greeks": 0.0, "pct_iv": 0.0, "nivel": "caido"}

    con_oi = sum(1 for c in contratos if (c.get("open_interest") or 0) > 0)
    con_gk = sum(1 for c in contratos if (c.get("delta") or 0) != 0)
    con_iv = sum(1 for c in contratos if (c.get("iv") or 0) > 0)
    pct_oi, pct_gk, pct_iv = con_oi / n * 100, con_gk / n * 100, con_iv / n * 100

    # Umbrales calibrados sobre la cobertura medida en julio 2026
    # (OI ~78%, greeks ~98%): el resto son contratos muertos sin precio
    if pct_oi >= 50 and pct_gk >= 80:
        nivel = "ok"
    else:
        nivel = "degradado"

    return {"fuente": "CBOE", "contratos": n, "pct_oi": round(pct_oi, 1),
            "pct_greeks": round(pct_gk, 1), "pct_iv": round(pct_iv, 1),
            "nivel": nivel}


def mostrar_salud_datos(simbolo: str) -> None:
    """Pinta una línea discreta con el estado de la fuente de opciones.

    Verde/gris = normal · amarillo = cobertura caída · rojo = sin datos.
    """
    s = salud_datos(simbolo)

    if s["nivel"] == "caido":
        st.error(
            f"🔴 **Sin datos de opciones para {simbolo}.** O el activo no tiene opciones "
            f"listadas (BMV, cripto), o la fuente CBOE dejó de responder. "
            f"Los muros, Max Pain y greeks NO son fiables ahora mismo."
        )
    elif s["nivel"] == "degradado":
        st.warning(
            f"🟡 **Cobertura de datos baja en {simbolo}** — CBOE · {s['contratos']:,} contratos · "
            f"OI {s['pct_oi']}% · greeks {s['pct_greeks']}% · IV {s['pct_iv']}%. "
            f"Lo normal es OI ~78% y greeks ~98%: revisa antes de operar estos niveles."
        )
    else:
        st.caption(
            f"📡 Opciones: **CBOE** · {s['contratos']:,} contratos · "
            f"OI {s['pct_oi']}% · greeks {s['pct_greeks']}% · IV {s['pct_iv']}% · retardo ~15 min"
        )


def mostrar_salud_datos_lista(simbolos: list[str]) -> None:
    """Versión agregada para los módulos que escanean varios activos.

    Una línea por ticker sería ruido; lo que importa aquí es distinguir
    "este activo es ilíquido" (normal) de "la fuente se cayó" (todos en 0).
    """
    if not simbolos:
        return
    saludes = [salud_datos(s) for s in simbolos]
    con_datos = [s for s in saludes if s["contratos"] > 0]

    if not con_datos:
        st.error(
            f"🔴 **Ningún activo devolvió datos de opciones** ({len(simbolos)} consultados). "
            f"Probablemente la fuente CBOE dejó de responder: los resultados de abajo "
            f"NO son fiables."
        )
        return

    pct_oi = sum(s["pct_oi"] for s in con_datos) / len(con_datos)
    pct_gk = sum(s["pct_greeks"] for s in con_datos) / len(con_datos)
    total  = sum(s["contratos"] for s in con_datos)
    aviso  = "" if len(con_datos) == len(simbolos) else \
             f" · ⚠️ {len(simbolos) - len(con_datos)} sin opciones listadas"

    texto = (f"📡 Opciones: **CBOE** · {len(con_datos)}/{len(simbolos)} activos · "
             f"{total:,} contratos · OI {pct_oi:.0f}% · greeks {pct_gk:.0f}% · "
             f"retardo ~15 min{aviso}")
    (st.warning if (pct_oi < 50 or pct_gk < 80) else st.caption)(texto)


def _partes_occ(simbolo_occ: str) -> Optional[dict]:
    """Descompone un símbolo OCC en vencimiento, tipo y strike."""
    m = _PATRON_OCC.match(simbolo_occ or "")
    if not m:
        return None
    try:
        return {
            "vencimiento": datetime.datetime.strptime(m.group(2), "%y%m%d").date(),
            "tipo":        "call" if m.group(3) == "C" else "put",
            "strike":      int(m.group(4)) / 1000.0,
        }
    except Exception:
        return None


def vencimientos_disponibles(simbolo: str) -> list[str]:
    """Lista de vencimientos ('YYYY-MM-DD') que CBOE tiene para el activo."""
    cruda = descargar_cadena_cruda(simbolo)
    fechas = set()
    for contrato in cruda.get("contratos", []):
        partes = _partes_occ(contrato.get("option", ""))
        if partes:
            fechas.add(partes["vencimiento"].strftime("%Y-%m-%d"))
    return sorted(fechas)


def precio_spot(simbolo: str) -> Optional[float]:
    """Precio del subyacente según CBOE (evita una llamada extra a Yahoo)."""
    spot = descargar_cadena_cruda(simbolo).get("spot") or 0
    return float(spot) if spot > 0 else None


# ==========================================
# 🔗 CADENA EN FORMATO yfinance (API PÚBLICA)
# ==========================================
class CadenaOpciones:
    """Contenedor con la misma forma que yfinance.option_chain().

    Atributos:
        calls / puts: DataFrames con las columnas de yfinance + greeks reales.
        fuente:  'cboe' si se pudo descargar, 'vacio' si no.
        rellenos: nº de contratos cuyos greeks se calcularon con Black-Scholes.
    """

    def __init__(self, calls: pd.DataFrame, puts: pd.DataFrame,
                 fuente: str = "cboe", rellenos: int = 0):
        self.calls    = calls
        self.puts     = puts
        self.fuente   = fuente
        self.rellenos = rellenos


# Columnas que los módulos existentes esperan encontrar (formato yfinance)
_COLUMNAS = [
    "contractSymbol", "strike", "lastPrice", "bid", "ask", "volume",
    "openInterest", "impliedVolatility", "inTheMoney",
    "delta", "gamma", "theta", "vega",
]


def cadena_cboe(
    simbolo: str,
    fecha_venc: str,
    spot: Optional[float] = None,
    tasa: float = TASA_LIBRE_DEFECTO,
) -> CadenaOpciones:
    """Cadena de un vencimiento con OI, greeks e IV reales.

    Rellena con Black-Scholes los contratos donde CBOE no trae greeks
    (típicamente los ilíquidos, ~20% de la cadena).

    Args:
        simbolo: subyacente (ej. "AAPL").
        fecha_venc: vencimiento "YYYY-MM-DD".
        spot: precio del subyacente; si es None se toma el de CBOE.
        tasa: tasa libre de riesgo anual (0.04 = 4%).

    Returns:
        CadenaOpciones con .calls y .puts en formato yfinance.
    """
    cruda = descargar_cadena_cruda(simbolo)
    contratos = cruda.get("contratos", [])
    if not contratos:
        return CadenaOpciones(pd.DataFrame(), pd.DataFrame(), fuente="vacio")

    S = float(spot or cruda.get("spot") or 0)

    # Años al vencimiento (mínimo 1 hora para no dividir entre cero en 0DTE)
    try:
        dias = (datetime.datetime.strptime(fecha_venc, "%Y-%m-%d").date()
                - datetime.date.today()).days
    except Exception:
        dias = 1
    T = max(max(dias, 0) / 365.0, 1.0 / (365.0 * 24.0))

    filas_call, filas_put, rellenos = [], [], 0

    for contrato in contratos:
        partes = _partes_occ(contrato.get("option", ""))
        if not partes or partes["vencimiento"].strftime("%Y-%m-%d") != fecha_venc:
            continue

        es_call = partes["tipo"] == "call"
        strike  = partes["strike"]
        bid     = float(contrato.get("bid") or 0)
        ask     = float(contrato.get("ask") or 0)
        iv      = float(contrato.get("iv") or 0)
        delta   = float(contrato.get("delta") or 0)

        # Mid del bid/ask = mejor estimador del valor justo; si no hay
        # cotización usable, caemos al último precio operado
        mid = ((bid + ask) / 2.0) if (bid > 0 and ask >= bid) else float(
            contrato.get("last_trade_price") or 0
        )

        # Relleno Black-Scholes cuando CBOE no calculó greeks/IV
        gamma = float(contrato.get("gamma") or 0)
        theta = float(contrato.get("theta") or 0)
        vega  = float(contrato.get("vega") or 0)
        if (iv <= 0 or delta == 0) and S > 0 and mid > 0:
            iv_calc = calcular_iv(mid, S, strike, T, tasa, es_call)
            if iv_calc:
                g = calcular_greeks(S, strike, T, tasa, iv_calc, es_call)
                iv    = iv_calc
                delta = g["delta"]
                gamma = g["gamma"] if gamma == 0 else gamma
                theta = g["theta"] if theta == 0 else theta
                vega  = g["vega"]  if vega  == 0 else vega
                rellenos += 1

        fila = {
            "contractSymbol":    contrato.get("option", ""),
            "strike":            strike,
            "lastPrice":         float(contrato.get("last_trade_price") or 0),
            "bid":               bid,
            "ask":               ask,
            "volume":            float(contrato.get("volume") or 0),
            "openInterest":      float(contrato.get("open_interest") or 0),
            "impliedVolatility": iv if iv > 0 else np.nan,
            "inTheMoney":        (S > strike) if es_call else (S < strike),
            "delta":             delta if delta != 0 else np.nan,
            "gamma":             gamma if gamma != 0 else np.nan,
            "theta":             theta if theta != 0 else np.nan,
            "vega":              vega  if vega  != 0 else np.nan,
        }
        (filas_call if es_call else filas_put).append(fila)

    def _construir(filas: list) -> pd.DataFrame:
        if not filas:
            return pd.DataFrame(columns=_COLUMNAS)
        return pd.DataFrame(filas)[_COLUMNAS].sort_values("strike").reset_index(drop=True)

    return CadenaOpciones(
        _construir(filas_call), _construir(filas_put),
        fuente="cboe", rellenos=rellenos,
    )
