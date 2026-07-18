"""
contexto_macro.py — Análisis Macroeconómico Top-Down y Sentimiento
===================================================================

Ensambla el contexto macro global con datos reales de Yahoo Finance
para que la IA analice de arriba hacia abajo (Top-Down):

  1. MACRO:       tasas de interés (10 años ^TNX, corto plazo ^IRX),
                  curva de rendimientos (invertida = señal de recesión),
                  proxies de inflación (petróleo WTI, oro, dólar DXY)
  2. SENTIMIENTO: VIX con régimen (complacencia / normal / miedo / pánico)

Cacheado 1 hora (st.cache_data): el macro no cambia por minuto y así
el análisis individual no paga latencia extra en cada click.
"""

import pandas as pd
import streamlit as st

SIMBOLOS_MACRO = ["^VIX", "^TNX", "^IRX", "DX-Y.NYB", "CL=F", "GLD"]


def _variacion_5d(cierres: pd.Series) -> float:
    """% de cambio vs hace 5 sesiones (0.0 si no hay historia suficiente)."""
    if len(cierres) < 6:
        return 0.0
    return float((cierres.iloc[-1] - cierres.iloc[-6]) / cierres.iloc[-6] * 100)


@st.cache_data(ttl=3600, show_spinner=False)
def obtener_contexto_macro_global() -> dict:
    """Descarga y resume el contexto macro global en una sola llamada batch.

    Returns:
        dict con métricas numéricas + 'texto' listo para inyectar al prompt.
        Si Yahoo falla, devuelve dict con 'texto' vacío (el análisis sigue).
    """
    import yfinance as yf
    try:
        data = yf.download(
            SIMBOLOS_MACRO, period="10d", progress=False,
            group_by="ticker", threads=True,
        )

        def _serie(sym):
            return data[sym]["Close"].dropna()

        vix   = _serie("^VIX");    vix_now = float(vix.iloc[-1]);  vix_chg = _variacion_5d(vix)
        tnx   = _serie("^TNX");    t10 = float(tnx.iloc[-1])
        irx   = _serie("^IRX");    t3m = float(irx.iloc[-1])
        dxy   = _serie("DX-Y.NYB"); dxy_now = float(dxy.iloc[-1]); dxy_chg = _variacion_5d(dxy)
        wti   = _serie("CL=F");    wti_now = float(wti.iloc[-1]);  wti_chg = _variacion_5d(wti)
        oro   = _serie("GLD");     oro_chg = _variacion_5d(oro)

        # Régimen de sentimiento según VIX
        if vix_now < 15:    regimen = "COMPLACENCIA (riesgo de sorpresas)"
        elif vix_now < 20:  regimen = "NORMAL"
        elif vix_now < 30:  regimen = "MIEDO (volatilidad elevada)"
        else:               regimen = "PÁNICO (capitulación posible)"

        # Curva de rendimientos: 10 años vs 3 meses
        spread = t10 - t3m
        if spread < 0:
            curva = f"INVERTIDA ({spread:+.2f} pts) — señal histórica de recesión"
        elif spread < 0.5:
            curva = f"PLANA ({spread:+.2f} pts) — ciclo maduro"
        else:
            curva = f"NORMAL ({spread:+.2f} pts) — expansión"

        texto = (
            f"VIX: {vix_now:.1f} ({vix_chg:+.0f}% en 5d) — régimen {regimen}. "
            f"Tasa 10 años: {t10:.2f}% | Tasa 3 meses: {t3m:.2f}% | Curva: {curva}. "
            f"Dólar DXY: {dxy_now:.1f} ({dxy_chg:+.1f}% 5d). "
            f"Petróleo WTI: USD {wti_now:.1f} ({wti_chg:+.1f}% 5d). "
            f"Oro (GLD): {oro_chg:+.1f}% 5d."
        )

        return {
            "vix": vix_now, "vix_chg_5d": vix_chg, "regimen_vix": regimen,
            "tasa_10y": t10, "tasa_3m": t3m, "spread_curva": spread, "curva": curva,
            "dxy": dxy_now, "wti": wti_now, "wti_chg_5d": wti_chg,
            "texto": texto,
        }
    except Exception:
        return {"texto": ""}


def mostrar_metricas_macro(ctx: dict) -> None:
    """Fila de métricas macro para la UI (llamar dentro del módulo de análisis)."""
    if not ctx.get("texto"):
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("😱 VIX", f"{ctx['vix']:.1f}", f"{ctx['vix_chg_5d']:+.0f}% 5d", delta_color="inverse")
    c2.metric("🏦 Tasa 10 años", f"{ctx['tasa_10y']:.2f}%")
    c3.metric("📉 Curva 10a-3m", f"{ctx['spread_curva']:+.2f}",
              "Invertida ⚠️" if ctx['spread_curva'] < 0 else "Normal")
    c4.metric("🛢️ WTI", f"${ctx['wti']:.1f}", f"{ctx['wti_chg_5d']:+.1f}% 5d", delta_color="inverse")
