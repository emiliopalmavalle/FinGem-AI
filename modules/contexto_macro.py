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

        # Guardias POR SÍMBOLO: antes, si Yahoo omitía uno solo (p. ej. ^IRX),
        # el KeyError tumbaba TODO el contexto macro — se perdían también VIX,
        # tasas y petróleo que sí habían llegado. Cada pieza cae por separado.
        def _serie(sym):
            try:
                s = data[sym]["Close"].dropna()
                return s if not s.empty else None
            except Exception:
                return None

        vix = _serie("^VIX"); tnx = _serie("^TNX"); irx = _serie("^IRX")
        dxy = _serie("DX-Y.NYB"); wti = _serie("CL=F"); oro = _serie("GLD")

        partes, res = [], {}

        if vix is not None:
            vix_now = float(vix.iloc[-1]); vix_chg = _variacion_5d(vix)
            if vix_now < 15:    regimen = "COMPLACENCIA (riesgo de sorpresas)"
            elif vix_now < 20:  regimen = "NORMAL"
            elif vix_now < 30:  regimen = "MIEDO (volatilidad elevada)"
            else:               regimen = "PÁNICO (capitulación posible)"
            partes.append(f"VIX: {vix_now:.1f} ({vix_chg:+.0f}% en 5d) — régimen {regimen}.")
            res.update({"vix": vix_now, "vix_chg_5d": vix_chg, "regimen_vix": regimen})

        if tnx is not None and irx is not None:
            t10 = float(tnx.iloc[-1]); t3m = float(irx.iloc[-1])
            spread = t10 - t3m
            if spread < 0:
                curva = f"INVERTIDA ({spread:+.2f} pts) — señal histórica de recesión"
            elif spread < 0.5:
                curva = f"PLANA ({spread:+.2f} pts) — ciclo maduro"
            else:
                curva = f"NORMAL ({spread:+.2f} pts) — expansión"
            partes.append(f"Tasa 10 años: {t10:.2f}% | Tasa 3 meses: {t3m:.2f}% | Curva: {curva}.")
            res.update({"tasa_10y": t10, "tasa_3m": t3m, "spread_curva": spread, "curva": curva})

        if dxy is not None:
            dxy_now = float(dxy.iloc[-1]); dxy_chg = _variacion_5d(dxy)
            partes.append(f"Dólar DXY: {dxy_now:.1f} ({dxy_chg:+.1f}% 5d).")
            res["dxy"] = dxy_now

        if wti is not None:
            wti_now = float(wti.iloc[-1]); wti_chg = _variacion_5d(wti)
            partes.append(f"Petróleo WTI: USD {wti_now:.1f} ({wti_chg:+.1f}% 5d).")
            res.update({"wti": wti_now, "wti_chg_5d": wti_chg})

        if oro is not None:
            partes.append(f"Oro (GLD): {_variacion_5d(oro):+.1f}% 5d.")

        res["texto"] = " ".join(partes)
        return res
    except Exception:
        return {"texto": ""}


def mostrar_metricas_macro(ctx: dict) -> None:
    """Fila de métricas macro para la UI (llamar dentro del módulo de análisis)."""
    if not ctx.get("texto"):
        return
    # Accesos .get(): con la resiliencia por símbolo, cualquier clave puede
    # faltar si Yahoo omitió ese ticker — se muestra N/A en vez de crashear
    c1, c2, c3, c4 = st.columns(4)
    vix, spread, wti = ctx.get("vix"), ctx.get("spread_curva"), ctx.get("wti")
    c1.metric("😱 VIX", f"{vix:.1f}" if vix is not None else "N/A",
              f"{ctx.get('vix_chg_5d', 0):+.0f}% 5d" if vix is not None else None,
              delta_color="inverse")
    c2.metric("🏦 Tasa 10 años", f"{ctx['tasa_10y']:.2f}%" if ctx.get("tasa_10y") is not None else "N/A")
    c3.metric("📉 Curva 10a-3m", f"{spread:+.2f}" if spread is not None else "N/A",
              ("Invertida ⚠️" if spread < 0 else "Normal") if spread is not None else None)
    c4.metric("🛢️ WTI", f"${wti:.1f}" if wti is not None else "N/A",
              f"{ctx.get('wti_chg_5d', 0):+.1f}% 5d" if wti is not None else None,
              delta_color="inverse")
