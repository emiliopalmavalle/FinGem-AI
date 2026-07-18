"""
validador_plan.py — Validación numérica del plan operativo de la IA
====================================================================

Recomendación central de la auditoría: "el LLM redacta; la aritmética
la haces tú". La IA emite al final de cada análisis un bloque JSON con
{sesgo, entrada, stop, tp1}; aquí se parsea y se verifica en Python:

  - Coherencia direccional: alcista → stop < entrada < tp1
                            bajista → tp1 < entrada < stop
  - Ratio Riesgo/Beneficio recalculado (no el que diga el texto)
  - Distancia del stop vs ATR (ni pegado al precio ni desproporcionado)

Si algo no cuadra, el reporte se anota con avisos visibles — nunca se
descarta silenciosamente.
"""

import json
import re

RB_MINIMO = 1.5  # debajo de esto el plan se marca como no operable


def extraer_plan(texto_analisis: str) -> tuple[dict | None, str]:
    """Extrae el bloque JSON del plan y lo retira del texto mostrado.

    Returns:
        (plan_dict | None, texto_sin_bloque_json)
    """
    patron = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
    m = patron.search(texto_analisis)
    if not m:
        # fallback: JSON suelto con la clave "entrada"
        m = re.search(r"\{[^{}]*\"entrada\"[^{}]*\}", texto_analisis, re.DOTALL)
        if not m:
            return None, texto_analisis
        crudo, texto_limpio = m.group(0), texto_analisis.replace(m.group(0), "").rstrip()
    else:
        crudo, texto_limpio = m.group(1), patron.sub("", texto_analisis).rstrip()

    try:
        plan = json.loads(crudo)
        return plan if isinstance(plan, dict) else None, texto_limpio
    except (json.JSONDecodeError, ValueError):
        return None, texto_limpio


def validar_plan(plan: dict, precio_actual: float, atr: float | None = None) -> dict:
    """Verifica la aritmética del plan. Devuelve métricas + avisos.

    Returns:
        dict con sesgo, entrada, stop, tp1, rb (o None), avisos (list[str])
        y operable (bool).
    """
    avisos = []
    sesgo   = str(plan.get("sesgo", "")).lower().strip()
    entrada = _num(plan.get("entrada"))
    stop    = _num(plan.get("stop"))
    tp1     = _num(plan.get("tp1"))

    resultado = {"sesgo": sesgo or "n/a", "entrada": entrada, "stop": stop,
                 "tp1": tp1, "rb": None, "avisos": avisos, "operable": False}

    if sesgo == "neutral" or not all(v and v > 0 for v in (entrada, stop, tp1)):
        avisos.append("ℹ️ La IA recomienda quedarse fuera (sin plan operativo).")
        return resultado

    # Coherencia direccional
    if sesgo == "alcista" and not (stop < entrada < tp1):
        avisos.append("⚠️ Plan INCOHERENTE: en un largo el orden debe ser stop < entrada < TP.")
        return resultado
    if sesgo == "bajista" and not (tp1 < entrada < stop):
        avisos.append("⚠️ Plan INCOHERENTE: en un corto el orden debe ser TP < entrada < stop.")
        return resultado

    # Ratio Riesgo/Beneficio recalculado en Python
    riesgo    = abs(entrada - stop)
    beneficio = abs(tp1 - entrada)
    if riesgo <= 0:
        avisos.append("⚠️ Stop igual a la entrada — plan inválido.")
        return resultado
    rb = round(beneficio / riesgo, 2)
    resultado["rb"] = rb

    if rb < RB_MINIMO:
        avisos.append(f"⚠️ R/B {rb} < {RB_MINIMO} — plan NO operable con gestión de riesgo sana.")
    else:
        resultado["operable"] = True

    # Distancia de la entrada al precio actual (>8% huele a nivel irreal)
    if precio_actual > 0 and abs(entrada - precio_actual) / precio_actual > 0.08:
        avisos.append(
            f"⚠️ Entrada a {abs(entrada - precio_actual) / precio_actual * 100:.1f}% del precio actual — "
            "es una orden condicional lejana, no un trade inmediato."
        )

    # Sanidad del stop vs volatilidad real
    if atr and atr > 0:
        stops_en_atr = riesgo / atr
        if stops_en_atr < 0.5:
            avisos.append(f"⚠️ Stop a solo {stops_en_atr:.1f}x ATR — el ruido normal lo barrería.")
        elif stops_en_atr > 3.5:
            avisos.append(f"⚠️ Stop a {stops_en_atr:.1f}x ATR — riesgo desproporcionado para el horizonte.")

    if resultado["operable"] and not avisos:
        avisos.append(f"✅ Plan verificado: aritmética coherente (R/B {rb}).")
    return resultado


def _num(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None  # descarta NaN
    except (TypeError, ValueError):
        return None
