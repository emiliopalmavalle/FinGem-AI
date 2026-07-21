"""
validador_plan.py — Validación numérica del plan operativo de la IA
====================================================================

Recomendación central de la auditoría: "el LLM redacta; la aritmética
la haces tú". La IA emite al final de cada análisis un bloque JSON con
{sesgo, direccion, entrada, stop, tp1}; aquí se parsea y verifica:

  - "sesgo"     = dirección probable del MERCADO (informativo)
  - "direccion" = la OPERACIÓN del plan: largo | corto | fuera
    Pueden diferir: sesgo bajista con largo táctico de rebote es válido.
    Si la IA omite "direccion", se INFIERE del orden de los niveles.
  - Coherencia direccional: largo → stop < entrada < tp1
                            corto → tp1 < entrada < stop
  - Ratio Riesgo/Beneficio recalculado (no el que diga el texto)
  - Distancia del stop vs ATR (solo si el caller pasa un ATR aplicable)

Si algo no cuadra, el reporte se anota con avisos visibles — nunca se
descarta silenciosamente.
"""

import json
import re

RB_MINIMO = 1.5  # debajo de esto el plan se marca como no operable

# Bloque de disciplina de riesgo para TODOS los prompts que piden un plan
# (Análisis Individual, Derivados y Radar de Opciones). Vive aquí junto a
# RB_MINIMO para que lo que se le exige a la IA y lo que verifica el código
# nunca puedan discrepar.
REGLAS_RIESGO_PROMPT = f"""
DISCIPLINA DE RIESGO — CONDICIONES PREVIAS AL PLAN (obligatorias):
- CALCULA el ratio R/B = (TP − entrada) / (entrada − stop) ANTES de proponer nada.
  Si con niveles REALES de estructura no alcanza {RB_MINIMO}, la respuesta correcta es
  "direccion": "fuera", y dilo explícitamente: "la estructura no ofrece un R/B aceptable".
- PROHIBIDO fabricar el ratio: no alejes el TP ni acerques el stop para que salga el número.
  Los niveles los manda la estructura del precio, no la aritmética que te conviene.
- Si tu convicción es BAJA, o el sesgo es NEUTRAL sin catalizador claro, la recomendación
  por defecto es "fuera". No tienes ninguna obligación de encontrar una operación:
  quedarse fuera es una conclusión profesional legítima y FRECUENTE. Un reporte que
  concluye "hoy aquí no hay nada" vale tanto como uno que encuentra un setup.
- NO propongas un LARGO mientras describes el contexto como bajista o lateral-bajista
  (ni un CORTO en contexto alcista), salvo que justifiques un catalizador táctico concreto.
- Indica el % DE MOVIMIENTO que necesita el subyacente desde la entrada hasta el TP.
  Es el dato que decide si una opción puede capturarlo o si el spread y el theta se lo comen.
"""


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


def _inferir_direccion(entrada: float, stop: float, tp1: float) -> str | None:
    """Infiere la dirección de la operación del orden de los niveles."""
    if stop < entrada < tp1:
        return "largo"
    if tp1 < entrada < stop:
        return "corto"
    return None


def validar_plan(plan: dict, precio_actual: float, atr: float | None = None) -> dict:
    """Verifica la aritmética del plan. Devuelve métricas + avisos.

    Args:
        plan: dict con sesgo, direccion, entrada, stop, tp1.
        precio_actual: spot para el chequeo de entrada lejana.
        atr: ATR aplicable al horizonte del plan, o None para omitir el
             chequeo (p. ej. Semanal/Mensual, donde el stop se ancla a
             estructura y el ATR de vela alta daría falsos avisos).

    Returns:
        dict con sesgo, direccion, entrada, stop, tp1, rb (o None),
        avisos (list[str]) y operable (bool).
    """
    avisos = []
    sesgo     = str(plan.get("sesgo", "")).lower().strip()
    direccion = str(plan.get("direccion", "")).lower().strip()
    entrada = _num(plan.get("entrada"))
    stop    = _num(plan.get("stop"))
    tp1     = _num(plan.get("tp1"))

    resultado = {"sesgo": sesgo or "n/a", "direccion": direccion or "n/a",
                 "entrada": entrada, "stop": stop, "tp1": tp1,
                 "rb": None, "avisos": avisos, "operable": False}

    sin_niveles = not all(v and v > 0 for v in (entrada, stop, tp1))
    if direccion == "fuera" or (not direccion and sesgo == "neutral") or sin_niveles:
        resultado["direccion"] = "fuera"
        avisos.append("ℹ️ La IA recomienda quedarse fuera (sin plan operativo).")
        return resultado

    # Dirección de la operación: campo explícito, o inferida de los niveles
    # (compatibilidad con reportes viejos donde solo venía "sesgo")
    inferida = _inferir_direccion(entrada, stop, tp1)
    if direccion not in ("largo", "corto"):
        if inferida is None:
            avisos.append("⚠️ Plan INCOHERENTE: entrada, stop y TP no definen "
                          "ni un largo (stop < entrada < TP) ni un corto (TP < entrada < stop).")
            return resultado
        direccion = inferida
        resultado["direccion"] = direccion
        avisos.append(f"ℹ️ Dirección no declarada por la IA — inferida de los niveles: {direccion.upper()}.")
    elif inferida != direccion:
        orden = "stop < entrada < TP" if direccion == "largo" else "TP < entrada < stop"
        avisos.append(f"⚠️ Plan INCOHERENTE: la IA declara {direccion.upper()} "
                      f"pero los niveles no cumplen {orden}.")
        return resultado

    # Aviso informativo (no error) cuando la operación va contra el sesgo:
    # es un contra-tendencia legítimo, pero el lector debe saberlo
    if sesgo in ("alcista", "bajista"):
        contra = (sesgo == "bajista" and direccion == "largo") or \
                 (sesgo == "alcista" and direccion == "corto")
        if contra:
            avisos.append(f"ℹ️ Operación CONTRA-TENDENCIA: {direccion} táctico con sesgo "
                          f"de mercado {sesgo} — tamaño reducido y disciplina estricta de stop.")

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

    # Sanidad del stop vs volatilidad real (solo si el caller pasó un ATR aplicable)
    if atr and atr > 0:
        stops_en_atr = riesgo / atr
        if stops_en_atr < 0.5:
            avisos.append(f"⚠️ Stop a solo {stops_en_atr:.1f}x ATR — el ruido normal lo barrería.")
        elif stops_en_atr > 3.5:
            avisos.append(f"⚠️ Stop a {stops_en_atr:.1f}x ATR — riesgo desproporcionado para el horizonte.")

    if resultado["operable"] and not any(a.startswith("⚠️") for a in avisos):
        avisos.append(f"✅ Plan verificado: aritmética coherente (R/B {rb}).")
    return resultado


def _num(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None  # descarta NaN
    except (TypeError, ValueError):
        return None
