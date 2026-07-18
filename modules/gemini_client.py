"""
gemini_client.py — v2 (Thread-Safe + LRU Mejorado)
====================================================

Cliente centralizado de Gemini para toda la terminal.

Problemas que resuelve:
  - Free Tier con cupo diario limitado → el sistema anterior lo agotaba rápido
  - Error 429 RESOURCE_EXHAUSTED → la app se rompía sin recuperación
  - Cada módulo creaba su propio cliente genai.Client → sin coordinación
  - Race conditions en ThreadPoolExecutor al escribir session_state

Soluciones implementadas:
  1. Retry automático con backoff exponencial (respeta el retryDelay del error)
  2. Caché en st.session_state: mismo prompt = mismo resultado (0 requests extra)
  3. Contador de requests visible en la UI (por sesión)
  4. Fallback local: si se agota el cupo, genera un reporte estructurado
     con los datos crudos sin llamar a la API
  5. Selector de modelo: si 2.5-flash está saturado, cae a 2.5-flash-lite / 2.0-flash
  6. [v2] threading.Lock para sincronización thread-safe del caché y contador
  7. [v2] LRU mejorado con evicción de la entrada más antigua al superar límite

Changelog v2:
  - _gemini_lock global protege todas las escrituras a session_state
  - Lectura y escritura de caché envuelta en bloque with _gemini_lock
  - Incremento de contador atómico con el lock
  - LRU con collections.OrderedDict para evicción O(1) de entradas antiguas
"""

import time
import hashlib
import threading
import re
from collections import OrderedDict
import streamlit as st
from google import genai
from google.genai import errors as genai_errors

# ── Modelos en orden de preferencia (fallback automático)
# Nota: la serie gemini-1.5 fue retirada por Google para API keys nuevas.
MODELOS_PREFERENCIA = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",  # más ligero, cuota separada
    "gemini-2.0-flash",
]

# ── Configuración de retry
MAX_REINTENTOS  = 3
BACKOFF_BASE    = 2.0   # segundos base para backoff exponencial
BACKOFF_MAX     = 60.0  # cap máximo de espera

# ── Clave de session_state para el contador
_KEY_CONTADOR   = "_gemini_requests_hoy"
_KEY_CACHE      = "_gemini_cache"
_KEY_AGOTADO    = "_gemini_cupo_agotado"

# ── Límite máximo de entradas en caché (evita memory leak)
_CACHE_MAX_SIZE = 50

# ── Lock global para acceso thread-safe a session_state
# Protege escrituras concurrentes desde ThreadPoolExecutor
# (Streamlit session_state no es inherentemente thread-safe)
_gemini_lock = threading.Lock()


# ══════════════════════════════════════════════════════
# INICIALIZACIÓN DEL SESSION STATE
# ══════════════════════════════════════════════════════

def _init_state():
    """Inicializa las claves de session_state si no existen.

    Usa OrderedDict para el caché, habilitando evicción LRU en O(1)
    cuando se supera _CACHE_MAX_SIZE.
    """
    if _KEY_CONTADOR not in st.session_state:
        st.session_state[_KEY_CONTADOR] = 0
    if _KEY_CACHE not in st.session_state:
        st.session_state[_KEY_CACHE] = OrderedDict()
    if _KEY_AGOTADO not in st.session_state:
        st.session_state[_KEY_AGOTADO] = False


# ══════════════════════════════════════════════════════
# CACHÉ DE PROMPTS (THREAD-SAFE)
# ══════════════════════════════════════════════════════

def _hash_prompt(prompt: str) -> str:
    """Hash MD5 del prompt para usar como clave de caché."""
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def _get_cached(prompt: str) -> str | None:
    """Lectura thread-safe del caché.

    Args:
        prompt: texto del prompt a buscar.

    Returns:
        Respuesta cacheada o None si no existe.
    """
    _init_state()
    key = _hash_prompt(prompt)
    with _gemini_lock:
        cache = st.session_state[_KEY_CACHE]
        if key in cache:
            # Mover al final para LRU (acceso reciente = menos prioritario para evicción)
            cache.move_to_end(key)
            return cache[key]
    return None


def _set_cached(prompt: str, respuesta: str):
    """Escritura thread-safe en caché con evicción LRU.

    Si el caché excede _CACHE_MAX_SIZE, elimina la entrada más antigua
    (first in OrderedDict) en O(1).

    Args:
        prompt: texto del prompt (se hashea como clave).
        respuesta: texto de la respuesta a cachear.
    """
    _init_state()
    key = _hash_prompt(prompt)
    with _gemini_lock:
        cache = st.session_state[_KEY_CACHE]
        cache[key] = respuesta
        cache.move_to_end(key)
        # Evicción LRU: eliminar entradas más antiguas
        while len(cache) > _CACHE_MAX_SIZE:
            cache.popitem(last=False)


# ══════════════════════════════════════════════════════
# EXTRACCIÓN DEL DELAY DEL ERROR 429
# ══════════════════════════════════════════════════════

def _extraer_retry_delay(error) -> float:
    """Extrae el retryDelay sugerido por la API del error 429.

    Busca patrones como "retryDelay: '23s'" o "retry in 23.6s"
    en el mensaje de error. Si no encuentra, usa backoff base.

    Args:
        error: excepción capturada del cliente Gemini.

    Returns:
        float con los segundos de espera sugeridos (con +2s de margen).
    """
    try:
        msg = str(error)
        match = re.search(r"retry[^0-9]*(\d+\.?\d*)\s*s", msg, re.IGNORECASE)
        if match:
            return min(float(match.group(1)) + 2.0, BACKOFF_MAX)
    except Exception:
        pass
    return BACKOFF_BASE


# ══════════════════════════════════════════════════════
# FALLBACK LOCAL (sin API)
# ══════════════════════════════════════════════════════

def _generar_fallback(contexto: dict) -> str:
    """Genera un reporte estructurado con los datos disponibles sin llamar a la API.

    Se activa cuando el cupo está agotado o todos los modelos/reintentos fallan.

    Args:
        contexto: dict con claves del llamador. Esperado: ticker, precio,
                  senal, score_alcista, score_bajista, squeeze, rsi,
                  ema8, ema21, razones, mejor_call_strike, mejor_put_strike,
                  pcr, iv_avg.

    Returns:
        str con el reporte formateado.
    """
    ticker         = contexto.get("ticker",          "N/A")
    precio         = contexto.get("precio",          0)
    senal          = contexto.get("senal",           "WAIT")
    score_call     = contexto.get("score_alcista",   0)
    score_put      = contexto.get("score_bajista",   0)
    squeeze        = contexto.get("squeeze_activo",  False)
    sq_rel         = contexto.get("squeeze_release", False)
    rsi            = contexto.get("rsi",             50)
    ema8           = contexto.get("ema8",            precio)
    ema21          = contexto.get("ema21",           precio)
    razones        = contexto.get("razones",         [])
    call_strike    = contexto.get("mejor_call_strike", precio * 1.05)
    put_strike     = contexto.get("mejor_put_strike",  precio * 0.95)
    pcr            = contexto.get("pcr",             1.0)
    iv_avg         = contexto.get("iv_avg",          30)

    # Lógica mínima de setup
    es_call  = "CALL" in senal
    es_put   = "PUT"  in senal

    if es_call:
        entry     = f"USD {precio * 1.002:.2f} (ruptura de apertura)"
        sl        = f"USD {min(ema8, ema21) * 0.995:.2f}"
        tp1       = f"USD {call_strike * 0.98:.2f} (muro CALL al 98%)"
        tp2       = f"USD {call_strike:.2f} (muro CALL completo)"
        direccion = "ALCISTA"
    elif es_put:
        entry     = f"USD {precio * 0.998:.2f} (ruptura a la baja)"
        sl        = f"USD {max(ema8, ema21) * 1.005:.2f}"
        tp1       = f"USD {put_strike * 1.02:.2f} (muro PUT al 102%)"
        tp2       = f"USD {put_strike:.2f} (muro PUT completo)"
        direccion = "BAJISTA"
    else:
        entry = sl = tp1 = tp2 = "—"
        direccion = "NEUTRO"

    sq_txt = "RELEASE (explosión activa)" if sq_rel else ("ACTIVO" if squeeze else "Inactivo")
    raz_txt = " | ".join(razones) if razones else "Sin confluencias destacadas"

    reporte = f"""
⚠️ *Reporte generado localmente (cupo Gemini Free Tier agotado)*

═══════════════════════════
{ticker} | USD {precio:.2f} | Señal: {senal}
═══════════════════════════

**SESGO**: {direccion}
**Confluencias detectadas**: {raz_txt}

**Datos técnicos clave**:
  • Squeeze BB/KC  : {sq_txt}
  • RSI(14)        : {rsi:.1f}  {'(Sobreventa)' if rsi < 32 else '(Sobrecompra)' if rsi > 68 else ''}
  • EMA 8 vs 21    : {'EMA 8 ({:.2f}) > EMA 21 ({:.2f}) → Alcista'.format(ema8, ema21) if ema8 > ema21 else 'EMA 8 ({:.2f}) < EMA 21 ({:.2f}) → Bajista'.format(ema8, ema21)}
  • PCR            : {pcr:.2f}  {'(Extremo bajista)' if pcr > 1.4 else '(Extremo alcista)' if pcr < 0.5 else ''}
  • IV promedio    : {iv_avg:.1f}%
  • Score CALL     : {score_call}/100
  • Score PUT      : {score_put}/100

**Setup sugerido** (basado en muros y EMAs, sin confirmación IA):
  • Entrada         : {entry}
  • Stop Loss       : {sl}
  • Take Profit 1   : {tp1}
  • Take Profit 2   : {tp2}

**Muros clave**:
  • Resistencia CALL : USD {call_strike:.2f}
  • Soporte PUT      : USD {put_strike:.2f}

*Para análisis IA completo: espera a que se renueve el cupo diario o activa un plan de pago en Google AI Studio.*
""".strip()

    return reporte


# ══════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL: llamar_gemini()
# ══════════════════════════════════════════════════════

def llamar_gemini(
    prompt: str,
    api_key: str,
    contexto_fallback: dict | None = None,
    usar_cache: bool = True,
) -> str:
    """Llama a Gemini con retry automático, caché thread-safe y fallback local.

    Pipeline de decisión:
      1. Caché → respuesta instantánea si el prompt ya fue procesado.
      2. Cupo agotado → fallback local sin llamada API.
      3. Llamada API con retry por modelo en orden de preferencia.
      4. Si todo falla → fallback local con datos disponibles.

    Todas las escrituras a session_state están protegidas por _gemini_lock
    para evitar race conditions desde ThreadPoolExecutor.

    Args:
        prompt: texto completo del prompt.
        api_key: Gemini API Key.
        contexto_fallback: dict con datos para el reporte local si se agota el cupo.
        usar_cache: si True, reutiliza respuestas previas idénticas.

    Returns:
        str con la respuesta (IA, caché o fallback local).
    """
    if not api_key:
        return "❌ API Key de Gemini no configurada en secrets.toml"

    _init_state()

    # ── 1. Intentar desde caché (lectura thread-safe)
    if usar_cache:
        cached = _get_cached(prompt)
        if cached:
            return f"📋 *(desde caché — sin request extra)*\n\n{cached}"

    # ── 2. Si el cupo ya está marcado como agotado en esta sesión
    with _gemini_lock:
        if st.session_state[_KEY_AGOTADO]:
            fb = _generar_fallback(contexto_fallback or {})
            return fb

    # ── 3. Intentar con cada modelo en orden de preferencia
    ultimo_error = None

    cliente = genai.Client(api_key=api_key)

    for modelo in MODELOS_PREFERENCIA:
        for intento in range(1, MAX_REINTENTOS + 1):
            try:
                respuesta = cliente.models.generate_content(
                    model=modelo, contents=prompt
                )
                if not respuesta.text:
                    # Respuesta vacía (p.ej. bloqueo de seguridad): probar siguiente modelo
                    ultimo_error = ValueError(f"{modelo} devolvió respuesta vacía")
                    break
                import re as _re
                # Solo el $ pegado a cifras (no rompe código ni texto legítimo)
                texto = _re.sub(r"\$(?=\s?\d)", "USD ", respuesta.text)

                # Éxito: guardar en caché y actualizar contador (thread-safe)
                if usar_cache:
                    _set_cached(prompt, texto)
                with _gemini_lock:
                    st.session_state[_KEY_CONTADOR] += 1
                    # Registrar qué modelo respondió (para la etiqueta de autoría)
                    st.session_state["_gemini_modelo_usado"] = modelo

                # Badge informativo en sidebar (no interrumpe el flujo)
                try:
                    with _gemini_lock:
                        reqs = st.session_state[_KEY_CONTADOR]
                    st.sidebar.caption(f"🔢 Gemini requests esta sesión: **{reqs}**")
                except Exception:
                    pass

                return texto

            except Exception as e:
                ultimo_error = e
                msg = str(e)

                # Error 429 — cupo agotado o rate limit
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    delay = _extraer_retry_delay(e)

                    if "GenerateRequestsPerDay" in msg or "quota" in msg.lower():
                        # Cupo DIARIO agotado — no tiene sentido reintentar
                        with _gemini_lock:
                            st.session_state[_KEY_AGOTADO] = True
                        st.warning(
                            "⚠️ Cupo diario de Gemini Free Tier agotado. "
                            "Generando reporte local con los datos calculados. "
                            "El cupo se renueva a medianoche (hora del Pacífico)."
                        )
                        return _generar_fallback(contexto_fallback or {})

                    # Rate limit temporal — esperar y reintentar
                    if intento < MAX_REINTENTOS:
                        wait = min(delay * (BACKOFF_BASE ** (intento - 1)), BACKOFF_MAX)
                        st.toast(f"⏳ Rate limit en {modelo} — reintentando en {wait:.0f}s...")
                        time.sleep(wait)
                        continue
                    else:
                        # Agoté reintentos en este modelo → siguiente modelo
                        break

                # Error de autenticación — no reintentar
                elif "401" in msg or "403" in msg or "API_KEY" in msg.upper():
                    return f"❌ Error de autenticación Gemini: verifica tu API Key en secrets.toml"

                # Otros errores de red → reintentar
                else:
                    if intento < MAX_REINTENTOS:
                        time.sleep(BACKOFF_BASE * intento)
                        continue
                    break

    # Todos los modelos y reintentos fallaron
    fb = _generar_fallback(contexto_fallback or {})
    st.error(f"❌ Gemini no respondió tras {MAX_REINTENTOS} intentos en {len(MODELOS_PREFERENCIA)} modelos. Último error: {ultimo_error}")
    return fb


# ══════════════════════════════════════════════════════
# WIDGET DE ESTADO EN SIDEBAR (llamar desde fingem.py)
# ══════════════════════════════════════════════════════

def mostrar_estado_gemini_sidebar():
    """Muestra un indicador visual del estado del cupo de Gemini en el sidebar.

    Llamar una vez en fingem.py después de inicializar el sidebar.
    Lecturas protegidas por lock para consistencia con escrituras concurrentes.
    """
    _init_state()
    with _gemini_lock:
        reqs    = st.session_state[_KEY_CONTADOR]
        agotado = st.session_state[_KEY_AGOTADO]
        cached  = len(st.session_state[_KEY_CACHE])

    if agotado:
        st.sidebar.error("🔴 Cupo Gemini agotado (se renueva a medianoche, hora del Pacífico)")
    elif reqs > 0:
        st.sidebar.success(f"🟢 Gemini: {reqs} requests usados esta sesión")
    else:
        st.sidebar.info("🟢 Gemini: cupo disponible")

    if cached > 0:
        st.sidebar.caption(f"💾 {cached} respuestas en caché (ahorran requests)")
