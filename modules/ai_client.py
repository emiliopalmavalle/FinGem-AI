"""
ai_client.py — Cliente IA multi-proveedor
==========================================

Cadena de proveedores con degradación automática:

  1. Claude (Anthropic)  — cerebro principal, mejor calidad de análisis.
                            Se desactiva solo si no hay API key, se acaban
                            los créditos o falla la autenticación.
  2. Gemini (Google)      — fallback gratuito (free tier). Reutiliza el
                            cliente existente con retry y selección de modelo.
  3. Reporte local        — _generar_fallback() de gemini_client: reporte
                            estructurado sin IA, la app nunca se queda muda.

Caché GLOBAL con st.cache_data (TTL 24h): a diferencia del caché por sesión
de gemini_client, este se comparte entre usuarios y sobrevive recargas de
página — el mismo prompt no vuelve a gastar tokens de ningún proveedor.
Solo se cachean respuestas reales de IA (nunca el fallback local).

Configuración: FINGEM.py llama configurar_ia() una vez por rerun con las
keys desde st.secrets. Ningún módulo lee st.secrets directamente.
"""

import re
import threading
from datetime import datetime, date
import streamlit as st

from modules.gemini_client import llamar_gemini, _generar_fallback

MODELO_CLAUDE = "claude-opus-5"  # mismo precio que 4.8 ($5/$25 MTok), conocimiento hasta may-2026
MAX_TOKENS_CLAUDE = 16000  # límite duro de thinking + texto; Opus 5 piensa más que 4.8 y solo se cobra lo usado

# ── Configuración inyectada desde el orquestador (FINGEM.py)
_config = {"claude_key": "", "gemini_key": ""}
_lock = threading.Lock()

# ── Claves de session_state
_KEY_CLAUDE_OFF = "_claude_no_disponible"   # créditos agotados / auth inválida
_KEY_CONTADORES = "_ia_contadores"          # {"claude": n, "gemini": n}
_KEY_THINKING   = "_ia_thinking_stats"      # {origen: {llamadas, thinking, salida}}


class _ProveedoresAgotadosError(Exception):
    """Ningún proveedor de IA pudo responder."""


class _ClaudeNoUtilizable(RuntimeError):
    """Respuesta de Claude inservible: refusal, vacía o truncada por max_tokens.

    Excepción propia (no RuntimeError genérico) para que _generar_cacheado
    caiga a Gemini SOLO en estos casos — un RuntimeError real de otra parte
    del stack sigue explotando, como debe hacerlo un bug genuino.
    """


def configurar_ia(claude_api_key: str = "", gemini_api_key: str = "") -> None:
    """Inyecta las API keys. Llamar una vez por rerun desde FINGEM.py."""
    with _lock:
        _config["claude_key"] = claude_api_key or ""
        _config["gemini_key"] = gemini_api_key or ""


def _init_contadores() -> None:
    if _KEY_CONTADORES not in st.session_state:
        st.session_state[_KEY_CONTADORES] = {"claude": 0, "gemini": 0}


def _incrementar(proveedor: str) -> None:
    _init_contadores()
    with _lock:
        st.session_state[_KEY_CONTADORES][proveedor] += 1


def _registrar_thinking(origen: str, thinking: int, salida: int) -> None:
    """Acumula el gasto de thinking por tipo de análisis (solo llamadas reales).

    Con la distribución real por origen se puede decidir con datos propios
    si bajar effort a medium en los radares vale la pena — en vez de
    adivinarlo. Vive en session_state: es observabilidad, no telemetría.
    """
    with _lock:
        stats = st.session_state.setdefault(_KEY_THINKING, {})
        s = stats.setdefault(origen, {"llamadas": 0, "thinking": 0, "salida": 0})
        s["llamadas"] += 1
        s["thinking"] += thinking
        s["salida"] += salida


# ══════════════════════════════════════════════════════
# PROVEEDOR 1: CLAUDE
# ══════════════════════════════════════════════════════

def _llamar_claude(prompt: str) -> tuple[str, int, int]:
    """Llamada a Claude. Lanza _ClaudeNoUtilizable si la respuesta no sirve.

    El SDK de Anthropic ya reintenta 429/5xx con backoff (max_retries=2),
    así que no duplicamos lógica de retry aquí.

    Returns:
        (texto, thinking_tokens, output_tokens) — el desglose de thinking
        viene de usage.output_tokens_details (Opus 5).
    """
    import anthropic

    cliente = anthropic.Anthropic(api_key=_config["claude_key"])
    respuesta = cliente.messages.create(
        model=MODELO_CLAUDE,
        max_tokens=MAX_TOKENS_CLAUDE,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )

    if respuesta.stop_reason == "refusal":
        raise _ClaudeNoUtilizable("Claude rechazó la solicitud (safety)")
    if respuesta.stop_reason == "max_tokens":
        # Truncado = sin el bloque ```json final → extraer_plan() devolvería
        # None y el plan se saltaría validar_plan() en silencio. Mejor fallar
        # aquí y que el pipeline caiga a Gemini.
        raise _ClaudeNoUtilizable("Respuesta truncada por max_tokens")

    texto = next((b.text for b in respuesta.content if b.type == "text"), "")
    if not texto:
        raise _ClaudeNoUtilizable("Claude devolvió respuesta vacía")

    detalles = getattr(respuesta.usage, "output_tokens_details", None)
    thinking = getattr(detalles, "thinking_tokens", 0) or 0
    # Solo el $ pegado a cifras (no rompe bloques de código ni texto legítimo)
    texto = re.sub(r"\$(?=\s?\d)", "USD ", texto)
    return texto, thinking, respuesta.usage.output_tokens


def _claude_disponible() -> bool:
    return bool(_config["claude_key"]) and not st.session_state.get(_KEY_CLAUDE_OFF)


def _marcar_claude_no_disponible(motivo: str) -> None:
    """Desactiva Claude por el resto de la sesión (sin créditos / key inválida).

    Solo marca session_state — sin st.warning aquí: esta función corre dentro
    de una función @st.cache_data y Streamlit re-emite los elementos estáticos
    en cada cache hit, así que el aviso reaparecería aunque Claude ya funcione.
    El sidebar (mostrar_estado_ia_sidebar) ya comunica este estado.
    """
    st.session_state[_KEY_CLAUDE_OFF] = motivo


# ══════════════════════════════════════════════════════
# PIPELINE CACHEADO (Claude → Gemini)
# ══════════════════════════════════════════════════════

# Marcadores con los que llamar_gemini señala que NO devolvió IA real
_MARCADORES_FALLO_GEMINI = ("❌", "⚠️")


@st.cache_data(ttl=60 * 60 * 24, max_entries=200, show_spinner=False)
def _generar_cacheado(prompt: str, dia_cache: str, origen: str) -> tuple[str, str, str]:
    """Intenta Claude y luego Gemini. Cachea la primera respuesta real.

    dia_cache (fecha de hoy) forma parte de la clave: un análisis nunca
    sobrevive al cambio de día aunque el TTL no haya expirado (auditoría
    P9b — evita servir el reporte de ayer con datos de hoy).

    Lanza _ProveedoresAgotadosError si ambos fallan — la excepción evita
    que el fallback local quede cacheado como si fuera respuesta de IA.

    Returns:
        (texto_respuesta, proveedor, timestamp_generacion)
    """
    ts_generacion = datetime.now().strftime("%d/%m/%Y %H:%M")
    # ── 1. Claude
    if _claude_disponible():
        try:
            import anthropic
            try:
                texto, thinking, salida = _llamar_claude(prompt)
                _incrementar("claude")
                _registrar_thinking(origen, thinking, salida)
                return texto, "claude", ts_generacion
            except anthropic.AuthenticationError:
                _marcar_claude_no_disponible("API key inválida")
            except anthropic.PermissionDeniedError:
                _marcar_claude_no_disponible("key sin permisos")
            except anthropic.BadRequestError as e:
                # Créditos agotados llega como 400 con mensaje de "credit balance"
                if "credit" in str(e).lower():
                    _marcar_claude_no_disponible("créditos agotados")
                # Otro 400: prompt inválido — no desactivar, solo caer a Gemini
            except anthropic.APIStatusError:
                pass  # 429/5xx tras los retries del SDK → caer a Gemini
            except anthropic.APIConnectionError:
                pass  # sin red hacia Anthropic → caer a Gemini
            except _ClaudeNoUtilizable:
                pass  # refusal / truncado por max_tokens / respuesta vacía → caer a Gemini
        except ImportError:
            _marcar_claude_no_disponible("paquete 'anthropic' no instalado")

    # ── 2. Gemini (cliente existente: retry + selección de modelo)
    if _config["gemini_key"]:
        resultado = llamar_gemini(
            prompt, _config["gemini_key"],
            contexto_fallback=None, usar_cache=False,
        )
        if resultado and not resultado.lstrip().startswith(_MARCADORES_FALLO_GEMINI):
            _incrementar("gemini")
            return resultado, "gemini", ts_generacion

    raise _ProveedoresAgotadosError()


# ══════════════════════════════════════════════════════
# API PÚBLICA
# ══════════════════════════════════════════════════════

def _etiqueta_modelo(proveedor: str) -> str:
    """Nombre legible del modelo que generó la respuesta."""
    if proveedor == "claude":
        return f"Claude Opus 5 (Anthropic · {MODELO_CLAUDE})"
    if proveedor == "gemini":
        modelo = st.session_state.get("_gemini_modelo_usado", "gemini-2.5-flash")
        return f"Gemini (Google · {modelo})"
    return "Motor local (sin IA)"


def proveedor_activo() -> str:
    """Nombre corto del proveedor que atenderá la próxima llamada (para spinners)."""
    if _claude_disponible():
        return "Claude Opus 5"
    if _config["gemini_key"]:
        return "Gemini"
    return "el motor local"


def llamar_ia(prompt: str, contexto_fallback: dict | None = None,
              origen: str = "general") -> str:
    """Punto de entrada único para toda la IA de la terminal.

    Toda respuesta de IA real se encabeza con la línea de autoría
    "🧠 Análisis generado por: <modelo>" — así cada reporte (pantalla
    y Telegram) declara qué IA y qué versión lo produjo.

    Args:
        prompt: texto completo del prompt.
        contexto_fallback: datos para el reporte local si toda la IA falla.
        origen: etiqueta del tipo de análisis (individual/radar/derivados)
            para las estadísticas de thinking del sidebar.

    Returns:
        Respuesta de Claude, Gemini o el reporte local estructurado.
    """
    if not _config["claude_key"] and not _config["gemini_key"]:
        return "❌ No hay API keys de IA configuradas (ANTHROPIC_API_KEY / GEMINI_API_KEY)."

    try:
        texto, proveedor, ts = _generar_cacheado(prompt, date.today().isoformat(), origen)
        return (f"🧠 *Análisis generado por: {_etiqueta_modelo(proveedor)} · {ts}*\n\n{texto}")
    except _ProveedoresAgotadosError:
        # El reporte local ya se anuncia a sí mismo en su encabezado
        return _generar_fallback(contexto_fallback or {})


def mostrar_estado_ia_sidebar() -> None:
    """Indicador del estado de los proveedores IA en el sidebar."""
    _init_contadores()
    contadores = st.session_state[_KEY_CONTADORES]
    claude_off = st.session_state.get(_KEY_CLAUDE_OFF)

    if _config["claude_key"] and not claude_off:
        st.sidebar.success("🧠 IA: Claude activo (Gemini de respaldo)")
    elif claude_off:
        st.sidebar.warning(f"🟡 Claude off ({claude_off}) → usando Gemini")
    elif _config["gemini_key"]:
        st.sidebar.info("🟢 IA: Gemini (agrega ANTHROPIC_API_KEY para usar Claude)")
    else:
        st.sidebar.error("🔴 Sin API keys de IA configuradas")

    usados = contadores["claude"] + contadores["gemini"]
    if usados:
        st.sidebar.caption(
            f"🔢 Requests esta sesión — Claude: {contadores['claude']} · "
            f"Gemini: {contadores['gemini']}"
        )

    # Distribución de thinking por tipo de análisis (solo llamadas reales a
    # Claude). Sirve para decidir con datos si bajar effort donde no aporta.
    stats = st.session_state.get(_KEY_THINKING, {})
    if stats:
        lineas = []
        for origen, s in sorted(stats.items()):
            prom_think = s["thinking"] // max(s["llamadas"], 1)
            prom_total = s["salida"] // max(s["llamadas"], 1)
            lineas.append(f"{origen}: {prom_think:,} thinking / {prom_total:,} total (×{s['llamadas']})")
        st.sidebar.caption("🧠 Tokens promedio por análisis — " + " · ".join(lineas))
