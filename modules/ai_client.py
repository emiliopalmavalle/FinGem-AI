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

import threading
import streamlit as st

from modules.gemini_client import llamar_gemini, _generar_fallback

MODELO_CLAUDE = "claude-opus-4-8"
MAX_TOKENS_CLAUDE = 8000  # los reportes son ≤450 palabras; deja aire para thinking

# ── Configuración inyectada desde el orquestador (FINGEM.py)
_config = {"claude_key": "", "gemini_key": ""}
_lock = threading.Lock()

# ── Claves de session_state
_KEY_CLAUDE_OFF = "_claude_no_disponible"   # créditos agotados / auth inválida
_KEY_CONTADORES = "_ia_contadores"          # {"claude": n, "gemini": n}


class _ProveedoresAgotadosError(Exception):
    """Ningún proveedor de IA pudo responder."""


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


# ══════════════════════════════════════════════════════
# PROVEEDOR 1: CLAUDE
# ══════════════════════════════════════════════════════

def _llamar_claude(prompt: str) -> str:
    """Llamada a Claude. Lanza excepción si falla (el caller decide el fallback).

    El SDK de Anthropic ya reintenta 429/5xx con backoff (max_retries=2),
    así que no duplicamos lógica de retry aquí.
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
        raise RuntimeError("Claude rechazó la solicitud (safety)")

    texto = next((b.text for b in respuesta.content if b.type == "text"), "")
    if not texto:
        raise RuntimeError("Claude devolvió respuesta vacía")
    return texto.replace("$", "USD ")


def _claude_disponible() -> bool:
    return bool(_config["claude_key"]) and not st.session_state.get(_KEY_CLAUDE_OFF)


def _marcar_claude_no_disponible(motivo: str) -> None:
    """Desactiva Claude por el resto de la sesión (sin créditos / key inválida)."""
    st.session_state[_KEY_CLAUDE_OFF] = motivo
    st.warning(f"⚠️ Claude no disponible ({motivo}). Usando Gemini como respaldo.")


# ══════════════════════════════════════════════════════
# PIPELINE CACHEADO (Claude → Gemini)
# ══════════════════════════════════════════════════════

# Marcadores con los que llamar_gemini señala que NO devolvió IA real
_MARCADORES_FALLO_GEMINI = ("❌", "⚠️")


@st.cache_data(ttl=60 * 60 * 24, max_entries=200, show_spinner=False)
def _generar_cacheado(prompt: str) -> tuple[str, str]:
    """Intenta Claude y luego Gemini. Cachea 24h la primera respuesta real.

    Lanza _ProveedoresAgotadosError si ambos fallan — la excepción evita
    que el fallback local quede cacheado como si fuera respuesta de IA.

    Returns:
        (texto_respuesta, proveedor)
    """
    # ── 1. Claude
    if _claude_disponible():
        try:
            import anthropic
            try:
                texto = _llamar_claude(prompt)
                _incrementar("claude")
                return texto, "claude"
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
            return resultado, "gemini"

    raise _ProveedoresAgotadosError()


# ══════════════════════════════════════════════════════
# API PÚBLICA
# ══════════════════════════════════════════════════════

def _etiqueta_modelo(proveedor: str) -> str:
    """Nombre legible del modelo que generó la respuesta."""
    if proveedor == "claude":
        return f"Claude Opus 4.8 (Anthropic · {MODELO_CLAUDE})"
    if proveedor == "gemini":
        modelo = st.session_state.get("_gemini_modelo_usado", "gemini-2.5-flash")
        return f"Gemini (Google · {modelo})"
    return "Motor local (sin IA)"


def proveedor_activo() -> str:
    """Nombre corto del proveedor que atenderá la próxima llamada (para spinners)."""
    if _claude_disponible():
        return "Claude Opus 4.8"
    if _config["gemini_key"]:
        return "Gemini"
    return "el motor local"


def llamar_ia(prompt: str, contexto_fallback: dict | None = None) -> str:
    """Punto de entrada único para toda la IA de la terminal.

    Toda respuesta de IA real se encabeza con la línea de autoría
    "🧠 Análisis generado por: <modelo>" — así cada reporte (pantalla
    y Telegram) declara qué IA y qué versión lo produjo.

    Args:
        prompt: texto completo del prompt.
        contexto_fallback: datos para el reporte local si toda la IA falla.

    Returns:
        Respuesta de Claude, Gemini o el reporte local estructurado.
    """
    if not _config["claude_key"] and not _config["gemini_key"]:
        return "❌ No hay API keys de IA configuradas (ANTHROPIC_API_KEY / GEMINI_API_KEY)."

    try:
        texto, proveedor = _generar_cacheado(prompt)
        return f"🧠 *Análisis generado por: {_etiqueta_modelo(proveedor)}*\n\n{texto}"
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
