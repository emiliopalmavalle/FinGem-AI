"""
auth.py — Llave de entrada a la terminal
=========================================

Login simple con usuarios definidos en secrets.toml:

    [usuarios]
    emilio = "<sha256 de la contraseña>"

Las contraseñas nunca se guardan en claro — solo su hash SHA-256.
Para generar el hash de una contraseña nueva:

    python -c "import hashlib; print(hashlib.sha256('MiClave'.encode()).hexdigest())"

Uso en FINGEM.py (justo después de st.set_page_config):

    requerir_login()          # bloquea todo hasta autenticarse
    ...
    mostrar_usuario_sidebar() # dentro del sidebar: usuario + botón salir
"""

import hashlib
import time
import streamlit as st

_KEY_USUARIO = "usuario_autenticado"
_KEY_FALLOS  = "_login_intentos_fallidos"
_KEY_BLOQUEO = "_login_bloqueado_hasta"
MAX_INTENTOS = 5
BLOQUEO_SEG  = 60


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _credenciales_validas(usuario: str, password: str) -> bool:
    usuarios = st.secrets.get("usuarios", {})
    hash_guardado = usuarios.get(usuario)
    return bool(hash_guardado) and _hash(password) == hash_guardado


def requerir_login() -> None:
    """Muestra el formulario de acceso y detiene la app si no hay sesión.

    Si secrets no tiene la sección [usuarios], la app queda abierta
    (útil para desarrollo local sin configurar login).
    """
    if not st.secrets.get("usuarios"):
        return  # login no configurado → app abierta

    if st.session_state.get(_KEY_USUARIO):
        return  # ya autenticado

    st.title("🔐 Terminal de Inteligencia Financiera")
    st.caption("Acceso restringido — ingresa tus credenciales.")

    with st.form("form_login"):
        usuario = st.text_input("Usuario").strip().lower()
        password = st.text_input("Contraseña", type="password")
        enviar = st.form_submit_button("Entrar", type="primary", width="stretch")

    if enviar:
        # Rate limit (auditoría P9d): sleep progresivo por intento fallido
        # y bloqueo temporal tras MAX_INTENTOS — frena fuerza bruta básica
        bloqueado_hasta = st.session_state.get(_KEY_BLOQUEO, 0)
        if time.time() < bloqueado_hasta:
            restante = int(bloqueado_hasta - time.time())
            st.error(f"🔒 Demasiados intentos fallidos. Espera {restante}s e intenta de nuevo.")
        elif _credenciales_validas(usuario, password):
            st.session_state[_KEY_USUARIO] = usuario
            st.session_state.pop(_KEY_FALLOS, None)
            st.session_state.pop(_KEY_BLOQUEO, None)
            st.rerun()
        else:
            fallos = st.session_state.get(_KEY_FALLOS, 0) + 1
            st.session_state[_KEY_FALLOS] = fallos
            time.sleep(min(fallos * 2, 8))  # sleep progresivo
            if fallos >= MAX_INTENTOS:
                st.session_state[_KEY_BLOQUEO] = time.time() + BLOQUEO_SEG
                st.error(f"🔒 {MAX_INTENTOS} intentos fallidos — acceso bloqueado {BLOQUEO_SEG}s.")
            else:
                st.error(f"❌ Usuario o contraseña incorrectos ({fallos}/{MAX_INTENTOS}).")

    st.stop()  # nada debajo de esta línea se ejecuta sin login


def mostrar_usuario_sidebar() -> None:
    """Muestra el usuario activo y el botón de salir en el sidebar."""
    usuario = st.session_state.get(_KEY_USUARIO)
    if not usuario:
        return
    col1, col2 = st.sidebar.columns([3, 1])
    col1.caption(f"👤 **{usuario}**")
    if col2.button("Salir", key="btn_logout"):
        st.session_state.pop(_KEY_USUARIO, None)
        st.rerun()
