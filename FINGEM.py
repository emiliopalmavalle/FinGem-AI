import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import datetime
from deep_translator import GoogleTranslator

# ==========================================
# 📦 IMPORTACIÓN DE MÓDULOS (ARQUITECTURA)
# ==========================================
from modules.radar_acciones import (
    escaneo_institucional_dual, buscar_joyas_ocultas,
    buscar_universo_bmv, buscar_universo_flujo,
)
from modules.radar_etfs import escanear_etfs, ETFS_NY, ETFS_BMV
from modules.radar_derivados import escanear_flujo_institucional
from modules.radar_opciones import ejecutar_radar_opciones, construir_grafico_radar, escanear_calls_baratos
from modules.ai_client import configurar_ia, llamar_ia, mostrar_estado_ia_sidebar, proveedor_activo
from modules.contexto_macro import obtener_contexto_macro_global, mostrar_metricas_macro
from modules.auth import requerir_login, mostrar_usuario_sidebar
from modules.validador_plan import extraer_plan, validar_plan, RB_MINIMO, REGLAS_RIESGO_PROMPT
from modules.opciones_cboe import mostrar_salud_datos, mostrar_salud_datos_lista
from modules.broker_alpaca import (
    configurar_broker, broker_activo, estado_cuenta, mercado_abierto,
    posiciones_abiertas, ordenes, calcular_tamano,
    enviar_orden_opcion, enviar_orden_accion, cancelar_orden, cerrar_posicion,
)
from modules.motor_grafico import construir_grafico_tecnico
from modules.procesador_datos import procesar_datos_tecnicos, descargar_historia

# ==========================================
# 🔐 CONFIGURACIÓN DE CREDENCIALES (SECRETS)
# Centralizado aquí: ningún módulo accede a
# st.secrets directamente (arquitectura limpia).
# ==========================================
TELEGRAM_TOKEN    = st.secrets.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID  = st.secrets.get("TELEGRAM_CHAT_ID", "")
TELEGRAM_GROUP_ID = st.secrets.get("TELEGRAM_GROUP_ID", "")
GEMINI_API_KEY    = st.secrets.get("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
ETHERSCAN_API_KEY = st.secrets.get("ETHERSCAN_API_KEY", "")
ALPACA_API_KEY    = st.secrets.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY", "")

# Bróker de PAPEL (simulado): cierra el ciclo descubrir → discriminar → medir
configurar_broker(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# Cliente IA multi-proveedor: Claude principal → Gemini respaldo → reporte local
configurar_ia(claude_api_key=ANTHROPIC_API_KEY, gemini_api_key=GEMINI_API_KEY)

# ==========================================
# 🤖 IA: ANÁLISIS TÉCNICO CON GEMINI
# Función que antes faltaba y causaba NameError.
# Centralizada aquí para reutilizar en cualquier
# módulo de la terminal sin acoplar st.secrets.
# ==========================================
def analizar_con_gemini(
    simbolo: str,
    precio_actual: float,
    recomendacion: str,
    textos_noticias: str,
    tipo_mercado: str,
    datos_extra: str = "",
    sentimiento: str = "",
    datos_onchain: str = "",
    ciclo_macro: str = "",
    temporalidad: str = "Diario",
    macro_global: str = "",
) -> str:
    """
    Cerebro IA completo — restaurado del cerebro viejo con toda su riqueza:
      · Prompt separado por mercado (Bolsa vs Cripto)
      · Lógica dinámica por temporalidad (Macro vs Micro)
      · Contexto on-chain, sentimiento, ciclo halving y noticias
      · Pasa por llamar_gemini() para retry, caché y fallback automáticos
    """
    if not ANTHROPIC_API_KEY and not GEMINI_API_KEY:
        return "❌ Error: No hay API keys de IA en tus secrets (ANTHROPIC_API_KEY / GEMINI_API_KEY)."

    # Horizonte operativo que implica cada temporalidad — la conclusión
    # final del reporte se enmarca en este plazo, no en genérico
    horizontes_map = {
        "1 Hora":  "las próximas horas de la sesión (intradía)",
        "4 Horas": "las próximas 1-2 jornadas",
        "Diario":  "los próximos 1-5 días",
        "Semanal": "las próximas semanas",
        "Mensual": "los próximos meses",
    }
    horizonte_op = horizontes_map.get(temporalidad, "el corto plazo")

    # Regla de stop según temporalidad: el múltiplo de ATR solo tiene
    # sentido en marcos cortos; en Semanal/Mensual el ATR de vela alta
    # es enorme y el anclaje válido es la estructura
    if temporalidad in ("1 Hora", "4 Horas", "Diario"):
        regla_stop = ("STOP LOSS: aproximadamente 1.5x el ATR(14) desde la entrada, "
                      "colocado del otro lado del nivel que protege (mínimo estructural o Put/Call Wall).")
    else:
        regla_stop = ("STOP LOSS: ancla el stop a la ESTRUCTURA (bajo el mínimo relevante o sobre el "
                      "máximo relevante); NO uses el múltiplo de ATR — en velas de esta temporalidad es demasiado amplio.")

    # Disciplina de riesgo compartida (validador_plan.REGLAS_RIESGO_PROMPT):
    # una sola fuente de verdad para los TRES prompts que piden plan
    # (Individual, Derivados y Radar) y para el umbral que verifica el código.
    bloque_disciplina = REGLAS_RIESGO_PROMPT

    bloque_conclusion = f"""
        SECCIÓN FINAL OBLIGATORIA — cierra SIEMPRE el reporte con una sección
        titulada exactamente "🎯 Conclusión ({temporalidad})" que responda en 3-4 líneas:
        - Dirección más probable del precio en {horizonte_op} y con qué convicción (alta/media/baja).
        - Nivel que CONFIRMA ese escenario y nivel que lo INVALIDA (USD exactos).
        - La acción concreta a tomar: entrar ya, esperar confirmación en X nivel, o quedarse fuera.
        Esta conclusión debe ser coherente con la temporalidad {temporalidad}: no des consejos
        intradía si el reporte es Semanal, ni visión de meses si es de 1 Hora."""

    bloque_json = """
        Después de la conclusión, añade al FINAL un bloque de código con SOLO este JSON
        (números sin comillas, sin texto extra dentro del bloque):
        ```json
        {"sesgo": "alcista|bajista|neutral", "direccion": "largo|corto|fuera", "entrada": 0.0, "stop": 0.0, "tp1": 0.0}
        ```
        OJO: "sesgo" es la dirección más probable del MERCADO; "direccion" es la OPERACIÓN del plan.
        Pueden diferir (ej.: sesgo bajista con un largo táctico de rebote). Si tu recomendación es
        quedarse fuera, usa "direccion": "fuera" y pon 0 en los niveles."""

    # ── BOLSA (NY / MX)
    if "NY" in tipo_mercado or "MX" in tipo_mercado or "Análisis Individual" in tipo_mercado:
        bloque_macro = f"- CONTEXTO MACRO GLOBAL (Top-Down): {macro_global}" if macro_global else ""
        prompt = f"""
        Eres un analista financiero institucional. El horizonte operativo de este reporte es {horizonte_op}
        (temporalidad de velas: {temporalidad}) — TODO el plan debe dimensionarse a ese plazo.
        Tu método es Top-Down: primero el macro, luego el sentimiento, luego la empresa, al final el precio.
        Fecha actual: {datetime.datetime.now():%Y-%m-%d}.
        Analiza la acción: {simbolo}.
        - Precio actual: USD {precio_actual:.2f}
        {bloque_macro}
        - Datos Técnicos, Estructura y Niveles: {datos_extra}
        - Consenso de analistas: {recomendacion}
        - Noticias recientes: {textos_noticias}

        Genera un resumen ejecutivo en 5 puntos:
        1. 🌍 Macro Top-Down: ¿el entorno de tasas, curva de rendimientos, inflación (petróleo/oro/dólar)
           y momento del ciclo económico FAVORECE o CASTIGA a este sector y a esta acción? Sé específico.
        2. 🧠 Sentimiento de Mercado: régimen del VIX (miedo/complacencia), TONO de las noticias
           (¿optimistas, negativas, neutras?) y — si hay niveles de opciones — el PCR del vencimiento y
           los strikes con volumen inusual. OJO: la dirección de ese volumen (compra o venta) es
           DESCONOCIDA con estos datos — trátalos como zonas de interés/imanes, no como apuestas confirmadas.
        3. 🏰 Cualitativo: en 2-3 líneas evalúa con tu conocimiento de la empresa su ventaja competitiva
           (moat: patentes, marca, costos de cambio), la calidad de su gestión, y el riesgo regulatorio
           o geopolítico más relevante de su sector hoy.
        4. ⚙️ Acción del Precio: tendencia (EMAs, ADX, Monitor), momentum (RSI, RVOL, divergencias) y zonas de liquidez (FVG).
        5. 💡 Veredicto Institucional ({horizonte_op}): sesgo direccional y plan operativo — coherente con los puntos 1 y 2:
           si el macro o el sentimiento van en contra del setup técnico, exige más confirmación o reduce el tamaño.

        REGLAS PARA EL PLAN DEL PUNTO 5 (obligatorias):
        - ENTRADA: anclada a un nivel real de la ESTRUCTURA (máx/mín 5-20 velas, high/low previo) o a un muro de opciones — nunca un número redondo inventado.
        - {regla_stop}
        - TAKE PROFIT: antes del siguiente nivel de estructura o muro opuesto; indica el ratio riesgo/beneficio resultante.
        - Si TODOS los niveles de estructura quedan POR ENCIMA del precio actual (colapso reciente), el único
          anclaje válido de soporte es el Put Wall; si tampoco existe, declara "sin soporte estructural cercano" y sé conservador.
        - Si hay earnings dentro del horizonte operativo, adviértelo como riesgo de gap.
        {bloque_disciplina}
        {bloque_conclusion}
        {bloque_json}

        REGLA: Usa 'USD' en lugar del símbolo dólar. Directo y sin frases genéricas.
        """

    # ── CRIPTO — lógica dinámica Macro vs Micro
    else:
        if temporalidad in ["Semanal", "Mensual"]:
            # Temporalidad alta: el ciclo del halving es la brújula principal
            bloque_ciclo = f"- RELOJ DEL CICLO MACRO: {ciclo_macro}" if ciclo_macro else ""
            regla_ciclo  = '1. Contexto Cíclico: El "Reloj del Ciclo Macro" es tu brújula principal. Cruza la fase del Halving con el gráfico técnico.'
            punto_1      = "1. 🕰️ Análisis del Ciclo Macro y Huella Institucional (Halving + técnico)."
        else:
            # Temporalidad baja: ignorar ciclos de 4 años, enfocarse en liquidez inmediata
            bloque_ciclo = ""
            regla_ciclo  = f'1. Contexto de Corto/Medio Plazo: Enfócate ESTRICTAMENTE en la acción del precio en {temporalidad}. IGNORA los ciclos macro de 4 años — enfócate en la liquidez inmediata y los FVG activos.'
            punto_1      = f"1. ⚙️ Acción del Precio y Huella Institucional en {temporalidad}."

        prompt = f"""
        Eres un Analista Quant Senior de Criptomonedas. Tu especialidad es cruzar datos On-Chain y Análisis Técnico (T. Latino y SMC).
        Fecha actual: {datetime.datetime.now():%Y-%m-%d}. Horizonte operativo del reporte: {horizonte_op}.
        Analiza el activo {simbolo} en temporalidad de {temporalidad}:
        - Precio actual: USD {precio_actual:.2f}
        {bloque_ciclo}
        {f"- CONTEXTO MACRO GLOBAL (tasas, VIX, dólar — el viento de cola o en contra del riesgo): {macro_global}" if macro_global else ""}
        - Sentimiento Retail (Fear & Greed): {sentimiento if sentimiento else "No disponible"}
        - Actividad On-Chain (Escáner/Perfilado de Ballenas): {datos_onchain if datos_onchain else "Sin datos on-chain en esta sesión"}
        - Datos Técnicos (EMA 10/55/200, ADX, Monitor MACD, FVG): {datos_extra}
        - Noticias recientes: {textos_noticias if textos_noticias else "Sin noticias disponibles"}

        Reglas de análisis:
        {regla_ciclo}
        2. Correlación Ballena/Precio: Evalúa si la actividad on-chain confirma o contradice la acción del precio. ¿La ballena es líder o seguidora?
        3. SMC y T. Latino: Confirma las zonas de liquidez (FVG activos) y la direccionalidad del Monitor (MACD diff).

        Genera un reporte agresivo y directo en 3 puntos:
        {punto_1}
        2. 🐋 Análisis de Liquidez y On-Chain (ballenas, sentimiento retail, flujo institucional).
        3. 💡 Veredicto Institucional y Operativa ({horizonte_op}) — sesgo, entrada USD exacta anclada
           a estructura, Stop Loss y Take Profit. {regla_stop}
        {bloque_disciplina}
        {bloque_conclusion}
        {bloque_json}

        REGLA ABSOLUTA: Usa 'USD' en lugar del símbolo dólar. Sin frases genéricas. Máximo 350 palabras.
        """

    ctx = {
        "ticker": simbolo,
        "precio": precio_actual,
        "razones": [datos_extra[:80]] if datos_extra else [],
    }
    return llamar_ia(prompt, contexto_fallback=ctx)


# ==========================================
# 🛠️ FUNCIONES AUXILIARES
# ==========================================

def enviar_alerta_telegram(mensaje: str) -> None:
    """
    Envía el análisis al chat privado y al grupo configurado.
    Fragmenta mensajes largos en bloques de 4000 chars (límite de Telegram).
    """
    if not TELEGRAM_TOKEN:
        st.error("❌ TELEGRAM_TOKEN no configurado en secrets.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    fragmentos = [mensaje[i:i + 4000] for i in range(0, len(mensaje), 4000)]
    destinos = [d for d in (TELEGRAM_CHAT_ID, TELEGRAM_GROUP_ID) if d]

    envios_exitosos = 0
    for chat_destino in destinos:
        for fragmento in fragmentos:
            payload = {
                "chat_id":    chat_destino,
                "text":       fragmento,
                "parse_mode": "Markdown",
            }
            try:
                respuesta = requests.post(url, json=payload, timeout=10)
                if respuesta.status_code != 200:
                    # Reintento sin Markdown si falla el formato — también
                    # cuenta como éxito si esta segunda vía entrega
                    reintento = requests.post(
                        url, json={"chat_id": chat_destino, "text": fragmento}, timeout=10
                    )
                    if reintento.status_code == 200:
                        envios_exitosos += 1
                else:
                    envios_exitosos += 1
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error de conexión al enviar a {chat_destino}: {e}")

    if envios_exitosos > 0:
        st.toast("✅ ¡Análisis enviado a Telegram (Privado y Grupo)!")


def obtener_sentimiento_macro() -> str:
    """Fear & Greed Index de Crypto (alternative.me)."""
    try:
        datos = requests.get("https://api.alternative.me/fng/", timeout=5).json()
        val   = datos['data'][0]['value']
        label = datos['data'][0]['value_classification']
        return f"{val}/100 ({label})"
    except Exception:
        return "Desconocido"


def calcular_fase_ciclo() -> tuple:
    """
    Calcula semanas desde el último Halving y determina la fase macro.
    Restaurado del cerebro viejo — alimenta el prompt de cripto semanal/mensual.
    """
    import datetime as _dt
    fecha_halving = _dt.datetime(2024, 4, 19)
    hoy           = _dt.datetime.now()
    semanas       = int((hoy - fecha_halving).days / 7)

    if semanas < 40:
        fase  = "Post-Halving (Acumulación temprana / Choque de oferta)"
        color = "🔵"
    elif 40 <= semanas < 77:
        fase  = "Markup Parabólico (Profit START) — Tendencia Alcista Fuerte"
        color = "🟢"
    elif 77 <= semanas < 135:
        fase  = "Distribución / Corrección (PROFIT END superado) — Mercado Bajista"
        color = "🔴"
    else:
        fase  = "DCA START (Suelo del Mercado / Acumulación Pre-Halving)"
        color = "🟡"

    return semanas, fase, color


def rastrear_ballena_btc(direccion_btc: str) -> str:
    """Rastrea actividad de una billetera BTC vía mempool.space."""
    url_stats = f"https://mempool.space/api/address/{direccion_btc}"
    url_txs   = f"https://mempool.space/api/address/{direccion_btc}/txs"
    try:
        datos = requests.get(url_stats, timeout=5).json()
        stats = datos['chain_stats']

        balance_btc = (stats['funded_txo_sum'] - stats['spent_txo_sum']) / 100_000_000
        tx_entradas = stats['funded_txo_count']
        tx_salidas  = stats['spent_txo_count']
        total_txs   = tx_entradas + tx_salidas

        resp_txs  = requests.get(url_txs, timeout=5)
        historial = resp_txs.json() if resp_txs.status_code == 200 else []

        fecha_ultima_tx = "Desconocida"
        if historial and 'block_time' in historial[0]:
            fecha_ultima_tx = datetime.datetime.fromtimestamp(
                historial[0]['block_time']
            ).strftime('%Y-%m-%d %H:%M')

        # Clasificación de perfil
        if total_txs < 5:
            perfil = "Billetera Nueva / Posible Exchange Interno"
        elif tx_salidas == 0 and tx_entradas > 5:
            perfil = "💎 Diamond Hands (Acumulador Institucional — No Vende)"
        elif tx_salidas > 0 and tx_entradas > 50:
            perfil = "📈 Trader Activo / Fondo de Inversión"
        else:
            perfil = "Indeterminado"

        return (
            f"💰 **Balance Actual:** {balance_btc:,.4f} BTC\n\n"
            f"🧠 **Perfil Smart Money:** {perfil}\n\n"
            f"🔄 **Transacciones:** {total_txs} (Entradas: {tx_entradas} | Salidas: {tx_salidas})\n\n"
            f"⏱️ **Último Movimiento:** {fecha_ultima_tx}"
        )
    except Exception:
        return "❌ No se pudo rastrear. Verifica que la dirección BTC sea válida."


def rastrear_ballena_eth(direccion_eth: str) -> str:
    """Rastrea actividad de una billetera ETH vía Etherscan."""
    if not ETHERSCAN_API_KEY:
        return "❌ Falta la API Key de Etherscan en tus secrets."

    try:
        # API V2 de Etherscan (la V1 fue dada de baja en 2025); chainid=1 = Mainnet
        url_balance = (
            f"https://api.etherscan.io/v2/api?chainid=1&module=account&action=balance"
            f"&address={direccion_eth}&tag=latest&apikey={ETHERSCAN_API_KEY}"
        )
        res_balance = requests.get(url_balance, timeout=5).json()
        balance_eth = int(res_balance['result']) / 1e18 if res_balance.get('status') == '1' else 0

        url_tx = (
            f"https://api.etherscan.io/v2/api?chainid=1&module=proxy&action=eth_getTransactionCount"
            f"&address={direccion_eth}&tag=latest&apikey={ETHERSCAN_API_KEY}"
        )
        res_tx   = requests.get(url_tx, timeout=5).json()
        tx_count = int(res_tx['result'], 16) if 'result' in res_tx else "Desconocido"

        return (
            f"🔷 **Balance Actual:** {balance_eth:,.4f} ETH\n"
            f"🔄 **Total de Transacciones:** {tx_count}\n"
            f"🔗 **Red:** Ethereum Mainnet"
        )
    except Exception:
        return "❌ No se pudo rastrear. Verifica la dirección ETH y tu API Key."


def escanear_anomalias_btc() -> dict:
    """Detecta ballenas en el último bloque BTC via mempool.space."""
    try:
        block_hash = requests.get(
            "https://mempool.space/api/blocks/tip/hash", timeout=5
        ).text
        txs = requests.get(
            f"https://mempool.space/api/block/{block_hash}/txs", timeout=10
        ).json()

        miner_tx = txs[0]
        recompensa_minero = sum(
            out.get('value', 0) for out in miner_tx.get('vout', [])
        ) / 100_000_000

        mayor_volumen = 0
        tx_ballena    = ""
        dir_ballena   = ""

        for tx in txs[1:]:
            volumen_tx     = 0
            max_vout_value = 0
            dir_temp       = ""

            for out in tx.get('vout', []):
                val = out.get('value', 0)
                volumen_tx += val
                if val > max_vout_value and 'scriptpubkey_address' in out:
                    max_vout_value = val
                    dir_temp = out['scriptpubkey_address']

            volumen_btc = volumen_tx / 100_000_000
            if volumen_btc > mayor_volumen:
                mayor_volumen = volumen_btc
                tx_ballena    = tx.get('txid', '')
                dir_ballena   = dir_temp

        return {
            "exito":                True,
            "bloque_hash":          block_hash[:15] + "...",
            "recompensa_minero":    recompensa_minero,
            "ballena_tx":           tx_ballena[:15] + "..." if tx_ballena else "N/A",
            "volumen_ballena":      mayor_volumen,
            "direccion_destinatario": dir_ballena,
        }
    except Exception:
        return {"exito": False}


# ==========================================
# 🖥️ INTERFAZ STREAMLIT
# ==========================================
st.set_page_config(page_title="Terminal Financiero AI", page_icon="📈", layout="wide")

# El dropdown de los selectbox se desplegaba hacia abajo y la ventana lo cortaba
# (la opción del fondo quedaba inalcanzable). Limitamos el alto de la lista para
# que haga scroll interno en vez de salirse de la pantalla.
st.markdown(
    """
    <style>
    ul[role="listbox"] { max-height: 45vh !important; overflow-y: auto !important; }
    section[data-testid="stSidebar"] { overflow-y: auto; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 🔐 Llave de entrada: nada se renderiza sin autenticación
requerir_login()

st.title("📊 Terminal de Inteligencia Financiera")

st.sidebar.header("Panel de Control")
mostrar_usuario_sidebar()
mostrar_estado_ia_sidebar()
tipo_mercado = st.sidebar.radio(
    "Selecciona el Módulo:",
    [
        "📈 Análisis Individual (NY / MX)",
        "🪙 Criptomonedas",
        "🌐 Escáner Global (Value/Momentum)",
        "🧺 Radar de ETFs (NY / BMV)",
        "🧱 Flujo de Opciones (Derivados)",
        "🎯 Radar de Opciones (Score Quant)",
        "💸 CALLs Baratos (Capital Pequeño)",
        "💼 Paper Trading (Alpaca)",
    ],
    index=1,  # Arranca en Criptomonedas → BTC-USD (símbolo default al iniciar)
)

# ==========================================
# 🎛️ LÓGICA DE LOS MENÚS LATERALES
# ==========================================
simbolo           = ""
temporalidad      = "Diario"
red_onchain       = "Ninguna"
direccion_ballena = ""
datos_escaner     = None

if tipo_mercado == "📈 Análisis Individual (NY / MX)":
    region = st.sidebar.selectbox(
        "Región:",
        ["🇺🇸 Wall Street (NY)", "🇲🇽 Bolsa Mexicana (BMV)", "✍️ Búsqueda Manual"]
    )
    if region == "🇺🇸 Wall Street (NY)":
        opciones_ny = {
            "Apple (AAPL)": "AAPL", "Nvidia (NVDA)": "NVDA",
            "Microsoft (MSFT)": "MSFT", "Tesla (TSLA)": "TSLA", "S&P 500 (SPY)": "SPY",
        }
        simbolo = opciones_ny.get(st.sidebar.selectbox("Empresa:", list(opciones_ny.keys())), "")
    elif region == "🇲🇽 Bolsa Mexicana (BMV)":
        opciones_mx = {
            "Grupo México": "GMEXICOB.MX",
            "Walmart":      "WALMEX.MX",
            "América Móvil": "AMXB.MX",
        }
        simbolo = opciones_mx.get(st.sidebar.selectbox("Empresa:", list(opciones_mx.keys())), "")
    else:
        simbolo = st.sidebar.text_input(
            "Símbolo:", "AMD",
            help="Formato exacto de Yahoo Finance: AAPL (NY), BIMBOA.MX (BMV con sufijo .MX), BTC-USD (cripto).",
        ).upper()
    # select_slider en vez de selectbox: el dropdown se cortaba al
    # quedar al fondo del sidebar sin posibilidad de scroll
    temporalidad = st.sidebar.select_slider(
        "Temporalidad (Velas):",
        options=["1 Hora", "4 Horas", "Diario", "Semanal", "Mensual"],
        value="Diario",
    )

elif tipo_mercado == "🪙 Criptomonedas":
    opciones_cripto = {
        "Bitcoin (BTC)":  "BTC-USD",
        "Ethereum (ETH)": "ETH-USD",
        "Solana (SOL)":   "SOL-USD",
    }
    seleccion = st.sidebar.selectbox(
        "Criptomoneda:", list(opciones_cripto.keys()) + ["✍️ Búsqueda Manual"]
    )
    simbolo = (
        st.sidebar.text_input("Símbolo:", "DOGE-USD").upper()
        if seleccion == "✍️ Búsqueda Manual"
        else opciones_cripto.get(seleccion, "")
    )
    temporalidad = st.sidebar.select_slider(
        "Temporalidad (Velas):",
        options=["1 Hora", "4 Horas", "Diario", "Semanal", "Mensual"],
        value="Diario",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🕵️‍♂️ Radar On-Chain")
    red_onchain = st.sidebar.selectbox(
        "Rastrear Billetera Específica:", ["Ninguna", "Bitcoin (BTC)", "Ethereum (ETH)"]
    )
    if red_onchain != "Ninguna":
        direccion_ballena = st.sidebar.text_input("Dirección de la Billetera:")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Escáner Caza-Ballenas")
    # Guardado en session_state: sin esto el resultado se pierde en el
    # siguiente rerun de Streamlit y nunca llega al prompt de la IA.
    if st.sidebar.button("🔎 Escanear Último Bloque (BTC)"):
        with st.sidebar.status("Conectando a mempool.space..."):
            st.session_state["datos_escaner_btc"] = escanear_anomalias_btc()
    datos_escaner = st.session_state.get("datos_escaner_btc")
    if datos_escaner and datos_escaner.get("exito"):
        if st.sidebar.button("🗑️ Limpiar resultado del escáner"):
            st.session_state.pop("datos_escaner_btc", None)
            datos_escaner = None

elif tipo_mercado == "🧱 Flujo de Opciones (Derivados)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuración Quant")
    opciones_derivados = {
        "S&P 500 (SPY)":     "SPY",
        "Nasdaq 100 (QQQ)":  "QQQ",
        "Bitcoin ETF (IBIT)": "IBIT",
        "Tesla (TSLA)":      "TSLA",
        "Apple (AAPL)":      "AAPL",
    }
    # Preselección corta (5 ítems) para que el dropdown no se corte al fondo
    # del sidebar, + campo manual SIEMPRE visible: así no hace falta abrir ni
    # scrollear el desplegable para escribir un ticker cualquiera.
    seleccion = st.sidebar.selectbox(
        "Activo a escanear:", list(opciones_derivados.keys())
    )
    simbolo_manual = st.sidebar.text_input(
        "…o escribe un símbolo manual:",
        "",
        help="Déjalo vacío para usar la preselección de arriba. "
             "Formato exacto de Yahoo Finance: NVDA, SPY, IBIT, TSLA.",
    ).upper().strip()
    simbolo = simbolo_manual if simbolo_manual else opciones_derivados.get(seleccion, "")

elif tipo_mercado == "🌐 Escáner Global (Value/Momentum)":
    st.header("📡 Radar Institucional Separado")
    st.write("Escanea el mercado y clasifica las oportunidades separando Acciones de ETFs para evitar ruido.")

    universos = {
        "🇺🇸 Top 50 Wall Street (Megacaps)": (
            "AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, BRK-B, LLY, AVGO, V, JPM, WMT, UNH, "
            "MA, PG, JNJ, XOM, HD, ORCL, COST, BAC, ABBV, CRM, CVX, NFLX, AMD, PEP, TMO, KO, "
            "WFC, DIS, CSCO, ACN, MCD, LIN, INTU, IBM, QCOM, CAT, GE, TXN, AMAT, ISRG, PM, "
            "NOW, UNP, COP, GS, AXP"
        ),
        "🇲🇽 BMV (IPC Ampliado — 36 emisoras)": (
            "AC.MX, ALSEA.MX, AMXB.MX, ASURB.MX, BBAJIOO.MX, BIMBOA.MX, BOLSAA.MX, "
            "CEMEXCPO.MX, CHDRAUIB.MX, CUERVO.MX, FEMSAUBD.MX, GAPB.MX, GCARSOA1.MX, "
            "GCC.MX, GENTERA.MX, GFINBURO.MX, GFNORTEO.MX, GMEXICOB.MX, GRUMAB.MX, "
            "KIMBERA.MX, KOFUBL.MX, LABB.MX, LIVEPOLC-1.MX, MEGACPO.MX, OMAB.MX, "
            "ORBIA.MX, PE&OLES.MX, PINFRA.MX, Q.MX, RA.MX, TLEVISACPO.MX, VESTA.MX, "
            "WALMEX.MX, VOLARA.MX, SORIANAB.MX, LACOMERUBC.MX"
        ),
        "🇲🇽🔍 Joyas BMV (Screener Dinámico)": "AUTO_MX",
        "🔍 Búsqueda Profunda (Joyas Ocultas)": "AUTO",
        "✍️ Lista Personalizada": "",
    }

    seleccion_universo = st.radio("Selecciona el Universo a Escanear:", list(universos.keys()))

    if seleccion_universo == "🔍 Búsqueda Profunda (Joyas Ocultas)":
        st.info("🤖 **Filtros Quant Activos:** P/E < 20, crecimiento EPS positivo, Volumen > 500K, Precio > USD 5, solo NYSE/NASDAQ.")
        tickers_input = ""
    elif seleccion_universo == "🇲🇽🔍 Joyas BMV (Screener Dinámico)":
        st.info(
            "🤖 **Descubrimiento dinámico BMV:** las 30 emisoras más líquidas de la Bolsa Mexicana "
            "(volumen > 100K, precio > MXN 5) vía screener de Yahoo — incluye FIBRAs y emisoras "
            "fuera del IPC. El motor dual las clasifica en Value/Momentum."
        )
        tickers_input = ""
    else:
        tickers_default = universos[seleccion_universo]
        tickers_input = st.text_area(
            "Tickers a escanear (separados por coma):", value=tickers_default, height=100
        )

    if st.button("🚀 Iniciar Barrido Inteligente"):
        lista_tickers = []

        if seleccion_universo == "🔍 Búsqueda Profunda (Joyas Ocultas)":
            with st.spinner("🕵️‍♂️ Buscando joyas con el screener de Yahoo..."):
                lista_tickers = buscar_joyas_ocultas(max_resultados=25)
                if lista_tickers:
                    st.success(f"✅ {len(lista_tickers)} activos encontrados. Iniciando motor Quant...")
                else:
                    st.warning("⚠️ El screener no arrojó resultados. Intenta de nuevo en unos minutos.")
        elif seleccion_universo == "🇲🇽🔍 Joyas BMV (Screener Dinámico)":
            with st.spinner("🕵️‍♂️ Descubriendo emisoras líquidas de la BMV..."):
                lista_tickers = buscar_universo_bmv(max_resultados=30)
                if lista_tickers:
                    st.success(f"✅ {len(lista_tickers)} emisoras encontradas. Iniciando motor Quant...")
                else:
                    st.warning("⚠️ El screener no arrojó resultados. Intenta de nuevo en unos minutos.")
        else:
            lista_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        if lista_tickers:
            barra_progreso = st.progress(0)
            st.info(f"⏳ Analizando {len(lista_tickers)} activos en paralelo...")

            df_val, df_mom_acc, df_mom_etf = escaneo_institucional_dual(lista_tickers)
            barra_progreso.progress(100)

            st.markdown("---")
            st.subheader("💎 Oportunidades de Valor — Descuento vs Máximo 52 Semanas")
            if not df_val.empty:
                st.dataframe(df_val, width="stretch", hide_index=True)
                st.caption(
                    "**Score** = profundidad del descuento + P/E barato + bono de estabilización. "
                    "**vs SMA200**: ✅ sobre la media de 200 días (estabilizada) · ⚠️ debajo (aún en caída)."
                )
            else:
                # El motor de Valor es solo-acciones (los ETFs no tienen P/E ni
                # EPS). Si el universo trae ETFs (lista personalizada), el valor
                # sale vacío por diseño: se analizan en el módulo Radar de ETFs.
                if not df_mom_etf.empty:
                    st.info(
                        "💡 Hay **ETFs** en este universo — el análisis de Valor (P/E, EPS) solo "
                        "aplica a acciones. Para ver ETFs y su condición usa el módulo "
                        "**🧺 Radar de ETFs (NY / BMV)** en el panel izquierdo."
                    )
                else:
                    st.warning("No se encontraron acciones con descuento significativo y fundamentales sanos.")

            st.subheader("🏢 Momentum Acciones")
            if not df_mom_acc.empty:
                st.dataframe(df_mom_acc, width="stretch", hide_index=True)
            else:
                st.warning("No hay acciones rompiendo al alza con volumen.")


# ==========================================
# 🧺 MÓDULO: RADAR DE ETFs (NY / BMV)
# ==========================================
# Sección propia para ETFs (separada del Escáner Global de acciones).
# Muestra los mejores ETFs de cada bolsa y su condición técnica:
# tendencia, momentum semanal, RVOL y distancia al máximo de 52s.
# ==========================================
elif tipo_mercado == "🧺 Radar de ETFs (NY / BMV)":
    st.header("🧺 Radar de ETFs")
    st.write(
        "Los ETFs más líquidos de cada bolsa con su **condición técnica** actual, "
        "ordenados de más fuerte a más débil. "
        "Condición: 🟢 Fuerte (sobre SMA200 y subiendo) · 🟡 Neutral · 🔴 Débil."
    )

    bolsa_etf = st.radio(
        "Selecciona la bolsa:",
        ["🇺🇸 Bolsa de NY (USD)", "🇲🇽 Bolsa Mexicana (BMV — MXN)"],
        horizontal=True,
    )
    catalogo_etf = ETFS_NY if bolsa_etf.startswith("🇺🇸") else ETFS_BMV

    if bolsa_etf.startswith("🇲🇽"):
        st.caption(
            "BMV: NAFTRAC/MEXTRAC son los locales del IPC; el resto son listados "
            "SIC de ETFs internacionales cotizados en pesos. Precios en MXN."
        )

    st.markdown("---")

    if st.button("🔭 Escanear ETFs", type="primary", width="stretch"):
        with st.spinner(f"Analizando {len(catalogo_etf)} ETFs en paralelo..."):
            df_etf = escanear_etfs(catalogo_etf)

        if df_etf.empty:
            st.warning(
                "⚠️ No se obtuvieron datos. Puede ser un rate-limit temporal de "
                "Yahoo — reintenta en unos segundos."
            )
        else:
            fuertes = int((df_etf["Condición"] == "🟢 Fuerte").sum())
            debiles = int((df_etf["Condición"] == "🔴 Débil").sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("ETFs escaneados", len(df_etf))
            c2.metric("🟢 En fuerza", fuertes)
            c3.metric("🔴 Débiles", debiles)

            st.dataframe(df_etf, width="stretch", hide_index=True)
            st.caption(
                "**Semana**: cambio en 5 sesiones · **RVOL**: volumen vs promedio 20d · "
                "**Tendencia**: posición vs SMA200 (✅ encima / ⚠️ debajo) · "
                "**vs Máx 52s**: distancia al máximo de 52 semanas."
            )


# ==========================================
# ⚙️ MÓDULO PRINCIPAL: ANÁLISIS INDIVIDUAL
# ==========================================
if tipo_mercado in ["📈 Análisis Individual (NY / MX)", "🪙 Criptomonedas"] and simbolo:
    st.write("---")
    st.write(f"## Analizando: {simbolo} — Gráfico de {temporalidad}")

    # Mapeo de temporalidad a parámetros yfinance
    config_temporal = {
        "1 Hora":   ("2mo",  "1h"),
        "4 Horas":  ("3mo",  "1h"),
        "Semanal":  ("10y",  "1wk"),
        "Mensual":  ("max",  "1mo"),
        "Diario":   ("5y",   "1d"),
    }
    periodo_yf, intervalo_yf = config_temporal.get(temporalidad, ("5y", "1d"))

    # ✅ Descarga cacheada: re-renders por toggle son instantáneos
    hist = descargar_historia(simbolo, periodo_yf, intervalo_yf)

    if not hist.empty:
        if temporalidad == "4 Horas":
            hist = hist.resample('4h').agg({
                'Open': 'first', 'High': 'max',
                'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()

        precio_actual = hist['Close'].iloc[-1]
        datos = procesar_datos_tecnicos(hist)

        # --- Métricas de UI ---
        st.subheader(f"⚙️ Setup Institucional ({temporalidad})")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precio Actual",   f"${precio_actual:,.2f}")
        col2.metric("T. Latino: Monitor", datos['direccion_monitor'])
        col3.metric("SMC: EMA 200",    f"${datos['ema_200']:,.2f}" if datos['ema_200'] > 0 else "N/A")
        if "Cripto" in tipo_mercado:
            col4.metric("🌐 Sentimiento Macro", obtener_sentimiento_macro())
        else:
            col4.metric("📊 Mercado", "Renta Variable")

        # --- ⏰ Reloj del Ciclo Halving (restaurado del cerebro viejo) ---
        if "Cripto" in tipo_mercado:
            semanas_h, fase_h, color_h = calcular_fase_ciclo()
            st.markdown(f"""
            <div style="padding:15px;border-radius:5px;background-color:rgba(255,255,255,0.05);
                        border-left:5px solid #F23645;margin-bottom:10px;">
                <h4 style="margin:0;padding:0;">{color_h} Reloj del Ciclo Halving (Semana +{semanas_h}w)</h4>
                <p style="margin:5px 0 0 0;font-size:14px;opacity:0.8;"><b>Fase Actual:</b> {fase_h}</p>
            </div>
            """, unsafe_allow_html=True)

        st.write(
            f"**T. Latino:** ADX {datos['adx_actual']:.2f} ({datos['pendiente_adx']}) | "
            f"EMA 55: ${datos['ema_55']:,.2f}"
        )
        st.write(
            f"**SMC Liquidez (FVG):** Imbalance Alc: `{datos['fvg_bullish']}` | "
            f"Imbalance Baj: `{datos['fvg_bearish']}`"
        )

        # --- Panel de Toggles ---
        st.write("🎛️ **Capas de Trading (On/Off)**")
        # El toggle de halving solo aparece donde tiene sentido (cripto semanal)
        es_cripto_semanal = temporalidad == "Semanal" and "Cripto" in tipo_mercado
        t_cols = st.columns(5 if es_cripto_semanal else 4)
        toggles = {
            "EMAs":         t_cols[0].toggle("🧠 EMAs", value=True),
            "SR":           t_cols[1].toggle("📏 S/R", value=False),
            "SMI":          t_cols[2].toggle("🌊 SMI", value=False),
            "Fondo_Oscuro": t_cols[3].toggle("🌙 Fondo oscuro", value=True),
        }
        if es_cripto_semanal:
            toggles["Halving"] = t_cols[4].toggle("🔮 Ciclo Halving", value=True)

        # --- Gráfico Plotly ---
        fig = construir_grafico_tecnico(
            datos['hist'], datos['ha_df'], datos['ema_200'],
            temporalidad, tipo_mercado, toggles
        )
        st.plotly_chart(fig, width="stretch", config={
            'scrollZoom': True, 'displayModeBar': False
        })

        # Leyenda Halving (solo cripto semanal y con el toggle encendido)
        if es_cripto_semanal and toggles.get("Halving", True):
            st.markdown("""
            <div style="padding:12px;border-radius:8px;background-color:rgba(255,255,255,0.05);
                        border:1px solid rgba(255,255,255,0.1);margin-top:5px;margin-bottom:15px;">
                <div style="font-size:14px;display:flex;justify-content:space-around;flex-wrap:wrap;gap:10px;">
                    <div><span style="color:#FF9800;font-weight:bold;">- -</span> <b>Halving</b> (0w)</div>
                    <div><span style="color:#089981;font-weight:bold;">——</span> <b>Profit Start</b> (+40w)</div>
                    <div><span style="color:#F23645;font-weight:bold;">——</span> <b>Profit End</b> (+77w)</div>
                    <div><span style="color:#FFD700;font-weight:bold;">——</span> <b>DCA Start</b> (+135w)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- Resultados On-Chain ---
        if "Cripto" in tipo_mercado:
            if datos_escaner and datos_escaner.get("exito"):
                st.success("✅ **Escáner On-Chain Completado (Último Bloque BTC)**")
                c1, c2 = st.columns(2)
                with c1:
                    st.info(
                        f"🐋 **Mayor Transacción (Ballena):**\n\n"
                        f"Volumen: **{datos_escaner['volumen_ballena']:,.2f} BTC**\n\n"
                        f"TX ID: `{datos_escaner['ballena_tx']}`"
                    )
                    if datos_escaner.get('direccion_destinatario'):
                        st.write("🎯 **Dirección de la Ballena (Clic ícono para copiar):**")
                        st.code(datos_escaner['direccion_destinatario'], language=None)
                with c2:
                    st.warning(
                        f"⛏️ **Actividad de Mineros:**\n\n"
                        f"Recompensa/Movimiento: **{datos_escaner['recompensa_minero']:,.2f} BTC**\n\n"
                        f"Bloque: `{datos_escaner['bloque_hash']}`"
                    )

            if red_onchain != "Ninguna" and direccion_ballena:
                if red_onchain == "Bitcoin (BTC)":
                    resultado_ballena = rastrear_ballena_btc(direccion_ballena)
                elif red_onchain == "Ethereum (ETH)":
                    resultado_ballena = rastrear_ballena_eth(direccion_ballena)
                else:
                    resultado_ballena = ""

                if resultado_ballena:
                    st.info(f"**Billetera Monitoreada:** `{direccion_ballena}`\n\n{resultado_ballena}")

        # --- Análisis IA ---
        st.write("---")
        st.subheader("🤖 Análisis Asistido por Inteligencia Artificial (Técnico / On-Chain)")

        enviar_telegram = st.checkbox("📤 Enviar copia de este reporte a Telegram", value=False)

        if st.button(f"Generar Análisis ({temporalidad}) 🤖", type="primary"):
            with st.spinner(f"Ensamblando contexto y consultando {proveedor_activo()}..."):

                # ── 0. Contexto macro global Top-Down (VIX, tasas, curva, inflación)
                ctx_macro_global = obtener_contexto_macro_global()
                macro_global_str = ctx_macro_global.get("texto", "")

                # ── 1. Datos técnicos completos: tendencia (EMAs/ADX) +
                #       contexto operable (ATR, RSI, RVOL, estructura)
                divergencia_txt = (
                    "Divergencia ALCISTA reciente (posible giro arriba)" if datos.get('div_bull_reciente')
                    else "Divergencia BAJISTA reciente (posible giro abajo)" if datos.get('div_bear_reciente')
                    else "Sin divergencias"
                )
                datos_extra_str = (
                    f"Temporalidad: {temporalidad}. "
                    f"EMA 10: USD {datos['ema_10']:.2f}, "
                    f"EMA 55: USD {datos['ema_55']:.2f}, "
                    f"EMA 200: USD {datos['ema_200']:.2f}. "
                    f"ADX: {datos['adx_actual']:.2f} ({datos['pendiente_adx']}). "
                    f"Monitor: {datos['direccion_monitor']}. "
                    f"FVG Alcista: {datos['fvg_bullish']}. "
                    f"FVG Bajista: {datos['fvg_bearish']}. "
                    f"ATR(14): USD {datos['atr_14']:.2f} (volatilidad por vela). "
                    f"RSI(14): {datos['rsi_14']:.1f}. "
                    f"RVOL: {datos['rvol']:.1f}x su media de 20 velas. "
                    f"{divergencia_txt}. "
                    f"ESTRUCTURA — Máx/Mín 5 velas: USD {datos['max_5']:.2f} / USD {datos['min_5']:.2f}; "
                    f"Máx/Mín 20 velas: USD {datos['max_20']:.2f} / USD {datos['min_20']:.2f}; "
                    f"High/Low vela previa: USD {datos['high_prev']:.2f} / USD {datos['low_prev']:.2f}."
                )

                # ── 1b. Niveles de opciones (solo bolsa con opciones listadas):
                #        dónde están posicionados los market makers
                # Marca si el prompt llegó a incluir niveles de opciones: solo
                # entonces tiene sentido mostrar la salud de esa fuente (en BMV
                # y cripto no hay opciones listadas y no es una anomalía)
                opciones_usadas = False
                if "Individual" in tipo_mercado:
                    try:
                        # Fuente CBOE: yfinance devuelve openInterest=0, bid/ask=0
                        # e IV basura, lo que anulaba los muros y disparaba falsos
                        # "flujo fresco" (vol > OI siempre cierto con OI=0)
                        from modules.opciones_cboe import cadena_cboe, vencimientos_disponibles
                        fechas_op = vencimientos_disponibles(simbolo)
                        if fechas_op:
                            opciones_usadas = True
                            from modules.radar_derivados import (
                                calcular_niveles_dia, encontrar_fecha_daytrading, encontrar_fecha_cercana,
                            )
                            # Vencimiento acorde a la temporalidad del análisis:
                            # muros que expiran en días no sirven para un plan semanal
                            if temporalidad in ("1 Hora", "4 Horas", "Diario"):
                                fecha_op = encontrar_fecha_daytrading(fechas_op)
                            elif temporalidad == "Semanal":
                                fecha_op = encontrar_fecha_cercana(fechas_op, 45)
                            else:  # Mensual
                                fecha_op = encontrar_fecha_cercana(fechas_op, 90)
                            niv_op = calcular_niveles_dia(
                                cadena_cboe(simbolo, fecha_op, spot=precio_actual), precio_actual
                            )
                            _fo = lambda v: f"USD {v:.2f}" if v is not None else "N/A"
                            datos_extra_str += (
                                f" NIVELES DE OPCIONES (vencimiento {fecha_op}): "
                                f"Soporte Put Wall {_fo(niv_op.get('put_wall'))}, "
                                f"Resistencia Call Wall {_fo(niv_op.get('call_wall'))}, "
                                f"Max Pain {_fo(niv_op.get('max_pain'))}, "
                                f"PCR {round(niv_op['pcr'], 2) if niv_op.get('pcr') else 'N/A'}. "
                                f"Flujo fresco hoy: {'; '.join(niv_op.get('flujo_fresco', [])) or 'ninguno'}."
                            )
                    except Exception:
                        pass  # sin opciones o Yahoo falló: el análisis sigue sin este bloque

                # ── 2. Sentimiento macro (Fear & Greed)
                sentimiento_str = ""
                if "Cripto" in tipo_mercado:
                    sentimiento_str = obtener_sentimiento_macro()

                # ── 3. Reloj del ciclo Halving
                ciclo_str = ""
                if "Cripto" in tipo_mercado:
                    semanas_h, fase_h, color_h = calcular_fase_ciclo()
                    ciclo_str = f"Semana {semanas_h} post-halving. Fase: {fase_h}."

                # ── 4. Datos on-chain acumulados en esta sesión
                onchain_str = ""
                if "Cripto" in tipo_mercado:
                    if datos_escaner and datos_escaner.get("exito"):
                        onchain_str += (
                            f"Escáner último bloque BTC: ballena movió "
                            f"{datos_escaner['volumen_ballena']:.2f} BTC "
                            f"(TX: {datos_escaner['ballena_tx']}). "
                            f"Minero: {datos_escaner['recompensa_minero']:.2f} BTC. "
                        )
                    if red_onchain != "Ninguna" and direccion_ballena:
                        try:
                            if red_onchain == "Bitcoin (BTC)":
                                res_b = rastrear_ballena_btc(direccion_ballena)
                            else:
                                res_b = rastrear_ballena_eth(direccion_ballena)
                            onchain_str += f"Billetera vigilada {direccion_ballena}: {res_b}"
                        except Exception:
                            pass

                # ── 5. Noticias traducidas al español
                noticias_str = ""
                try:
                    ticker_obj   = yf.Ticker(simbolo)
                    noticias_raw = ticker_obj.news[:3] if hasattr(ticker_obj, "news") else []
                    if noticias_raw:
                        traductor = GoogleTranslator(source="auto", target="es")
                        for n in noticias_raw:
                            # yfinance >= 0.2.5x anida el titular en content.title;
                            # se conserva n["title"] como fallback para versiones viejas
                            contenido = n.get("content") or {}
                            titular = contenido.get("title") or n.get("title", "")
                            if titular:
                                try:    titular_es = traductor.translate(titular)
                                except Exception: titular_es = titular
                                noticias_str += f"- {titular_es}\n"
                except Exception:
                    noticias_str = "Sin noticias disponibles."

                # ── 6. Consenso + fundamentales + earnings próximos (solo bolsa)
                recomendacion_str = "N/A"
                fund_ui = {}       # métricas para la fila visible (tabla del auditor)
                ficha_empresa = {} # identidad: nombre, sector, actividad, país
                if "NY" in tipo_mercado or "MX" in tipo_mercado or "Individual" in tipo_mercado:
                    try:
                        # Una sola llamada .info para consenso Y fundamentales
                        tk_fund = yf.Ticker(simbolo)
                        info_ticker = tk_fund.info
                        recomendacion_str = info_ticker.get("recommendationKey", "N/A") or "N/A"

                        # ── Ficha de identidad: nombre, sector, actividad y país.
                        #    Sector/país por mapa ES (offline); la actividad
                        #    (resumen de negocio) se traduce y se recorta breve.
                        from modules.radar_acciones import SECTORES_ES, PAISES_ES
                        nombre_emp = info_ticker.get("longName") or info_ticker.get("shortName") or simbolo
                        sector_en  = info_ticker.get("sector")
                        indus_en   = info_ticker.get("industry")
                        pais_en    = info_ticker.get("country")
                        resumen_en = info_ticker.get("longBusinessSummary") or ""
                        # Actividad: 1-2 frases del resumen (o la industria si no hay resumen)
                        actividad_en = ""
                        if resumen_en:
                            frases = resumen_en.replace("\n", " ").split(". ")
                            actividad_en = ". ".join(frases[:2]).strip()
                            if actividad_en and not actividad_en.endswith("."):
                                actividad_en += "."
                        try:
                            _trad = GoogleTranslator(source="auto", target="es")
                            indus_es     = _trad.translate(indus_en) if indus_en else ""
                            actividad_es = _trad.translate(actividad_en) if actividad_en else ""
                        except Exception:
                            indus_es, actividad_es = (indus_en or ""), (actividad_en or "")
                        ficha_empresa = {
                            "nombre":    nombre_emp,
                            "sector":    SECTORES_ES.get(sector_en, sector_en) if sector_en else "N/A",
                            "industria": indus_es or (indus_en or "N/A"),
                            "actividad": actividad_es or (actividad_en or ""),
                            "pais":      PAISES_ES.get(pais_en, pais_en) if pais_en else "N/A",
                        }
                        # Perfil real al prompt: ancla el análisis a la empresa
                        # correcta (sector, actividad y país) en vez de inferir del ticker
                        datos_extra_str = (
                            f"PERFIL DE LA EMPRESA — Nombre: {nombre_emp}; "
                            f"Sector: {sector_en or 'N/A'}; Industria: {indus_en or 'N/A'}; "
                            f"País: {pais_en or 'N/A'}. "
                            f"A qué se dedica: {actividad_en or 'N/A'} "
                        ) + datos_extra_str

                        # Fundamentales clave (auditoría P8 + tabla del auditor):
                        # valoración (P/E, P/S), eficiencia (ROE, margen) y
                        # riesgo financiero (Deuda/EBITDA, FCF, deuda total)
                        _fmt_b = lambda v: f"USD {v/1e9:,.1f}B" if v else "N/A"
                        pe_t   = info_ticker.get('trailingPE')
                        fpe    = info_ticker.get('forwardPE')
                        ps     = info_ticker.get('priceToSalesTrailing12Months')
                        roe    = info_ticker.get('returnOnEquity')
                        margen = info_ticker.get('operatingMargins')
                        deuda  = info_ticker.get('totalDebt')
                        ebitda = info_ticker.get('ebitda')
                        fcf    = info_ticker.get('freeCashflow')
                        deuda_ebitda = (deuda / ebitda) if (deuda and ebitda and ebitda > 0) else None

                        # Yahoo omite trailingPE cuando las ganancias son negativas
                        eps_t = info_ticker.get('trailingEps')
                        if pe_t:
                            pe_txt = f"{pe_t:.1f}x"
                        elif eps_t is not None and eps_t < 0:
                            pe_txt = "Negativo"  # como en la tabla del auditor (caso INTC)
                        else:
                            pe_txt = "N/A"
                        fund_ui = {
                            "P/E":          pe_txt,
                            "P/S":          f"{ps:.1f}x" if ps else "N/A",
                            "ROE":          f"{roe*100:.1f}%" if roe is not None else "N/A",
                            "Deuda/EBITDA": f"{deuda_ebitda:.2f}x" if deuda_ebitda is not None else "N/A",
                            "FCF":          _fmt_b(fcf),
                        }

                        datos_extra_str += (
                            f" FUNDAMENTALES — Valoración: P/E {fund_ui['P/E']}, "
                            f"Forward P/E {f'{fpe:.1f}x' if fpe else 'N/A'}, P/S (ventas) {fund_ui['P/S']}. "
                            f"Eficiencia: ROE {fund_ui['ROE']}, Margen operativo "
                            f"{f'{margen*100:.1f}%' if margen else 'N/A'}. "
                            f"Riesgo financiero: Deuda/EBITDA {fund_ui['Deuda/EBITDA']}, "
                            f"Deuda total {_fmt_b(deuda)}, Free Cash Flow {_fmt_b(fcf)}."
                        )

                        # Earnings próximos: ningún plan a días debería ignorar un gap inminente
                        try:
                            cal = tk_fund.calendar
                            fechas_earnings = (cal or {}).get("Earnings Date", []) if isinstance(cal, dict) else []
                            if fechas_earnings:
                                prox = fechas_earnings[0]
                                dias_e = (pd.Timestamp(prox).date() - datetime.date.today()).days
                                if dias_e >= 0:
                                    datos_extra_str += (
                                        f" ⚠️ PRÓXIMOS EARNINGS: {prox} (en {dias_e} días) — "
                                        f"riesgo de gap si cae dentro del horizonte operativo."
                                    )
                        except Exception:
                            pass
                    except Exception:
                        pass

                # ── Llamada al cerebro completo
                analisis = analizar_con_gemini(
                    simbolo         = simbolo,
                    precio_actual   = precio_actual,
                    recomendacion   = recomendacion_str,
                    textos_noticias = noticias_str or "Sin noticias.",
                    tipo_mercado    = tipo_mercado,
                    datos_extra     = datos_extra_str,
                    sentimiento     = sentimiento_str,
                    datos_onchain   = onchain_str,
                    ciclo_macro     = ciclo_str,
                    temporalidad    = temporalidad,
                    macro_global    = macro_global_str,
                )

            st.success("Análisis completado:")

            # ── 🏢 Ficha de la empresa: nombre, sector, actividad y país
            #     (identidad de un vistazo, antes del análisis técnico)
            if ficha_empresa:
                st.markdown(f"#### 🏢 {ficha_empresa['nombre']}")
                ie1, ie2, ie3 = st.columns(3)
                ie1.metric("Sector",    ficha_empresa["sector"])
                ie2.metric("Industria", ficha_empresa["industria"])
                ie3.metric("País",      ficha_empresa["pais"])
                if ficha_empresa.get("actividad"):
                    st.caption(f"**A qué se dedica:** {ficha_empresa['actividad']}")

            mostrar_metricas_macro(ctx_macro_global)

            # ── 📊 Fila de fundamentales (tabla del auditor: valoración,
            #     eficiencia y riesgo financiero de un vistazo)
            if fund_ui:
                fc = st.columns(5)
                fc[0].metric("P/E (ganancias)",  fund_ui["P/E"])
                fc[1].metric("P/S (ventas)",     fund_ui["P/S"])
                fc[2].metric("ROE (eficiencia)", fund_ui["ROE"])
                fc[3].metric("Deuda/EBITDA",     fund_ui["Deuda/EBITDA"])
                fc[4].metric("Free Cash Flow",   fund_ui["FCF"])

            # ── 📡 Salud de la fuente de opciones: estos niveles (muros, Max
            #     Pain, flujo) entraron al prompt, así que conviene ver de dónde
            #     salieron y con qué cobertura
            if opciones_usadas:
                mostrar_salud_datos(simbolo)

            # ── Validación numérica del plan (auditoría P7):
            #    la IA redacta, la aritmética se verifica en Python
            plan, analisis_limpio = extraer_plan(analisis)
            st.markdown(analisis_limpio)

            if plan:
                # ATR de sanidad solo en marcos cortos: en Semanal/Mensual el
                # stop se ancla a estructura y el ATR de vela alta dispararía
                # falsos avisos (espejo de regla_stop — parche del auditor)
                atr_sanidad = datos.get('atr_14') if temporalidad in ("1 Hora", "4 Horas", "Diario") else None
                v = validar_plan(plan, precio_actual, atr=atr_sanidad)
                # .get() defensivo: si la nube recarga FINGEM antes que los
                # módulos (desfase de redeploy), un dict viejo sin "direccion"
                # no debe tumbar el reporte completo
                direccion_v = v.get("direccion") or v.get("sesgo", "n/a")
                if v.get("entrada") and direccion_v not in ("fuera", "neutral"):
                    pc1, pc2, pc3, pc4 = st.columns(4)
                    pc1.metric("Operación", str(direccion_v).capitalize(),
                               delta=f"sesgo mercado: {v.get('sesgo', 'n/a')}", delta_color="off")
                    pc2.metric("Entrada", f"USD {v['entrada']:,.2f}")
                    pc3.metric("Stop", f"USD {v['stop']:,.2f}")
                    # Marca visual del R/B: un 0.9 pasa desapercibido entre
                    # números si no se señala que está bajo el mínimo sano
                    rb_val = v.get("rb")
                    if rb_val:
                        rb_txt = f" · R/B {rb_val} " + ("⚠️" if rb_val < RB_MINIMO else "✅")
                    else:
                        rb_txt = ""
                    pc4.metric("TP / R:B", f"USD {v['tp1']:,.2f}{rb_txt}")
                for aviso in v.get("avisos", []):
                    (st.warning if aviso.startswith("⚠️") else st.info if aviso.startswith("ℹ️") else st.success)(aviso)

            if enviar_telegram:
                enviar_alerta_telegram(analisis_limpio)

    else:
        st.error(
            f"No se pudieron descargar datos para **{simbolo}**. Posibles causas:\n\n"
            f"1. **Símbolo incorrecto** — usa el formato exacto de Yahoo Finance: "
            f"`AAPL` (NY), `BIMBOA.MX` / `WALMEX.MX` (BMV, con sufijo .MX), `BTC-USD` (cripto).\n"
            f"2. **Yahoo limitó las peticiones** temporalmente — espera 1-2 minutos y reintenta.\n\n"
            f"💡 Busca el símbolo exacto en [finance.yahoo.com](https://finance.yahoo.com)."
        )


# ==========================================
# 🧱 MÓDULO DERIVADOS
# ==========================================
elif tipo_mercado == "🧱 Flujo de Opciones (Derivados)" and simbolo:
    st.write("---")
    st.title(f"🧱 Radar Quant (Derivados Multitiempo): {simbolo}")
    st.write(
        "Visualiza el Gamma Squeeze de esta semana y rastrea el posicionamiento "
        "de las Ballenas para los próximos 6 meses."
    )

    if st.button(f"🕵️‍♂️ Extraer Flujo Institucional para {simbolo}", type="primary"):
        with st.spinner(f"Escaneando 4 horizontes temporales y procesando IA Quant..."):
            df_muros, reporte_ia, fig_visual, niveles_dia = escanear_flujo_institucional(simbolo)

        mostrar_salud_datos(simbolo)

        # ── 🎯 Niveles operables del día (vencimiento 1-3d)
        if niveles_dia:
            st.markdown(
                f"#### 🎯 Niveles del Día — vencimiento {niveles_dia.get('vencimiento', 'N/A')} "
                f"({niveles_dia.get('dte', '?')} DTE)"
            )
            n1, n2, n3, n4 = st.columns(4)
            _f = lambda v: f"USD {v:,.2f}" if v is not None else "N/A"
            n1.metric("🛡️ Soporte (Put Wall)",      _f(niveles_dia.get("put_wall")))
            n2.metric("🧱 Resistencia (Call Wall)", _f(niveles_dia.get("call_wall")))
            n3.metric("🧲 Max Pain (imán)",         _f(niveles_dia.get("max_pain")))
            pcr_dia = niveles_dia.get("pcr")
            n4.metric(
                "PCR del vencimiento",
                f"{pcr_dia:.2f}" if pcr_dia is not None else "N/A",
                delta=("Bajista" if pcr_dia > 1.1 else "Alcista" if pcr_dia < 0.8 else "Neutral") if pcr_dia else None,
                delta_color="inverse",
            )
            if niveles_dia.get("flujo_fresco"):
                st.warning(
                    "🔥 **Volumen inusual hoy (vol > OI):** " + " · ".join(niveles_dia["flujo_fresco"])
                    + " — dirección desconocida: zonas de interés, no apuestas confirmadas."
                )

        if fig_visual is not None:
            st.plotly_chart(fig_visual, width="stretch", config={'displayModeBar': False})

        if not df_muros.empty:
            st.markdown("### 🧱 Flujo Institucional (Todas las expiraciones)")
            st.dataframe(df_muros, width="stretch", hide_index=True)
            st.markdown("### 🧠 Setup de Trading y Visión Macro (IA)")

            # Validación numérica del plan (mismo pipeline que Análisis Individual)
            plan_d, reporte_limpio = extraer_plan(reporte_ia)
            st.info(reporte_limpio)
            if plan_d:
                # ATR diario del ticker: plan de 1-3 días → vara correcta
                v = validar_plan(plan_d, niveles_dia.get("spot", 0) or 0,
                                 atr=niveles_dia.get("atr_14"))
                dir_v = v.get("direccion", "n/a")
                if v.get("entrada") and dir_v not in ("fuera", "neutral", "n/a"):
                    vc1, vc2, vc3, vc4 = st.columns(4)
                    vc1.metric("Operación", str(dir_v).capitalize(),
                               delta=f"sesgo mercado: {v.get('sesgo', 'n/a')}", delta_color="off")
                    vc2.metric("Entrada", f"USD {v['entrada']:,.2f}")
                    vc3.metric("Stop", f"USD {v['stop']:,.2f}")
                    rb_d = v.get("rb")
                    vc4.metric("TP / R:B", f"USD {v['tp1']:,.2f}" +
                               (f" · R/B {rb_d} " + ("⚠️" if rb_d < RB_MINIMO else "✅") if rb_d else ""))
                for aviso in v.get("avisos", []):
                    (st.warning if aviso.startswith("⚠️") else st.info if aviso.startswith("ℹ️") else st.success)(aviso)
        else:
            st.warning(reporte_ia)


# ==========================================
# 🎯 MÓDULO: RADAR DE OPCIONES (SCORE QUANT)
# ==========================================
# Pantalla completamente independiente.
# Escanea múltiples activos en paralelo y
# genera score 0-100 por horizonte temporal.
# ==========================================

# Todos los tickers de estas listas tienen cadenas de opciones líquidas
# (volumen alto y spreads estrechos). No añadir símbolos sin mercado de
# opciones profundo: el radar los descarta y desperdicia requests a Yahoo.
UNIVERSOS_RADAR = {
    "🔥 Activos de Alta Liquidez (default)": [
        "SPY", "QQQ", "AAPL", "NVDA", "TSLA", "META", "MSFT",
        "AMZN", "AMD", "GOOGL", "AVGO", "NFLX", "IBIT", "GLD", "TLT",
    ],
    "🏦 Mega-Caps + Bancos": [
        "JPM", "GS", "BAC", "WFC", "MS", "C", "V", "MA", "AXP",
        "SCHW", "BRK-B", "XLF", "AAPL", "MSFT", "AMZN",
    ],
    "⚡ Tecnología / Semis": [
        "NVDA", "AMD", "INTC", "AVGO", "QCOM", "AMAT", "MU", "LRCX",
        "SMH", "SOXX", "TSM", "ASML", "ARM", "MRVL", "ON",
    ],
    "🚀 Alta Volatilidad / Momentum": [
        "TSLA", "NVDA", "PLTR", "COIN", "MSTR", "AMD", "MARA", "RIOT",
        "SOFI", "HOOD", "RIVN", "AFRM", "SNAP", "RBLX", "U",
    ],
    "🛢️ Energía + Materias Primas": [
        "XLE", "XOM", "CVX", "OXY", "SLB", "COP", "USO", "UNG",
        "GLD", "SLV", "FCX", "GDX",
    ],
    "🩺 Salud / Farma": [
        "LLY", "UNH", "JNJ", "PFE", "MRK", "ABBV", "AMGN", "MRNA",
        "BMY", "GILD", "XLV",
    ],
    "🛒 Consumo + Retail": [
        "AMZN", "WMT", "COST", "HD", "MCD", "KO", "PEP", "NKE",
        "SBUX", "DIS", "TGT", "LULU", "XLY",
    ],
    "🐉 China ADRs": [
        "BABA", "PDD", "JD", "NIO", "BIDU", "LI", "XPEV", "FXI", "KWEB",
    ],
    "🪙 Cripto-vinculados": [
        "IBIT", "MSTR", "COIN", "MARA", "RIOT", "CLSK", "GBTC", "HOOD",
    ],
    "📊 ETFs / Índices Macro": [
        "SPY", "QQQ", "IWM", "DIA", "TLT", "HYG", "GLD", "SLV",
        "VXX", "EEM", "XLF", "XLK", "XLE", "SMH",
    ],
    "✍️ Lista personalizada": [],
}

if tipo_mercado == "🎯 Radar de Opciones (Score Quant)":
    st.header("🎯 Radar de Opciones Institucional")
    st.caption(
        "Escanea múltiples activos en paralelo. "
        "Score 0–100 por horizonte: 🟢 ≥65 setup activo · 🟡 40-64 monitorear · 🔴 <40 sin señal. "
        "El umbral de disparo de señales BUY se ajusta en el sidebar."
    )

    # ── Sidebar: configuración del radar
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuración del Radar")

    universo_sel = st.sidebar.selectbox(
        "Universo de activos:", list(UNIVERSOS_RADAR.keys())
    )

    if universo_sel == "✍️ Lista personalizada":
        tickers_raw = st.sidebar.text_area(
            "Tickers (separados por coma):",
            value="SPY, QQQ, TSLA, NVDA, AAPL",
            height=90,
        )
        lista_radar = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    else:
        lista_radar = UNIVERSOS_RADAR[universo_sel]

    umbral_radar = st.sidebar.slider(
        "🎚️ Umbral de señal (score mínimo):",
        min_value=40, max_value=80, value=55, step=5,
        help=(
            "Score necesario para disparar BUY CALL/PUT. El score máximo teórico es ~89, "
            "así que 65+ exige confluencia casi perfecta. Con 55 detecta setups sólidos. "
            "Scores a menos de 10 puntos del umbral aparecen como LEAN (casi disparan)."
        ),
    )

    st.sidebar.info(
        f"**{len(lista_radar)} activos** a escanear.\n\n"
        f"Tiempo estimado: ~{len(lista_radar) * 2 // 8 + 5}–{len(lista_radar) * 3 // 8 + 10} seg."
    )

    # ── Mostrar lista seleccionada en main
    st.write(f"**Universo:** {universo_sel} — `{', '.join(lista_radar)}`")
    st.markdown("---")

    # ── Botón de escaneo
    if st.button("🚀 Iniciar Radar Quant", type="primary", width="stretch"):

        prog = st.progress(0, text="Inicializando conexiones...")
        status_box = st.empty()

        with st.spinner(f"Escaneando {len(lista_radar)} activos en los 3 horizontes..."):
            prog.progress(15, text="Descargando cadenas de opciones...")
            df_quant, df_swing, fig_gamma, reporte_ia, raw, contexto_macro = ejecutar_radar_opciones(
                lista_radar,
                umbral_senal=umbral_radar,
            )
            prog.progress(100, text="Análisis completado.")

        status_box.empty()
        # Resultado en session_state: el selector "Ver gráfico detallado" y el
        # botón de Telegram provocan un rerun, el botón del escaneo vuelve a
        # False y sin esto la pantalla completa se borraba tras cada interacción
        st.session_state["_radar_scan"] = {
            "df_quant": df_quant, "df_swing": df_swing, "fig_gamma": fig_gamma,
            "reporte_ia": reporte_ia, "raw": raw, "contexto": contexto_macro,
            "lista": lista_radar,
        }

    _scan = st.session_state.get("_radar_scan")
    if _scan:
        df_quant, df_swing  = _scan["df_quant"], _scan["df_swing"]
        fig_gamma, reporte_ia = _scan["fig_gamma"], _scan["reporte_ia"]
        raw, contexto_macro = _scan["raw"], _scan["contexto"]
        mostrar_salud_datos_lista(_scan["lista"])

        if df_quant.empty and df_swing.empty:
            st.warning("⚠️ No se obtuvieron datos de opciones. Verifica que los tickers tengan opciones listadas.")
        else:
            # ── Banner contexto macro
            ses = contexto_macro.get("sesgo_macro", "desconocido")
            ses_color = "#089981" if ses == "alcista" else "#F23645" if ses == "bajista" else "#FF9800"
            ses_emoji = "🟢" if ses == "alcista" else "🔴" if ses == "bajista" else "🟡"
            st.markdown(f"""            <div style="padding:10px 16px;border-radius:8px;background:rgba(255,255,255,0.04);
                        border-left:4px solid {ses_color};margin-bottom:12px;">
              <b>{ses_emoji} Contexto Macro del día — SPY/QQQ:</b>
              &nbsp; Sesgo: <b style="color:{ses_color}">{ses.upper()}</b>
              &nbsp;|&nbsp; SPY {contexto_macro.get('spy_ret3d',0):+.2f}% (3d)
              &nbsp;|&nbsp; QQQ {contexto_macro.get('qqq_ret3d',0):+.2f}% (3d)
            </div>""", unsafe_allow_html=True)

            # ── Métricas resumen (BUY firme vs LEAN casi-señal)
            if not df_swing.empty:
                señales_buy  = (df_swing["Señal"].str.contains("BUY")).sum()
                señales_lean = (df_swing["Señal"].str.contains("LEAN")).sum()
                señales_wait = (df_swing["Señal"].str.contains("WAIT")).sum()
            else:
                señales_buy = señales_lean = señales_wait = 0
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Activos escaneados", len(df_quant) if not df_quant.empty else 0)
            m2.metric("🟢 Señales BUY",  señales_buy)
            m3.metric("🟩 LEAN (casi)",  señales_lean)
            m4.metric("🟡 WAIT",         señales_wait)

            st.markdown("---")

            # ── TAB 1: Swing Deep Scan (señal direccional)
            tab1, tab2 = st.tabs(["🎯 Swing Deep Scan (1-3d)", "📊 Score Quant por Horizonte"])

            with tab1:
                st.caption("Señales direccionales BUY CALL / BUY PUT / WAIT con filtros técnicos (Squeeze · IV Rank · EMA · RSI · Liquidez). Ordenadas por score máximo.")
                if not df_swing.empty:
                    st.dataframe(df_swing, width="stretch", hide_index=True,
                        column_config={
                            "Score CALL": st.column_config.ProgressColumn("Score CALL", min_value=0, max_value=100),
                            "Score PUT":  st.column_config.ProgressColumn("Score PUT",  min_value=0, max_value=100),
                        })
                else:
                    st.info("Sin datos de Deep Scan.")

            with tab2:
                st.caption("Score Quant original 0-100 por horizonte: PCR extremo (30%) + Volumen inusual (25%) + Delta/OI (25%) + IV Spike (20%).")
                if not df_quant.empty:
                    st.dataframe(df_quant, width="stretch", hide_index=True,
                        column_config={
                            "Score Global": st.column_config.ProgressColumn("Score Global", min_value=0, max_value=100),
                        })
                else:
                    st.info("Sin datos de Score Quant.")

            st.markdown("---")

            # ── Gráfico detallado (selector de ticker)
            tickers_disp = df_swing["Ticker"].tolist() if not df_swing.empty else (df_quant["Ticker"].tolist() if not df_quant.empty else [])
            if tickers_disp:
                st.subheader("🔬 Análisis Detallado por Activo")
                ticker_detail = st.selectbox("Ver gráfico detallado de:", tickers_disp, index=0)

                if ticker_detail in raw:
                    fig_sel = construir_grafico_radar(raw[ticker_detail])
                    if fig_sel is not None:
                        st.plotly_chart(fig_sel, width="stretch", config={"displayModeBar": False})
                    else:
                        st.info("Datos de opciones insuficientes para graficar este activo.")

                    # Tabla de muros
                    todos_muros = []
                    for hk in ["scalping", "swing", "posicional"]:
                        dh = raw[ticker_detail].get(hk)
                        if dh and dh.get("muros"):
                            for m in dh["muros"]:
                                todos_muros.append({
                                    "Horizonte": hk.capitalize(), "Tipo": m["tipo"],
                                    "Strike": f"${m['strike']:.2f}", "OI": f"{m['oi']:,}",
                                    "Valor Total": f"${m['dinero_M']:.1f}M", "IV%": f"{m['iv']:.1f}%",
                                })
                    if todos_muros:
                        st.markdown(f"#### 🧱 Muros institucionales — {ticker_detail}")
                        st.dataframe(pd.DataFrame(todos_muros), width="stretch", hide_index=True)

                    # Sub-scores del Deep Scan
                    sc = raw[ticker_detail].get("swing_scan", {})
                    if sc:
                        st.markdown(f"#### 🧪 Desglose Score Swing — {ticker_detail}")
                        sub = {
                            "Filtro": ["RVOL Precio (15%)", "Squeeze BB/KC (20%)", "IV Rank (20%)",
                                       "Momentum EMA Bull (20%)", "Momentum EMA Bear (20%)",
                                       "RSI Bull (10%)", "RSI Bear (10%)", "Liquidez (15%)"],
                            "Sub-Score": [sc.get("_score_rvol","?"), sc.get("_score_squeeze","?"),
                                          sc.get("_score_iv_rank","?"), sc.get("_score_mom_bull","?"),
                                          sc.get("_score_mom_bear","?"), sc.get("_score_rsi_bull","?"),
                                          sc.get("_score_rsi_bear","?"), sc.get("_score_liquidez","?")],
                        }
                        st.dataframe(pd.DataFrame(sub), width="stretch", hide_index=True)

            st.markdown("---")

            # ── Reporte IA del ticker líder (con validación numérica del plan)
            if reporte_ia:
                lider_nombre = df_swing["Ticker"].iloc[0] if not df_swing.empty else "líder"
                st.subheader(f"🤖 Setup Accionable IA — {lider_nombre}")

                plan_r, reporte_limpio = extraer_plan(reporte_ia)
                st.info(reporte_limpio)
                if plan_r:
                    lider_raw = raw.get(lider_nombre, {})
                    precio_lider = lider_raw.get("precio", 0) or 0
                    # ATR diario ya calculado en el swing scan del líder
                    atr_lider = (lider_raw.get("swing_scan") or {}).get("atr_14")
                    v = validar_plan(plan_r, precio_lider, atr=atr_lider)
                    dir_v = v.get("direccion", "n/a")
                    if v.get("entrada") and dir_v not in ("fuera", "neutral", "n/a"):
                        rc1, rc2, rc3, rc4 = st.columns(4)
                        rc1.metric("Operación", str(dir_v).capitalize(),
                                   delta=f"sesgo mercado: {v.get('sesgo', 'n/a')}", delta_color="off")
                        rc2.metric("Entrada", f"USD {v['entrada']:,.2f}")
                        rc3.metric("Stop", f"USD {v['stop']:,.2f}")
                        rb_r = v.get("rb")
                        rc4.metric("TP / R:B", f"USD {v['tp1']:,.2f}" +
                                   (f" · R/B {rb_r} " + ("⚠️" if rb_r < RB_MINIMO else "✅") if rb_r else ""))
                    for aviso in v.get("avisos", []):
                        (st.warning if aviso.startswith("⚠️") else st.info if aviso.startswith("ℹ️") else st.success)(aviso)

                if st.button("📤 Enviar reporte a Telegram"):
                    enviar_alerta_telegram(f"🎯 *Radar Opciones v3 — {lider_nombre}*\n\n{reporte_limpio}")


# ==========================================
# 💸 MÓDULO: CALLS BARATOS (CAPITAL PEQUEÑO)
# ==========================================
# Encuentra contratos CALL concretos que caben
# en un presupuesto pequeño y son operables:
# liquidez real, spread razonable y delta viable.
# Datos: Yahoo Finance (los mismos que ve Webull).
# ==========================================

OPCION_FLUJO = "🔥 Flujo inusual (dinámico)"


@st.cache_data(ttl=180, show_spinner=False)
def _universo_flujo_cacheado(precio_min: float, precio_max: float, volumen_min: int) -> list[str]:
    """Screener de flujo con caché de 3 min.

    Sin la caché, cada rerun de Streamlit (mover CUALQUIER slider) dispararía
    un screener nuevo a Yahoo → castiga el rate limit. La caché reutiliza el
    resultado salvo que cambien los filtros del flujo; 3 min es coherente con
    'lo que se mueve hoy'.
    """
    return buscar_universo_flujo(
        precio_min=precio_min, precio_max=precio_max, volumen_min=volumen_min
    )


UNIVERSOS_BARATOS = {
    "💸 Acciones baratas líquidas (< USD 30)": [
        "F", "NIO", "SOFI", "AAL", "T", "PFE", "VALE", "ITUB",
        "MARA", "RIOT", "LCID", "RIVN", "SNAP", "CCL", "PLUG",
        "GRAB", "OPEN", "BBAI", "CHPT", "NOK",
    ],
    "🔥 Alta liquidez (contratos OTM baratos)": [
        "SPY", "QQQ", "AAPL", "AMD", "INTC", "PLTR", "COIN",
        "TSLA", "NVDA", "IBIT", "GLD", "XLF", "EEM",
    ],
}

if tipo_mercado == "💸 CALLs Baratos (Capital Pequeño)":
    st.header("💸 Buscador de CALLs Baratos")
    st.caption(
        "Contratos CALL concretos que caben en tu presupuesto y son **operables de verdad**: "
        "bid/ask activos, spread configurable, Open Interest real y delta ≥ 0.10 (sin loterías). "
        "Datos de Yahoo Finance (~15 min de retraso) — la misma cadena de opciones que muestra Webull.\n\n"
        "⚠️ **Earnings = 'Sí'** significa que el contrato vence DESPUÉS del próximo reporte: "
        "la prima puede desplomarse por IV crush aunque el precio suba. Trátalo como riesgo, no como oportunidad."
    )

    # ── Sidebar: configuración
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuración del Buscador")

    presupuesto = st.sidebar.slider(
        "💵 Presupuesto máx. por contrato (USD):",
        min_value=25, max_value=500, value=100, step=25,
        help="Costo total del contrato = prima (ask) × 100 acciones.",
    )
    dte_rango = st.sidebar.slider(
        "📅 Días al vencimiento (DTE):",
        min_value=7, max_value=120, value=(14, 60),
        help="Menos de 14d: theta te come rápido. Más de 60d: prima más cara pero más tiempo para que funcione.",
    )
    oi_minimo = st.sidebar.select_slider(
        "🌊 Open Interest mínimo:", options=[25, 50, 100, 300, 500], value=50,
        help="Contratos abiertos: más OI = más fácil entrar y salir a buen precio.",
    )
    spread_max = st.sidebar.slider(
        "📉 Spread bid/ask máximo (%):",
        min_value=10, max_value=40, value=25, step=5,
        help="Diferencia entre compra y venta. Menor = menos fuga al entrar y salir. "
             "Para capital pequeño, 25% o menos es lo sano.",
    )

    # Preselección + campo de lista manual SIEMPRE visible (mismo patrón que
    # derivados): no hace falta abrir/scrollear el desplegable para escribir
    # tu propia lista, y el dropdown queda corto y no se corta al fondo.
    # El tercer universo (flujo) es dinámico: se descubre con un screener.
    universo_b = st.sidebar.selectbox(
        "Universo:", list(UNIVERSOS_BARATOS.keys()) + [OPCION_FLUJO]
    )

    # Sliders del screener SOLO cuando se elige el universo dinámico
    if universo_b == OPCION_FLUJO:
        st.sidebar.caption("Filtros del screener dinámico:")
        rango_precio = st.sidebar.slider(
            "💲 Rango de precio del subyacente (USD):",
            min_value=1, max_value=100, value=(2, 60), step=1,
            help="Acota a precios donde el contrato cabe en presupuesto pequeño. "
                 "Precios muy bajos (<USD 2) concentran los pump-and-dump.",
        )
        vol_min_m = st.sidebar.select_slider(
            "🔊 Volumen diario mínimo del subyacente (millones):",
            options=[0.5, 1, 2, 5, 10, 20], value=2,
            help="Más volumen = más difícil de manipular con poco dinero. "
                 "Subirlo filtra los nombres finos y sospechosos.",
        )

    lista_manual_raw = st.sidebar.text_input(
        "…o escribe tu propia lista (coma):",
        "",
        help="Déjalo vacío para usar el universo de arriba. Ej: F, NIO, SOFI, SNAP",
    ).strip()

    # Prioridad: lista manual > universo elegido (estático o dinámico)
    if lista_manual_raw:
        lista_baratos = [t.strip().upper() for t in lista_manual_raw.split(",") if t.strip()]
    elif universo_b == OPCION_FLUJO:
        with st.spinner("🔥 Rastreando nombres con actividad inusual hoy..."):
            lista_baratos = _universo_flujo_cacheado(
                float(rango_precio[0]), float(rango_precio[1]), int(vol_min_m * 1_000_000)
            )
        if not lista_baratos:
            st.warning("⚠️ El screener no devolvió nombres. Prueba ampliar el rango "
                       "de precio o bajar el volumen mínimo, o reintenta en un minuto.")
    else:
        lista_baratos = UNIVERSOS_BARATOS[universo_b]

    st.write(f"**Universo:** {universo_b} — `{', '.join(lista_baratos)}`")
    st.markdown("---")

    # Banner de riesgo obligatorio para el universo dinámico: informar, no esconder.
    if universo_b == OPCION_FLUJO:
        st.warning(
            "⚠️ **Universo de alto riesgo — analiza a fondo antes de operar.**\n\n"
            "Estos nombres salen por su actividad inusual de HOY, que es también "
            "donde más abundan las trampas:\n"
            "- **Prima inflada / IV crush:** el volumen dispara la volatilidad implícita; "
            "puedes pagar el call carísimo y perder aunque el precio suba. Mira la columna "
            "**IV** y la de **⚠️ Earnings** antes de entrar.\n"
            "- **Pump-and-dump:** un volumen repentino en un nombre pequeño puede ser un "
            "movimiento coordinado que se desinfla igual de rápido.\n"
            "- **Falsos positivos:** volumen alto ≠ dirección alcista. Puede ser venta.\n\n"
            "La herramienta te acerca al flujo; **no distingue el flujo legítimo del cebo**. "
            "Ese filtro es tuyo. Con CALLs, la pérdida máxima es el 100% de la prima."
        )

    if st.button("🔍 Buscar CALLs dentro del presupuesto", type="primary", width="stretch"):
        with st.spinner(f"Escaneando cadenas de opciones de {len(lista_baratos)} activos..."):
            df_baratos = escanear_calls_baratos(
                lista_baratos,
                presupuesto_max=presupuesto,
                dte_min=dte_rango[0],
                dte_max=dte_rango[1],
                oi_min=oi_minimo,
                spread_max=spread_max,
            )

        mostrar_salud_datos_lista(lista_baratos)

        if df_baratos.empty:
            st.warning(
                "⚠️ Ningún contrato cumple los filtros. Prueba: subir el presupuesto, "
                "ampliar el rango DTE o bajar el Open Interest mínimo."
            )
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Contratos encontrados", len(df_baratos))
            costo_min = df_baratos["💵 Costo"].str.replace("$", "").str.replace(",", "").astype(float)
            m2.metric("Más barato", f"${costo_min.min():,.0f}")
            m3.metric("Score máximo", int(df_baratos["Score"].max()))

            st.dataframe(
                df_baratos, width="stretch", hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                },
            )
            st.caption(
                "**Cómo leer la tabla:** 💵 Costo = lo que pagas por 1 contrato (prima × 100). "
                "Breakeven = precio que debe alcanzar la acción al vencimiento para no perder. "
                "% al BE = cuánto tiene que subir el subyacente. "
                "Delta ≈ probabilidad aproximada de terminar ITM y sensibilidad al precio. "
                "El Score premia delta cercana a 0.40, breakeven cercano, IV baja, spread bajo y OI alto."
            )
            if universo_b == OPCION_FLUJO:
                st.caption(
                    "🔥 En el universo dinámico, desconfía de **IV > 80%**: suele ser prima inflada "
                    "que se desploma tras el evento. Un call con IV altísima necesita un movimiento "
                    "enorme solo para no perder."
                )


# ==========================================
# 💼 MÓDULO: PAPER TRADING (ALPACA)
# ==========================================
# Cierra el ciclo de la terminal: descubrir → discriminar → EJECUTAR Y MEDIR.
# Sin registro de resultados no hay forma de distinguir un sistema que
# funciona de una racha con suerte.
#
# ⚠️ Opera SIEMPRE contra la cuenta de PAPEL (simulada). La URL está fijada
# en broker_alpaca.py y no es configurable: no puede tocar dinero real.
# ==========================================
elif tipo_mercado == "💼 Paper Trading (Alpaca)":
    st.header("💼 Paper Trading — Cuenta Simulada")
    st.caption(
        "Ejecuta los planes de la terminal con dinero ficticio y mide el resultado real. "
        "🔒 Conectado exclusivamente a la cuenta de **papel** de Alpaca."
    )

    if not broker_activo():
        st.error(
            "🔑 **Faltan las credenciales de Alpaca.** Deben ir en los *secrets* de Streamlit "
            "como `ALPACA_API_KEY` y `ALPACA_SECRET_KEY`, con las llaves de **paper trading** "
            "(empiezan con `PK`)."
        )
        # Diagnóstico: adivinar por qué no las ve cuesta más que mirarlo.
        # Se muestran SOLO los nombres de las claves, nunca sus valores.
        try:
            claves = list(st.secrets.keys())
        except Exception:
            claves = []
        st.markdown("**Qué está viendo la app en tus secrets ahora mismo:**")
        if not claves:
            st.write("· *(ninguna)* — el archivo de secrets está vacío o no se guardó.")
        else:
            for k in claves:
                anidada = isinstance(st.secrets[k], dict)
                marca = "📁 sección (contenido anidado)" if anidada else "🔑 clave suelta"
                st.write(f"· `{k}` — {marca}")
        st.info(
            "**Las dos causas habituales:**\n\n"
            "1. **Encabezado de sección.** Si escribiste algo como `[alpaca]` encima, las claves "
            "quedan anidadas y la app no las encuentra. Deben ir **sueltas, sin ninguna leyenda "
            "antepuesta**, igual que tus claves de Telegram o Gemini.\n"
            "2. **Falta reiniciar.** Tras guardar los secrets, Streamlit Cloud a veces sigue con "
            "los viejos: usa **Manage app → Reboot app**.\n\n"
            "Formato correcto (cada una en su propia línea, al mismo nivel que las demás):\n\n"
            "`ALPACA_API_KEY = \"PK...\"`\n\n"
            "`ALPACA_SECRET_KEY = \"...\"`"
        )
        st.stop()

    cuenta = estado_cuenta()
    if not cuenta.get("ok"):
        st.error(f"No se pudo conectar con Alpaca: {cuenta.get('error')}")
        st.stop()

    # ── Estado del mercado: si está cerrado, las órdenes quedan en cola
    reloj = mercado_abierto()
    if reloj.get("abierto"):
        st.success(f"🟢 Mercado ABIERTO · cierra {reloj.get('proximo_cierre', '')[:16].replace('T', ' ')}")
    else:
        st.info(
            f"🔴 Mercado cerrado · abre {reloj.get('proxima_apertura', '')[:16].replace('T', ' ')}. "
            f"Las órdenes que envíes quedarán en cola hasta la apertura."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", f"USD {cuenta['equity']:,.2f}",
              delta=f"{cuenta['pl_dia']:+,.2f} hoy ({cuenta['pl_dia_pct']:+.2f}%)")
    c2.metric("Efectivo",         f"USD {cuenta['efectivo']:,.2f}")
    c3.metric("Poder de compra",  f"USD {cuenta['poder_compra']:,.2f}")
    c4.metric("Nivel de opciones", f"Nivel {cuenta['nivel_opciones']}")

    st.markdown("---")

    # ── Posiciones abiertas
    st.subheader("📌 Posiciones Abiertas")
    df_pos = posiciones_abiertas()
    if df_pos.empty:
        st.info("Sin posiciones abiertas. Envía una orden abajo para empezar.")
    else:
        st.dataframe(df_pos, width="stretch", hide_index=True)
        st.caption(
            "Ordenadas por P&L: las que peor van aparecen primero, "
            "para revisar el riesgo antes que la ganancia."
        )
        col_cerrar, _ = st.columns([2, 3])
        simbolo_cerrar = col_cerrar.selectbox(
            "Cerrar posición:", [""] + df_pos["Símbolo"].tolist(), key="cerrar_pos"
        )
        if simbolo_cerrar and col_cerrar.button(f"❌ Cerrar {simbolo_cerrar} a mercado"):
            ok_c, msg_c = cerrar_posicion(simbolo_cerrar)
            (st.success if ok_c else st.error)(msg_c)
            st.rerun()

    st.markdown("---")

    # ── Envío de órdenes
    st.subheader("🚀 Enviar Orden")
    instrumento = st.radio(
        "Instrumento:", ["🎯 Opción (contrato OCC)", "🏢 Acción"], horizontal=True
    )
    es_opcion_ui = instrumento.startswith("🎯")

    oc1, oc2 = st.columns(2)
    if es_opcion_ui:
        simbolo_orden = oc1.text_input(
            "Contrato OCC:", "",
            help="Formato: TICKER + AAMMDD + C/P + strike x1000 en 8 dígitos. "
                 "Ej: AAPL260724C00330000 = call de AAPL, 24-jul-2026, strike 330.",
        ).upper().strip()
        oc2.info(
            "⚠️ Alpaca **no admite bracket en opciones**: solo se envía la entrada. "
            "El stop y el objetivo debes vigilarlos tú desde la terminal."
        )
        stop_ui = tp_ui = None
    else:
        simbolo_orden = oc1.text_input("Ticker:", "", help="Ej: AAPL, NVDA, SPY").upper().strip()
        stop_ui = oc2.number_input("Stop loss (USD, 0 = sin bracket):",   min_value=0.0, value=0.0, step=0.01)
        tp_ui   = oc2.number_input("Take profit (USD, 0 = sin bracket):", min_value=0.0, value=0.0, step=0.01)

    tc1, tc2, tc3 = st.columns(3)
    lado_ui = tc1.selectbox("Lado:", ["Comprar", "Vender"])
    tipo_ui = tc2.selectbox("Tipo de orden:", ["limit", "market"])
    precio_lim = tc3.number_input(
        "Precio límite (USD):", min_value=0.0, value=0.0, step=0.01,
        disabled=(tipo_ui == "market"),
    )

    # ── Calculadora de tamaño: la pieza que faltaba en la terminal
    st.markdown("##### 📏 Tamaño de la posición")
    st.caption(
        "El plan de la IA dice dónde entrar y salir, pero no **cuánto**. "
        "Con capital pequeño esa es la decisión que más pesa."
    )
    sc1, sc2, sc3 = st.columns(3)
    riesgo_pct_ui = sc1.slider("Riesgo por operación (% del equity):", 0.25, 5.0, 1.0, 0.25)
    precio_ref = precio_lim if precio_lim > 0 else 0.0
    sugerido = calcular_tamano(
        capital        = cuenta["equity"],
        riesgo_pct     = riesgo_pct_ui,
        precio_entrada = precio_ref,
        precio_stop    = (stop_ui if (stop_ui and not es_opcion_ui) else None),
        es_opcion      = es_opcion_ui,
    )
    sc2.metric("Sugerido",   f"{sugerido['cantidad']} {'contratos' if es_opcion_ui else 'acciones'}")
    sc3.metric("Riesgo real", f"USD {sugerido.get('riesgo_real', 0):,.2f}")
    if sugerido.get("aviso"):
        st.warning(sugerido["aviso"])
    if precio_ref <= 0:
        st.caption("💡 Escribe un **precio límite** arriba para que calcule el tamaño sugerido.")

    cantidad_ui = st.number_input(
        "Cantidad a enviar:", min_value=0, value=int(sugerido["cantidad"]), step=1
    )
    if cantidad_ui > 0 and precio_ref > 0:
        costo_est = cantidad_ui * precio_ref * (100 if es_opcion_ui else 1)
        st.caption(f"Costo estimado de la operación: **USD {costo_est:,.2f}**")

    # ── Confirmación explícita: nada se envía por accidente
    confirmar = st.checkbox("Confirmo que quiero enviar esta orden a la cuenta de papel")
    if st.button("🚀 Enviar orden", type="primary", disabled=not confirmar):
        if not simbolo_orden:
            st.error("Falta el símbolo.")
        elif cantidad_ui < 1:
            st.error("La cantidad debe ser al menos 1.")
        else:
            lado_api = "buy" if lado_ui == "Comprar" else "sell"
            if es_opcion_ui:
                ok_o, msg_o = enviar_orden_opcion(
                    simbolo_orden, int(cantidad_ui), lado=lado_api,
                    tipo=tipo_ui, precio_limite=(precio_lim or None),
                )
            else:
                ok_o, msg_o = enviar_orden_accion(
                    simbolo_orden, int(cantidad_ui), lado=lado_api,
                    tipo=tipo_ui, precio_limite=(precio_lim or None),
                    stop_loss=(stop_ui or None), take_profit=(tp_ui or None),
                )
            (st.success if ok_o else st.error)(msg_o)

    st.markdown("---")

    # ── Historial de órdenes
    st.subheader("📜 Órdenes Recientes")
    df_ord = ordenes(estado="all", limite=25)
    if df_ord.empty:
        st.info("Todavía no has enviado ninguna orden.")
    else:
        st.dataframe(df_ord.drop(columns=["id"]), width="stretch", hide_index=True)
        pendientes = df_ord[df_ord["Estado"].isin(
            ["new", "accepted", "pending_new", "partially_filled"]
        )]
        if not pendientes.empty:
            id_cancelar = st.selectbox(
                "Cancelar orden pendiente:", [""] + pendientes["id"].tolist(),
                format_func=lambda x: "" if not x else
                    f"{pendientes[pendientes['id'] == x]['Símbolo'].iloc[0]} ({x[:8]})",
            )
            if id_cancelar and st.button("❌ Cancelar orden"):
                ok_x, msg_x = cancelar_orden(id_cancelar)
                (st.success if ok_x else st.error)(msg_x)
                st.rerun()

