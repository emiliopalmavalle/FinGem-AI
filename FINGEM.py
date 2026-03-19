import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import requests
import datetime
from google import genai
from deep_translator import GoogleTranslator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 🔐 CONFIGURACIÓN DE CREDENCIALES (SECRETS)
# ==========================================
# st.secrets lee de .streamlit/secrets.toml
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
ETHERSCAN_API_KEY = st.secrets["ETHERSCAN_API_KEY"]

# ==========================================
# 🛠️ FUNCIONES AUXILIARES (TELEGRAM, MACRO Y ON-CHAIN)
# ==========================================
def enviar_alerta_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    fragmentos = [mensaje[i:i+4000] for i in range(0, len(mensaje), 4000)]
    
    # 🎯 AQUÍ ESTÁ LA MAGIA: Lista de destinos
    # Reemplaza el número negativo por el ID real de tu grupo
    destinos = [TELEGRAM_CHAT_ID, "-1003711355206"] 
    
    envios_exitosos = 0
    for chat_destino in destinos:
        for fragmento in fragmentos:
            payload = {"chat_id": chat_destino, "text": fragmento, "parse_mode": "Markdown"}
            try:
                respuesta = requests.post(url, json=payload)
                if respuesta.status_code != 200:
                    payload_seguro = {"chat_id": chat_destino, "text": fragmento}
                    resp_segura = requests.post(url, json=payload_seguro)
                    if resp_segura.status_code != 200:
                        st.error(f"❌ Error Telegram ({chat_destino}): {resp_segura.text}")
                else:
                    envios_exitosos += 1
            except Exception as e:
                st.error(f"❌ Error de conexión al enviar a {chat_destino}: {e}")
                
    if envios_exitosos > 0:
        st.toast("✅ ¡Análisis enviado a Telegram (Privado y Grupo)!")

def obtener_sentimiento_macro():
    try:
        url = "https://api.alternative.me/fng/"
        respuesta = requests.get(url, timeout=5)
        datos = respuesta.json()
        valor = datos['data'][0]['value']
        clasificacion = datos['data'][0]['value_classification']
        return f"{valor}/100 ({clasificacion})"
    except:
        return "Desconocido"

# --- NUEVO: RELOJ DEL CICLO DEL HALVING ---
def calcular_fase_ciclo():
    """Calcula las semanas desde el último Halving y determina la fase macro"""
    fecha_halving = datetime.datetime(2024, 4, 19)
    hoy = datetime.datetime.now()
    dias_transcurridos = (hoy - fecha_halving).days
    semanas = int(dias_transcurridos / 7)
    
    # Lógica basada en el mapa de tiempo institucional
    if semanas < 40:
        fase = "Post-Halving (Acumulación temprana / Choque de oferta)"
        color = "🔵"
    elif 40 <= semanas < 77:
        fase = "Markup Parabólico (Profit START) - Tendencia Alcista Fuerte"
        color = "🟢"
    elif 77 <= semanas < 135:
        fase = "Distribución / Corrección (Last Call PROFIT END superado) - Mercado Bajista"
        color = "🔴"
    else:
        fase = "DCA START (Suelo del Mercado / Fase de Acumulación Pre-Halving)"
        color = "🟡"
        
    return semanas, fase, color

# --- RASTREO Y PERFILADO DE BALLENAS ---
def rastrear_ballena_btc(direccion_btc):
    url_stats = f"https://mempool.space/api/address/{direccion_btc}"
    url_txs = f"https://mempool.space/api/address/{direccion_btc}/txs"
    try:
        resp_stats = requests.get(url_stats, timeout=5)
        if resp_stats.status_code != 200: return "Error consultando mempool.space"
        datos = resp_stats.json()
        stats = datos['chain_stats']
        balance_btc = (stats['funded_txo_sum'] - stats['spent_txo_sum']) / 100000000
        tx_entradas = stats['funded_txo_count']
        tx_salidas = stats['spent_txo_count']
        total_txs = tx_entradas + tx_salidas
        
        resp_txs = requests.get(url_txs, timeout=5)
        historial = resp_txs.json() if resp_txs.status_code == 200 else []
        fecha_ultima_tx = "Desconocida"
        if historial and 'block_time' in historial[0]:
            fecha_ultima_tx = datetime.datetime.fromtimestamp(historial[0]['block_time']).strftime('%Y-%m-%d %H:%M')

        perfil = "Indeterminado"
        if total_txs < 5: perfil = "Billetera Nueva / Posible Exchange Interno (Baja Fiabilidad)"
        elif tx_salidas == 0 and tx_entradas > 5: perfil = "Diamond Hands / Acumulador Institucional (No vende)"
        elif tx_salidas > 0 and tx_entradas > 50: perfil = "Trader Activo / Fondo de Inversión (Alta Fiabilidad)"
            
        return f"💰 **Balance:** {balance_btc:,.2f} BTC\n\n📊 **Perfil Smart Money:** {perfil}\n\n🔄 **Transacciones:** {total_txs} (Entradas: {tx_entradas} | Salidas: {tx_salidas})\n\n⏱️ **Último Movimiento:** {fecha_ultima_tx}"
    except Exception as e:
        return f"Error: {e}"

def rastrear_ballena_eth(direccion_eth):
    url = f"https://api.etherscan.io/api?module=account&action=balance&address={direccion_eth}&tag=latest&apikey={ETHERSCAN_API_KEY}"
    try:
        respuesta = requests.get(url, timeout=5)
        datos = respuesta.json()
        if datos['status'] == '1':
            balance_eth = int(datos['result']) / 1000000000000000000
            return f"💰 **Balance:** {balance_eth:,.2f} ETH"
        return f"Error Etherscan: {datos['message']}"
    except Exception as e:
        return f"Error: {e}"

def escanear_anomalias_btc():
    try:
        hash_url = "https://mempool.space/api/blocks/tip/hash"
        block_hash = requests.get(hash_url, timeout=5).text
        txs_url = f"https://mempool.space/api/block/{block_hash}/txs"
        txs = requests.get(txs_url, timeout=10).json()

        miner_tx = txs[0]
        recompensa_minero = sum(out.get('value', 0) for out in miner_tx.get('vout', [])) / 100000000

        max_volumen = 0
        ballena_tx_id = ""
        for tx in txs[1:]:
            volumen_sats = sum(out.get('value', 0) for out in tx.get('vout', []))
            if volumen_sats > max_volumen:
                max_volumen = volumen_sats
                ballena_tx_id = tx['txid']

        return {"exito": True, "bloque_hash": block_hash[:10] + "...", "recompensa_minero": recompensa_minero, "ballena_tx": ballena_tx_id, "volumen_ballena": max_volumen / 100000000}
    except Exception as e:
        return {"exito": False, "error": str(e)}

# ==========================================
# 🧠 CEREBRO DE INTELIGENCIA ARTIFICIAL (GEMINI)
# ==========================================
def analizar_con_gemini(simbolo, precio, recomendacion, textos_noticias, mercado, datos_extra="", sentimiento="", datos_onchain="", ciclo_macro="", temporalidad=""):
    if mercado == "📈 Bolsa (NY / MX)":
        prompt = f"""
        Eres un analista financiero institucional. Analiza la acción: {simbolo}.
        - Precio actual: ${precio:.2f}
        - Datos Técnicos y Fundamentales: {datos_extra}
        - Consenso de analistas: {recomendacion}
        - Noticias: {textos_noticias}
        Genera un resumen ejecutivo en 3 puntos: 1. Acción del Precio. 2. Valoración Fundamental. 3. Veredicto Institucional.
        """
    else:
        # LÓGICA DINÁMICA: Separar Macro de Micro
        if temporalidad in ["Semanal", "Mensual"]:
            bloque_ciclo = f"- RELOJ DEL CICLO MACRO: {ciclo_macro}"
            regla_ciclo = '1. Contexto Cíclico: El "Reloj del Ciclo Macro" es tu brújula principal. Analiza la fase del Halving a largo plazo.'
            punto_1 = "1. 🕰️ Análisis del Ciclo Macro y Huella Institucional (Cruza el tiempo del Halving con el gráfico técnico)."
        else:
            bloque_ciclo = ""
            regla_ciclo = f'1. Contexto de Corto/Medio Plazo: Enfócate ESTRICTAMENTE en la acción del precio actual en {temporalidad}. IGNORA los ciclos macro de 4 años, enfócate en la liquidez inmediata.'
            punto_1 = f"1. ⚙️ Acción del Precio y Huella Institucional en {temporalidad}."

        prompt = f"""
        Eres un Analista Quant Senior de Criptomonedas. Tu especialidad es cruzar datos On-Chain y Análisis Técnico (T. Latino y SMC).
        Analiza el activo {simbolo} en temporalidad de {temporalidad}:
        - Precio actual: ${precio:.2f}
        {bloque_ciclo}
        - Sentimiento Retail (Fear & Greed): {sentimiento}
        - Actividad On-Chain (Escáner/Perfilado): {datos_onchain}
        - Datos Técnicos (EMA 55/200, ADX, Monitor, FVG): {datos_extra}
        - Noticias recientes: {textos_noticias}

        Reglas de análisis:
        {regla_ciclo}
        2. Correlación Ballena/Precio: Evalúa la fiabilidad de la ballena y si actúa como líder o seguidora respecto al ciclo y al gráfico.
        3. SMC y T. Latino: Confirma las zonas de liquidez (FVG) y la direccionalidad del Monitor.

        Genera un reporte agresivo y directo en 3 puntos:
        {punto_1}
        2. 🐋 Análisis de Liquidez y On-Chain.
        3. 💡 Veredicto Institucional y Operativa con levels.
        """

    client = genai.Client(api_key=GEMINI_API_KEY)
    respuesta = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
    return respuesta.text

# ==========================================
# 🖥️ INTERFAZ DE STREAMLIT Y BARRA LATERAL
# ==========================================
st.set_page_config(page_title="Terminal Financiero AI", page_icon="📈", layout="wide")
st.title("📊 Mi Terminal de Inteligencia Financiera")

st.sidebar.header("Panel de Control")
tipo_mercado = st.sidebar.radio("Selecciona el Mercado:", ["📈 Bolsa (NY / MX)", "🪙 Criptomonedas"])
simbolo = ""

if tipo_mercado == "📈 Bolsa (NY / MX)":
    region = st.sidebar.selectbox("Región:", ["🇺🇸 Wall Street (NY)", "🇲🇽 Bolsa Mexicana (BMV)", "✍️ Búsqueda Manual"])
    if region == "🇺🇸 Wall Street (NY)":
        opciones_ny = {"Apple (AAPL)": "AAPL", "Nvidia (NVDA)": "NVDA", "Microsoft (MSFT)": "MSFT", "Tesla (TSLA)": "TSLA"}
        simbolo = opciones_ny.get(st.sidebar.selectbox("Empresa:", list(opciones_ny.keys())), "")
    elif region == "🇲🇽 Bolsa Mexicana (BMV)":
        opciones_mx = {"Grupo México": "GMEXICOB.MX", "Walmart": "WALMEX.MX", "América Móvil": "AMXL.MX", "Banorte": "GFNORTEO.MX"}
        simbolo = opciones_mx.get(st.sidebar.selectbox("Empresa:", list(opciones_mx.keys())), "")
    else: simbolo = st.sidebar.text_input("Símbolo:", "AMD").upper()
else:
    opciones_cripto = {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "Solana (SOL)": "SOL-USD"}
    seleccion = st.sidebar.selectbox("Criptomoneda:", list(opciones_cripto.keys()) + ["✍️ Búsqueda Manual"])
    simbolo = st.sidebar.text_input("Símbolo:", "DOGE-USD").upper() if seleccion == "✍️ Búsqueda Manual" else opciones_cripto.get(seleccion, "")

temporalidad = st.sidebar.selectbox("Temporalidad (Velas):", ["1 Hora", "4 Horas", "Diario", "Semanal", "Mensual"], index=2)

# --- PANEL ON-CHAIN ---
red_onchain = "Ninguna"
direccion_ballena = ""
datos_escaner = None

if tipo_mercado == "🪙 Criptomonedas":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕵️‍♂️ Radar On-Chain")
    red_onchain = st.sidebar.selectbox("Rastrear Billetera Específica:", ["Ninguna", "Bitcoin (BTC)", "Ethereum (ETH)"])
    if red_onchain != "Ninguna": direccion_ballena = st.sidebar.text_input("Dirección de la Billetera:")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Escáner Caza-Ballenas")
    if st.sidebar.button("🔎 Escanear Último Bloque (BTC)"):
        with st.sidebar.status("Conectando a mempool.space..."): datos_escaner = escanear_anomalias_btc()

# ==========================================
# ⚙️ MOTOR PRINCIPAL DE DATOS Y GRÁFICOS
# ==========================================
if simbolo:
    st.write(f"---")
    st.write(f"## Analizando: {simbolo} ({tipo_mercado}) - Gráfico de {temporalidad}")
    ticker = yf.Ticker(simbolo)
    
    if temporalidad == "1 Hora": periodo_yf, intervalo_yf = "2mo", "1h"
    elif temporalidad == "4 Horas": periodo_yf, intervalo_yf = "3mo", "1h"
    elif temporalidad == "Semanal": periodo_yf, intervalo_yf = "10y", "1wk"
    elif temporalidad == "Mensual": periodo_yf, intervalo_yf = "max", "1mo"
    else: periodo_yf, intervalo_yf = "5y", "1d"
        
    hist = ticker.history(period=periodo_yf, interval=intervalo_yf)
    
    if not hist.empty:
        if temporalidad == "4 Horas": hist = hist.resample('4H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        precio_actual = hist['Close'].iloc[-1]
        info = ticker.info

        # Cálculos Técnicos
        hist['EMA_10'] = ta.trend.ema_indicator(hist['Close'], window=10)
        hist['EMA_55'] = ta.trend.ema_indicator(hist['Close'], window=55)
        hist['EMA_200'] = ta.trend.ema_indicator(hist['Close'], window=200)
        hist['ADX'] = ta.trend.adx(hist['High'], hist['Low'], hist['Close'], window=14)
        hist['Monitor'] = ta.trend.macd_diff(hist['Close'])
        
        ema_10, ema_55 = hist['EMA_10'].iloc[-1], hist['EMA_55'].iloc[-1]
        ema_200 = hist['EMA_200'].dropna().iloc[-1] if not hist['EMA_200'].dropna().empty else 0
        adx_actual, adx_previo = hist['ADX'].iloc[-1], hist['ADX'].iloc[-2]
        monitor_actual, monitor_previo = hist['Monitor'].iloc[-1], hist['Monitor'].iloc[-2]
        
        pendiente_adx = "Negativa (Pierde Fuerza) 📉" if adx_actual < adx_previo else "Positiva (Gana Fuerza) 📈"
        direccion_monitor = "Alcista 🟢" if monitor_actual > monitor_previo else "Bajista 🔴"

        fvg_bullish = fvg_bearish = "Sin Imbalance Cercano"
        for i in range(len(hist)-1, max(len(hist)-20, 2), -1):
            if hist['Low'].iloc[i] > hist['High'].iloc[i-2] and fvg_bullish == "Sin Imbalance Cercano": fvg_bullish = f"${hist['High'].iloc[i-2]:.2f} - ${hist['Low'].iloc[i]:.2f}"
            if hist['High'].iloc[i] < hist['Low'].iloc[i-2] and fvg_bearish == "Sin Imbalance Cercano": fvg_bearish = f"${hist['High'].iloc[i]:.2f} - ${hist['Low'].iloc[i-2]:.2f}"

        ha_df = hist.copy()
        ha_df['HA_Close'] = (hist['Open'] + hist['High'] + hist['Low'] + hist['Close']) / 4
        ha_open_list = [(hist['Open'].iloc[0] + hist['Close'].iloc[0]) / 2]
        for i in range(1, len(hist)): ha_open_list.append((ha_open_list[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2)
        ha_df['HA_Open'] = ha_open_list
        ha_df['HA_High'] = ha_df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
        ha_df['HA_Low'] = ha_df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

        # 🖥️ MÉTRICAS EN PANTALLA
        st.subheader(f"⚙️ Setup Institucional ({temporalidad})")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precio Actual", f"${precio_actual:,.2f}")
        col2.metric("T. Latino: Monitor", direccion_monitor)
        col3.metric("SMC: EMA 200", f"${ema_200:,.2f}" if ema_200 > 0 else "N/A")

        sentimiento_mercado = "N/A"
        datos_onchain_ia = ""
        ciclo_macro_ia = ""
        
        if tipo_mercado == "🪙 Criptomonedas":
            sentimiento_mercado = obtener_sentimiento_macro()
            col4.metric("🌐 Sentimiento Macro", sentimiento_mercado)
            
            # --- RENDERIZAR BANNER DEL RELOJ DEL CICLO ---
            semanas_h, fase_h, color_h = calcular_fase_ciclo()
            ciclo_macro_ia = f"Semana {semanas_h} post-halving. Fase: {fase_h}."
            
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 5px; background-color: rgba(255, 255, 255, 0.05); border-left: 5px solid {'#F23645' if 'Distribución' in fase_h else '#089981' if 'Markup' in fase_h else '#FFD700'};">
                <h4 style="margin:0; padding:0;">{color_h} Reloj del Ciclo Halving (Semana +{semanas_h}w)</h4>
                <p style="margin:5px 0 0 0; font-size:14px; opacity:0.8;"><b>Fase Actual:</b> {fase_h}</p>
            </div>
            <br>
            """, unsafe_allow_html=True)

            if datos_escaner and datos_escaner.get("exito"):
                st.success("✅ **Escáner On-Chain Completado (Último Bloque BTC)**")
                c1, c2 = st.columns(2)
                c1.info(f"🐋 **Mayor Transacción (Ballena):**\n\n Volumen: **{datos_escaner['volumen_ballena']:,.2f} BTC**\n\n TX ID: `{datos_escaner['ballena_tx']}`")
                c2.warning(f"⛏️ **Actividad de Mineros:**\n\n Recompensa/Movimiento: **{datos_escaner['recompensa_minero']:,.2f} BTC**\n\n Bloque: `{datos_escaner['bloque_hash']}`")
                datos_onchain_ia += f"Escáner en vivo: En el último bloque se detectó una transacción ballena de {datos_escaner['volumen_ballena']:.2f} BTC. El minero movió {datos_escaner['recompensa_minero']:.2f} BTC. "

            if red_onchain != "Ninguna" and direccion_ballena:
                if red_onchain == "Bitcoin (BTC)": resultado_ballena = rastrear_ballena_btc(direccion_ballena)
                elif red_onchain == "Ethereum (ETH)": resultado_ballena = rastrear_ballena_eth(direccion_ballena)
                st.info(f"**Billetera Monitoreada:** `{direccion_ballena}` \n\n {resultado_ballena}")
                datos_onchain_ia += f"Billetera vigilada: {direccion_ballena}. Estado: {resultado_ballena}."
        else:
            col4.metric("📊 Mercado", "Renta Variable")

        st.write(f"**T. Latino:** ADX {adx_actual:.2f} ({pendiente_adx}) | EMA 55: ${ema_55:,.2f}")
        st.write(f"**SMC Liquidez (FVG):** Imbalance Alc: `{fvg_bullish}` | Imbalance Baj: `{fvg_bearish}`")

        datos_extra = f"Temporalidad: {temporalidad}. EMA 10: ${ema_10:.2f}, EMA 55: ${ema_55:.2f}, EMA 200: ${ema_200:.2f}. ADX: {adx_actual:.2f} ({pendiente_adx}). Monitor: {direccion_monitor}. FVG Alcista: {fvg_bullish}. FVG Bajista: {fvg_bearish}."
        recomendacion_ia = "N/A"

        # 🎨 GRÁFICO PLOTLY Y ZOOM
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3], specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
        fig.add_trace(go.Candlestick(x=ha_df.index, open=ha_df['HA_Open'], high=ha_df['HA_High'], low=ha_df['HA_Low'], close=ha_df['HA_Close'], name='Heikin Ashi', increasing_line_color='#089981', decreasing_line_color='#F23645'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA_10'], mode='lines', line=dict(color='#2962FF', width=1.5), name='EMA 10'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA_55'], mode='lines', line=dict(color='#FF6D00', width=2), name='EMA 55'), row=1, col=1)
        if ema_200 > 0: fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA_200'], mode='lines', line=dict(color='white', width=2), name='EMA 200'), row=1, col=1)

        colores_monitor = []
        for i in range(len(hist)):
            if i == 0: colores_monitor.append('gray'); continue
            val, prev = hist['Monitor'].iloc[i], hist['Monitor'].iloc[i-1]
            if val >= 0: colores_monitor.append('#089981' if val > prev else '#006400')
            else: colores_monitor.append('#F23645' if val < prev else '#8B0000')

        fig.add_trace(go.Bar(x=hist.index, y=hist['Monitor'], marker_color=colores_monitor, name='Monitor', opacity=0.8), row=2, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['ADX'], mode='lines', line=dict(color='white', width=1.5), name='ADX'), row=2, col=1, secondary_y=True)
        fig.add_hline(y=23, line_dash="dot", line_color="gray", row=2, col=1, secondary_y=True)

     # --- CÁLCULO DE ZOOM (AMPLIADO Y MODO ORÁCULO) ---
        ultima_fecha = hist.index[-1]
        fecha_fin = ultima_fecha # Por defecto, el gráfico termina hoy
        
        if temporalidad == "1 Hora": 
            fecha_inicio = ultima_fecha - pd.Timedelta(days=10)
        elif temporalidad == "4 Horas": 
            fecha_inicio = ultima_fecha - pd.Timedelta(days=18)
        elif temporalidad == "Diario": 
            fecha_inicio = ultima_fecha - pd.DateOffset(months=7)
        elif temporalidad == "Semanal": 
            fecha_inicio = ultima_fecha - pd.DateOffset(years=7)
            fecha_fin = ultima_fecha + pd.DateOffset(months=10) # 🔮 MAGIA: 10 meses hacia el futuro
        else: 
            fecha_inicio = hist.index[0]
            
        fecha_inicio = max(fecha_inicio, hist.index[0])

        # --- 🗺️ MAPA DEL CICLO HALVING (SOLO EN SEMANAL Y CRIPTO) ---
        if temporalidad == "Semanal" and tipo_mercado == "🪙 Criptomonedas":
            ciclos = [
                {"halving": "2020-05-11", "start": "2021-02-15", "end": "2021-11-01", "dca": "2022-12-12"},
                {"halving": "2024-04-19", "start": "2025-01-24", "end": "2025-10-10", "dca": "2026-11-20"}
            ]
            for ciclo in ciclos:
                fig.add_vline(x=pd.to_datetime(ciclo["halving"]).timestamp() * 1000 if type(hist.index[0]) == pd.Timestamp else ciclo["halving"], line_dash="dot", line_color="#FF9800", line_width=2, opacity=0.8, row=1, col=1)
                fig.add_vline(x=pd.to_datetime(ciclo["start"]).timestamp() * 1000 if type(hist.index[0]) == pd.Timestamp else ciclo["start"], line_dash="solid", line_color="#089981", line_width=2, opacity=0.6, row=1, col=1)
                fig.add_vline(x=pd.to_datetime(ciclo["end"]).timestamp() * 1000 if type(hist.index[0]) == pd.Timestamp else ciclo["end"], line_dash="solid", line_color="#F23645", line_width=2, opacity=0.6, row=1, col=1)
                fig.add_vline(x=pd.to_datetime(ciclo["dca"]).timestamp() * 1000 if type(hist.index[0]) == pd.Timestamp else ciclo["dca"], line_dash="solid", line_color="#FFD700", line_width=2, opacity=0.6, row=1, col=1)

        fig.update_layout(template="plotly_dark", paper_bgcolor="#131722", plot_bgcolor="#131722", xaxis_rangeslider_visible=False, height=650, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        fig.update_xaxes(range=[fecha_inicio, fecha_fin], showgrid=False)                
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width='stretch')

        # 📰 NOTICIAS E IA
        st.write("---")
        noticias = ticker.news[:3]
        textos_noticias = ""
        if noticias:
            traductor = GoogleTranslator(source='auto', target='es')
            for n in noticias:
                titular = n.get('title', 'Sin título')
                try: titular_es = traductor.translate(titular)
                except: titular_es = titular
                textos_noticias += f"- {titular_es}\n"
        
    st.subheader("🤖 Análisis Asistido por Inteligencia Artificial")
    if st.button(f"Generar Análisis Híbrido ({temporalidad}) y Enviar a Telegram 🚀"):
            with st.spinner(f'Procesando Análisis Cuantitativo y Evaluando IA...'):
                try:
                    # AQUÍ ESTÁ EL CAMBIO: Agregamos la variable 'temporalidad' al final de esta línea 👇
                    analisis_ia = analizar_con_gemini(simbolo, precio_actual, recomendacion_ia, textos_noticias, tipo_mercado, datos_extra, sentimiento_mercado, datos_onchain_ia, ciclo_macro_ia, temporalidad)
                    
                    st.success("Análisis completado:")
                    st.markdown(analisis_ia.replace('$', '\\$'))
                    enviar_alerta_telegram(f"🚀 *REPORTE {temporalidad.upper()}: {simbolo}*\n\n{analisis_ia}")
                except Exception as e:
                    st.error(f"Error con la IA: {e}")
