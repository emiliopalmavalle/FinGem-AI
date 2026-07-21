import yfinance as yf
import pandas as pd
from datetime import datetime
from modules.ai_client import llamar_ia
import plotly.graph_objects as go
from typing import Optional


# ==========================================
# 🤖 ANÁLISIS IA — DESACOPLADO DE STREAMLIT
# ==========================================
# Usa el cliente multi-proveedor (Claude → Gemini → local).
# Las API keys las configura el orquestador vía configurar_ia().
# ==========================================

def analizar_muros_con_ia(
    ticker: str,
    precio: float,
    tabla_datos: str,
    pcr_global: float,
    niveles_dia: dict | None = None,
) -> str:
    """
    Genera el reporte IA de flujo institucional con foco en day trading 1-3d.

    Parámetros
    ----------
    ticker      : símbolo del activo
    precio      : precio spot actual
    tabla_datos : string con la tabla de muros (DataFrame.to_string)
    pcr_global  : Put/Call Ratio de todos los horizontes
    niveles_dia : niveles operables del vencimiento de day trading
                  (put_wall, call_wall, max_pain, pcr, flujo_fresco, dte)
    """
    from modules.validador_plan import REGLAS_RIESGO_PROMPT as reglas_riesgo

    n = niveles_dia or {}
    fmt = lambda v: f"USD {v:.2f}" if v is not None else "N/A"
    seccion_dia = f"""
    NIVELES OPERABLES DEL VENCIMIENTO DE DAY TRADING (DTE {n.get('dte', '?')}, {n.get('vencimiento', 'N/A')}):
    - Soporte (Put Wall, mayor OI abajo del spot): {fmt(n.get('put_wall'))}
    - Resistencia (Call Wall, mayor OI arriba del spot): {fmt(n.get('call_wall'))}
    - Max Pain (imán de vencimiento): {fmt(n.get('max_pain'))}
    - Put/Call Ratio de ESTE vencimiento: {n.get('pcr') if n.get('pcr') is not None else 'N/A'}
    - Volumen INUSUAL hoy (vol > OI): {'; '.join(n.get('flujo_fresco', [])) or 'ninguno detectado'}
      OJO: la dirección de ese volumen (compra o venta, spread o roll) es DESCONOCIDA con estos
      datos — trátalo como zona de interés institucional, NO como apuesta direccional confirmada.
    """

    prompt = f"""
    Actúa como un Quant Institucional especializado en day trading de 1 a 3 días.
    Analiza la liquidez para {ticker}. Precio actual Spot: USD {precio:.2f}.
    Put/Call Ratio Global (todos los horizontes): {pcr_global:.2f}.
    {seccion_dia}
    TABLA DE MUROS (Múltiples horizontes temporales):
    {tabla_datos}

    Tu tarea (prioridad 1 es lo más importante):
    1. PLAN DE DAY TRADING (1-3 días): usando Put Wall como soporte, Call Wall como resistencia
       y Max Pain como imán, define ENTRADA exacta, STOP LOSS estricto (fuera del muro) y
       TAKE PROFIT (antes del muro opuesto). Indica si el sesgo del día es alcista, bajista o
       rango, considerando el PCR del vencimiento y el flujo fresco.
    {reglas_riesgo}
    2. VOLUMEN INUSUAL: si hay strikes con volumen > OI, señálalos como zonas de interés/imanes
       del día. NO afirmes si fue compra o venta — esa información no existe en estos datos.
    3. VISIÓN MACRO (breve): ¿hacia dónde apuesta el Smart Money en 30/90/180 días?

    Al FINAL del reporte añade un bloque de código con SOLO este JSON (números sin comillas):
    ```json
    {{"sesgo": "alcista|bajista|neutral", "direccion": "largo|corto|fuera", "entrada": 0.0, "stop": 0.0, "tp1": 0.0}}
    ```
    "sesgo" es la dirección del MERCADO; "direccion" es la OPERACIÓN del plan (pueden diferir).
    Si recomiendas quedarse fuera, usa "direccion": "fuera" y 0 en los niveles.

    REGLA TÉCNICA CRÍTICA: NO USES EL SÍMBOLO DE DÓLAR. Usa 'USD' (ejemplo: USD 400).
    Usa viñetas cortas. Máximo 300 palabras.
    """

    ctx = {"ticker": ticker, "precio": precio}
    return llamar_ia(prompt, contexto_fallback=ctx)


# ==========================================
# 🗓️ UTILITARIOS DE FECHAS
# ==========================================

def encontrar_fecha_cercana(fechas: list[str], dias_objetivo: int) -> str:
    """
    Encuentra el vencimiento de opciones más cercano a N días desde hoy.
    Usado para mapear horizontes Scalping/30d/90d/180d a fechas reales.
    """
    hoy = datetime.now()
    fechas_dt = [datetime.strptime(f, '%Y-%m-%d') for f in fechas]
    diferencias = [abs((f - hoy).days - dias_objetivo) for f in fechas_dt]
    return fechas[diferencias.index(min(diferencias))]


def encontrar_fecha_daytrading(fechas: list[str]) -> str:
    """Vencimiento para operativa de 1-3 días: el primero con DTE >= 1.

    En tickers muy líquidos (SPY, QQQ, TSLA) fechas[0] suele ser el
    vencimiento de HOY (0DTE) — sus muros expiran antes de cerrar una
    posición de 1-3 días. Saltamos al siguiente vencimiento vivo.
    """
    hoy = datetime.now()
    for f in fechas:
        if (datetime.strptime(f, '%Y-%m-%d') - hoy).days >= 1:
            return f
    return fechas[0]


# ==========================================
# 🎯 NIVELES OPERABLES DEL DÍA (1-3 DÍAS)
# ==========================================

def calcular_max_pain(cadena) -> Optional[float]:
    """Max Pain: el strike donde el pago total a los compradores de
    opciones es mínimo — el precio "imán" hacia el que los market makers
    tienen incentivo a llevar el subyacente al vencimiento.
    """
    try:
        calls = cadena.calls[['strike', 'openInterest']].fillna(0)
        puts  = cadena.puts[['strike', 'openInterest']].fillna(0)
        strikes = sorted(set(calls['strike']).union(puts['strike']))
        if not strikes:
            return None
        # Sin Open Interest el max pain no existe: todos los strikes pagarían
        # 0 y ganaría el primero de la lista, devolviendo un número inventado
        # con apariencia de real. Preferimos N/A a un dato falso.
        if (calls['openInterest'].sum() + puts['openInterest'].sum()) <= 0:
            return None
        mejor, menor_pago = None, float('inf')
        for s in strikes:
            pago_calls = ((s - calls['strike']).clip(lower=0) * calls['openInterest']).sum()
            pago_puts  = ((puts['strike'] - s).clip(lower=0) * puts['openInterest']).sum()
            pago = pago_calls + pago_puts
            if pago < menor_pago:
                menor_pago, mejor = pago, s
        return float(mejor)
    except Exception:
        return None


def calcular_niveles_dia(cadena, precio_spot: float) -> dict:
    """Niveles operables del vencimiento de day trading.

    - put_wall:  mayor OI en puts DEBAJO del spot (soporte del día)
    - call_wall: mayor OI en calls ARRIBA del spot (resistencia del día)
    - max_pain:  imán de vencimiento
    - pcr:       Put/Call Ratio por volumen SOLO de este vencimiento
    - flujo_fresco: strikes con volumen de hoy > OI (posicionamiento
      nuevo, la señal de actividad inusual más útil para 1-3 días)
    """
    niveles = {"put_wall": None, "call_wall": None, "max_pain": None,
               "pcr": None, "flujo_fresco": []}
    try:
        calls = cadena.calls[['strike', 'openInterest', 'volume']].fillna(0)
        puts  = cadena.puts[['strike', 'openInterest', 'volume']].fillna(0)

        # Muros más cercanos al spot (dentro de ±8%: los operables del día)
        arriba = calls[(calls['strike'] > precio_spot) & (calls['strike'] <= precio_spot * 1.08)]
        if not arriba.empty and arriba['openInterest'].max() > 0:
            niveles["call_wall"] = float(arriba.loc[arriba['openInterest'].idxmax(), 'strike'])

        abajo = puts[(puts['strike'] < precio_spot) & (puts['strike'] >= precio_spot * 0.92)]
        if not abajo.empty and abajo['openInterest'].max() > 0:
            niveles["put_wall"] = float(abajo.loc[abajo['openInterest'].idxmax(), 'strike'])

        niveles["max_pain"] = calcular_max_pain(cadena)

        vol_c, vol_p = calls['volume'].sum(), puts['volume'].sum()
        niveles["pcr"] = float(vol_p / vol_c) if vol_c > 0 else None

        # Flujo fresco: volumen de hoy supera el OI acumulado (±10% del spot)
        frescos = []
        for tipo, df in [("CALL", calls), ("PUT", puts)]:
            zona = df[(df['strike'] >= precio_spot * 0.90) & (df['strike'] <= precio_spot * 1.10)]
            # El OI > 0 es obligatorio: si la fuente de datos lo entrega en 0
            # (yfinance lo hace), "volumen > OI" sería siempre cierto y cada
            # strike líquido se reportaría como posicionamiento nuevo
            inusual = zona[
                (zona['openInterest'] > 0)
                & (zona['volume'] > zona['openInterest'])
                & (zona['volume'] >= 500)
            ]
            for _, fila in inusual.sort_values('volume', ascending=False).head(2).iterrows():
                frescos.append(
                    f"{tipo} {fila['strike']:.0f} (vol {int(fila['volume']):,} vs OI {int(fila['openInterest']):,})"
                )
        niveles["flujo_fresco"] = frescos[:3]
    except Exception:
        pass
    return niveles


# ==========================================
# 📊 GRÁFICO DE MAPA GAMMA (SCALPING)
# ==========================================

def construir_grafico_opciones(
    cadena,
    precio_spot: float,
    ticker: str,
    fecha_cercana: str,
    max_pain: Optional[float] = None,
) -> go.Figure:
    """
    Construye el gráfico de barras de Open Interest por strike
    para el vencimiento de day trading.
    Calls = Resistencias (rojo) | Puts = Soportes (verde)
    Línea dorada = Max Pain (imán de vencimiento)
    """
    rango_min = precio_spot * 0.90
    rango_max = precio_spot * 1.10

    calls = cadena.calls[
        (cadena.calls['strike'] >= rango_min) &
        (cadena.calls['strike'] <= rango_max)
    ]
    puts = cadena.puts[
        (cadena.puts['strike'] >= rango_min) &
        (cadena.puts['strike'] <= rango_max)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=calls['strike'], y=calls['openInterest'],
        name='Calls (Resistencia)', marker_color='#F23645', opacity=0.85
    ))
    fig.add_trace(go.Bar(
        x=puts['strike'], y=puts['openInterest'],
        name='Puts (Soporte)', marker_color='#089981', opacity=0.85
    ))
    fig.add_vline(
        x=precio_spot,
        line_dash="solid", line_color="white", line_width=2,
        annotation_text=f"Spot: USD {precio_spot:.2f}"
    )
    if max_pain and rango_min <= max_pain <= rango_max:
        fig.add_vline(
            x=max_pain,
            line_dash="dot", line_color="#FFD700", line_width=2,
            annotation_text=f"🧲 Max Pain: USD {max_pain:.0f}",
            annotation_position="bottom right",
        )
    fig.update_layout(
        title=f"Mapa Gamma Day Trading ({fecha_cercana}) — {ticker}",
        xaxis_title="Strike Price (USD)",
        yaxis_title="Contratos Abiertos (OI)",
        barmode='group',
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ==========================================
# 🧱 MOTOR PRINCIPAL DE DERIVADOS
# ==========================================

def escanear_flujo_institucional(
    ticker_symbol: str,
) -> tuple[pd.DataFrame, str, Optional[go.Figure]]:
    """
    Escanea el flujo institucional de opciones para un activo.

    Cambios vs versión anterior:
    - IA vía cliente multi-proveedor (Claude → Gemini → local)
    - fillna(0) reemplazado por .fillna(0) con asignación explícita
    - Protección ante cadenas vacías en bucle de horizontes
    - Manejo de error más granular (no silencia todo con un except genérico)

    Retorna
    -------
    df_final    : DataFrame con los muros detectados por horizonte temporal
    reporte_ia  : Texto de análisis de la IA
    fig_visual  : Gráfico Plotly del mapa gamma de day trading (o None)
    niveles_dia : dict con put_wall, call_wall, max_pain, pcr, flujo_fresco,
                  dte y vencimiento del horizonte de 1-3 días
    """
    try:
        # Fuente CBOE: yfinance entrega openInterest=0, bid/ask=0 e IV basura,
        # lo que dejaba los muros en N/A y disparaba falsos "flujo fresco"
        from modules.opciones_cboe import (
            cadena_cboe, vencimientos_disponibles, precio_spot as spot_cboe,
        )

        ticker = yf.Ticker(ticker_symbol)
        fechas = vencimientos_disponibles(ticker_symbol)

        if not fechas:
            return pd.DataFrame(), "⚠️ Mercado ilíquido: no hay opciones disponibles para este activo.", None, {}

        hoy = datetime.now()

        # Precio spot: CBOE lo trae en la misma descarga de la cadena; solo
        # caemos a la llamada pesada de .info si esa vía falla
        precio_spot = spot_cboe(ticker_symbol) or 0.0
        if not precio_spot:
            info = ticker.info
            precio_spot = (
                info.get('regularMarketPrice') or
                info.get('currentPrice') or
                info.get('previousClose') or
                0.0
            )

        if precio_spot == 0.0:
            return pd.DataFrame(), "❌ No se pudo obtener el precio spot del activo.", None, {}

        # Mapa de horizontes: el de day trading salta el 0DTE (sus muros
        # expiran hoy — inútiles para posiciones de 1-3 días)
        fecha_dt = encontrar_fecha_daytrading(fechas)
        horizontes = {
            "⚡ Day Trading (1-3d)":  fecha_dt,
            "🗓️ Corto Plazo (30d)":  encontrar_fecha_cercana(fechas, 30),
            "🏛️ Mediano Plazo (90d)": encontrar_fecha_cercana(fechas, 90),
            "🐋 Largo Plazo (180d)": encontrar_fecha_cercana(fechas, 180),
        }

        muros_detectados = []
        fig_visual       = None
        niveles_dia      = {}
        fechas_procesadas = set()  # Evita duplicar si 30d y 90d caen en la misma fecha
        vol_c_total, vol_p_total = 0, 0

        for etiqueta, fecha_str in horizontes.items():
            if fecha_str in fechas_procesadas:
                continue
            fechas_procesadas.add(fecha_str)
            es_daytrading = etiqueta.startswith("⚡")

            try:
                dias_vencimiento = max(
                    (datetime.strptime(fecha_str, '%Y-%m-%d') - hoy).days, 0
                )
                cadena = cadena_cboe(ticker_symbol, fecha_str, spot=precio_spot)
            except Exception as e:
                # Error en un horizonte específico no detiene los demás
                continue

            # Niveles operables + gráfico solo para el horizonte de day trading
            if es_daytrading:
                niveles_dia = calcular_niveles_dia(cadena, precio_spot)
                niveles_dia["dte"] = dias_vencimiento
                niveles_dia["vencimiento"] = fecha_str
                niveles_dia["spot"] = precio_spot  # para el validador del plan en la UI
                # ATR(14) diario: vara de volatilidad para que el validador
                # chequee el stop del plan 1-3d (descarga ligera de 3 meses)
                try:
                    hist_atr = ticker.history(period="3mo")
                    if len(hist_atr) >= 15:
                        tr = pd.concat([
                            hist_atr['High'] - hist_atr['Low'],
                            (hist_atr['High'] - hist_atr['Close'].shift(1)).abs(),
                            (hist_atr['Low']  - hist_atr['Close'].shift(1)).abs(),
                        ], axis=1).max(axis=1)
                        niveles_dia["atr_14"] = float(tr.rolling(14).mean().iloc[-1])
                except Exception:
                    pass  # sin ATR el validador simplemente omite ese chequeo
                fig_visual = construir_grafico_opciones(
                    cadena, precio_spot, ticker_symbol, fecha_str,
                    max_pain=niveles_dia.get("max_pain"),
                )

            # Acumulación de volumen para PCR global
            if 'volume' in cadena.calls.columns:
                vol_c_total += cadena.calls['volume'].fillna(0).sum()
            if 'volume' in cadena.puts.columns:
                vol_p_total += cadena.puts['volume'].fillna(0).sum()

            # Detección de muros por tipo
            for tipo, df_opciones in [("CALL (Techo)", cadena.calls), ("PUT (Piso)", cadena.puts)]:
                if df_opciones.empty:
                    continue

                df_op = df_opciones.copy()
                for col in ('openInterest', 'lastPrice', 'volume'):
                    df_op[col] = df_op[col].fillna(0)

                # Valor apostado en millones
                df_op['Millones_Apostados'] = (
                    df_op['openInterest'] * df_op['lastPrice'] * 100
                ) / 1_000_000

                if es_daytrading:
                    # Day trading: los muros de gamma reales son OI puro
                    # CERCA del spot (±10%). Rankear por dinero apostado
                    # sesga a strikes ITM con prima alta que no actúan
                    # como soporte/resistencia intradía.
                    df_op = df_op[
                        (df_op['strike'] >= precio_spot * 0.90) &
                        (df_op['strike'] <= precio_spot * 1.10)
                    ]
                    df_muros_top = df_op.sort_values('openInterest', ascending=False).head(2)
                    umbral_ok = lambda fila: fila['openInterest'] >= 100
                else:
                    # Horizontes largos: el dinero apostado sí refleja el
                    # posicionamiento de ballenas (prima × OI)
                    df_muros_top = df_op.sort_values('Millones_Apostados', ascending=False).head(2)
                    umbral_ok = lambda fila: fila['Millones_Apostados'] > 0.5

                for _, fila in df_muros_top.iterrows():
                    if umbral_ok(fila):
                        flujo_nuevo = fila['volume'] > fila['openInterest'] > 0
                        muros_detectados.append({
                            "Horizonte":    etiqueta,
                            "DTE":          f"{dias_vencimiento}d",
                            "Tipo":         tipo,
                            "Strike":       f"USD {fila['strike']:.2f}",
                            "OI":           int(fila['openInterest']),
                            "Vol Hoy":      int(fila['volume']),
                            "Valor Total":  f"USD {fila['Millones_Apostados']:.2f} M",
                            "Señal":        "🔥 Flujo nuevo" if flujo_nuevo else "",
                        })

        # Put/Call Ratio global
        pcr_global = vol_p_total / vol_c_total if vol_c_total > 0 else 1.0

        df_final = pd.DataFrame(muros_detectados)

        # Análisis IA (multi-proveedor: Claude → Gemini → local)
        tabla_str = df_final.to_string(index=False) if not df_final.empty else "Sin muros detectados."
        reporte_ia = analizar_muros_con_ia(
            ticker_symbol, precio_spot, tabla_str, pcr_global, niveles_dia
        )

        return df_final, reporte_ia, fig_visual, niveles_dia

    except Exception as e:
        return pd.DataFrame(), f"❌ Error crítico en el escáner de derivados: {e}", None, {}
