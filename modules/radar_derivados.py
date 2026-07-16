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
) -> str:
    """
    Genera el reporte IA de flujo institucional.

    Parámetros
    ----------
    ticker      : símbolo del activo
    precio      : precio spot actual
    tabla_datos : string con la tabla de muros (DataFrame.to_string)
    pcr_global  : Put/Call Ratio calculado del batch
    """
    prompt = f"""
    Actúa como un Quant Institucional.
    Analiza la liquidez para {ticker}. Precio actual Spot: USD {precio:.2f}.
    Put/Call Ratio Global: {pcr_global:.2f}.

    TABLA DE MUROS (Múltiples horizontes temporales):
    {tabla_datos}

    Tu tarea:
    1. DAY TRADING (1-3 días): Analiza la fila 'Scalping/Semanal'. Define PRECIO DE ENTRADA, STOP LOSS estricto y TAKE PROFIT.
    2. VISIÓN MACRO: Analiza las filas de Corto, Mediano y Largo Plazo. ¿Hacia dónde está apostando el 'Smart Money' para los próximos meses?

    REGLA TÉCNICA CRÍTICA: NO USES EL SÍMBOLO DE DÓLAR. Usa 'USD' (ejemplo: USD 400). Usa viñetas cortas.
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


# ==========================================
# 📊 GRÁFICO DE MAPA GAMMA (SCALPING)
# ==========================================

def construir_grafico_opciones(
    cadena,
    precio_spot: float,
    ticker: str,
    fecha_cercana: str,
) -> go.Figure:
    """
    Construye el gráfico de barras de Open Interest por strike.
    Solo para la fecha de scalping (esta semana).
    Calls = Resistencias (rojo) | Puts = Soportes (verde)
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
    fig.update_layout(
        title=f"Mapa Gamma: Scalping de esta semana ({fecha_cercana}) — {ticker}",
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
    df_final   : DataFrame con los muros detectados por horizonte temporal
    reporte_ia : Texto de análisis generado por Gemini
    fig_visual : Gráfico Plotly del mapa gamma de scalping (o None)
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        fechas = ticker.options

        if not fechas:
            return pd.DataFrame(), "⚠️ Mercado ilíquido: no hay opciones disponibles para este activo.", None

        hoy = datetime.now()

        # Obtención del precio spot con fallbacks robustos
        info = ticker.info
        precio_spot = (
            info.get('regularMarketPrice') or
            info.get('currentPrice') or
            info.get('previousClose') or
            0.0
        )

        if precio_spot == 0.0:
            return pd.DataFrame(), "❌ No se pudo obtener el precio spot del activo.", None

        # Mapa de horizontes temporales
        horizontes = {
            "⚡ Scalping/Semanal":    fechas[0],
            "🗓️ Corto Plazo (30d)":  encontrar_fecha_cercana(fechas, 30),
            "🏛️ Mediano Plazo (90d)": encontrar_fecha_cercana(fechas, 90),
            "🐋 Largo Plazo (180d)": encontrar_fecha_cercana(fechas, 180),
        }

        muros_detectados = []
        fig_visual       = None
        fechas_procesadas = set()  # Evita duplicar si 30d y 90d caen en la misma fecha
        vol_c_total, vol_p_total = 0, 0

        for etiqueta, fecha_str in horizontes.items():
            if fecha_str in fechas_procesadas:
                continue
            fechas_procesadas.add(fecha_str)

            try:
                dias_vencimiento = max(
                    (datetime.strptime(fecha_str, '%Y-%m-%d') - hoy).days, 0
                )
                cadena = ticker.option_chain(fecha_str)
            except Exception as e:
                # Error en un horizonte específico no detiene los demás
                continue

            # Gráfico solo para scalping
            if etiqueta == "⚡ Scalping/Semanal":
                fig_visual = construir_grafico_opciones(
                    cadena, precio_spot, ticker_symbol, fecha_str
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
                df_op['openInterest'] = df_op['openInterest'].fillna(0)
                df_op['lastPrice']    = df_op['lastPrice'].fillna(0)

                # Valor apostado en millones
                df_op['Millones_Apostados'] = (
                    df_op['openInterest'] * df_op['lastPrice'] * 100
                ) / 1_000_000

                # Top 2 muros por horizonte
                df_muros_top = df_op.sort_values('Millones_Apostados', ascending=False).head(2)

                for _, fila in df_muros_top.iterrows():
                    if fila['Millones_Apostados'] > 0.5:  # Umbral mínimo de relevancia
                        muros_detectados.append({
                            "Horizonte":    etiqueta,
                            "DTE":          f"{dias_vencimiento}d",
                            "Tipo":         tipo,
                            "Strike":       f"USD {fila['strike']:.2f}",
                            "OI":           int(fila['openInterest']),
                            "Valor Total":  f"USD {fila['Millones_Apostados']:.2f} M",
                        })

        # Put/Call Ratio global
        pcr_global = vol_p_total / vol_c_total if vol_c_total > 0 else 1.0

        df_final = pd.DataFrame(muros_detectados)

        # Análisis IA (multi-proveedor: Claude → Gemini → local)
        tabla_str = df_final.to_string(index=False) if not df_final.empty else "Sin muros detectados."
        reporte_ia = analizar_muros_con_ia(
            ticker_symbol, precio_spot, tabla_str, pcr_global
        )

        return df_final, reporte_ia, fig_visual

    except Exception as e:
        return pd.DataFrame(), f"❌ Error crítico en el escáner de derivados: {e}", None
