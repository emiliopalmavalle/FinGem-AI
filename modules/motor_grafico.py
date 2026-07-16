import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# 🗺️ Mapa del Ciclo Halving BTC (fechas históricas + proyección del ciclo actual).
# Se dibuja solo en gráficos semanales de cripto.
CICLOS_HALVING = [
    {"halving": "2020-05-11", "start": "2021-02-15", "end": "2021-11-01", "dca": "2022-12-12"},
    {"halving": "2024-04-19", "start": "2025-01-24", "end": "2025-10-10", "dca": "2026-11-20"},
]
_COLORES_CICLO = {"halving": "#FF9800", "start": "#089981", "end": "#F23645", "dca": "#FFD700"}


def _dibujar_ciclo_halving(fig):
    """Dibuja las líneas verticales del ciclo halving en la fila de precio.

    Usa timestamps en milisegundos: add_vline con datetime directo
    tiene inconsistencias conocidas en ejes de velas de Plotly.
    """
    for ciclo in CICLOS_HALVING:
        for evento, fecha in ciclo.items():
            x_ms = pd.to_datetime(fecha).timestamp() * 1000
            fig.add_vline(
                x=x_ms, line_width=2, opacity=0.7,
                line_dash="dot" if evento == "halving" else "solid",
                line_color=_COLORES_CICLO[evento],
                row=1, col=1,
            )


def construir_grafico_tecnico(hist, ha_df, ema_200, temporalidad, tipo_mercado, toggles):
    show_emas = toggles.get("EMAs", True)
    show_utbot = toggles.get("UT_Bot", False)
    show_smi = toggles.get("SMI", False)

    # Si SMI está activo, necesitamos 3 filas en lugar de 2
    filas = 3 if show_smi else 2
    alturas = [0.6, 0.2, 0.2] if show_smi else [0.7, 0.3]
    specs = [[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]] if show_smi else [[{"secondary_y": False}], [{"secondary_y": True}]]

    fig = make_subplots(rows=filas, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=alturas, specs=specs)

    # Velas Principales
    fig.add_trace(go.Candlestick(
        x=ha_df.index, open=ha_df['HA_Open'], high=ha_df['HA_High'], 
        low=ha_df['HA_Low'], close=ha_df['HA_Close'], 
        name='Heikin Ashi', increasing_line_color='#089981', decreasing_line_color='#F23645'
    ), row=1, col=1)

    # 🎚️ TOGGLE: EMAs
    if show_emas:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA_10'], mode='lines', line=dict(color='#2962FF', width=1.5), name='EMA 10'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA_55'], mode='lines', line=dict(color='#FF6D00', width=2), name='EMA 55'), row=1, col=1)
        if ema_200 > 0: fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA_200'], mode='lines', line=dict(color='white', width=2), name='EMA 200'), row=1, col=1)

   # 🎚️ TOGGLE: UT Bot Alerts (Señales de Compra/Venta)
    if show_utbot:
        buys = hist[hist['Buy_Signal'] == True]
        sells = hist[hist['Sell_Signal'] == True]
        
        # Etiquetas BUY verdes
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys['Low'] * 0.95, mode='markers+text', 
            marker=dict(symbol='triangle-up', color='#00FF00', size=14), 
            text="BUY", textposition="bottom center", textfont=dict(color="#00FF00", size=11, weight="bold"), name='BUY'
        ), row=1, col=1)
        
        # Etiquetas SELL rojas
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells['High'] * 1.05, mode='markers+text', 
            marker=dict(symbol='triangle-down', color='#FF0000', size=14), 
            text="SELL", textposition="top center", textfont=dict(color="#FF0000", size=11, weight="bold"), name='SELL'
        ), row=1, col=1)

    # Fila 2: Monitor y ADX (Base)
    colores_monitor = ['#089981' if (val >= 0 and val > hist['Monitor'].iloc[i-1]) else '#006400' if val >= 0 else '#F23645' if (val < 0 and val < hist['Monitor'].iloc[i-1]) else '#8B0000' for i, val in enumerate(hist['Monitor'])]
    colores_monitor[0] = 'gray'
    
    fig.add_trace(go.Bar(x=hist.index, y=hist['Monitor'], marker_color=colores_monitor, name='Monitor', opacity=0.8), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['ADX'], mode='lines', line=dict(color='white', width=1.5), name='ADX'), row=2, col=1, secondary_y=True)
    fig.add_hline(y=23, line_dash="dot", line_color="gray", row=2, col=1, secondary_y=True)

    # 🎚️ TOGGLE: Fila 3 SMI
    if show_smi:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMI'], mode='lines', line=dict(color='#2962FF', width=2), name='SMI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMI_Signal'], mode='lines', line=dict(color='#F23645', width=1.5), name='SMI Signal'), row=3, col=1)
        fig.add_hline(y=40, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5)
        fig.add_hline(y=-40, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)

    # 🗺️ Ciclo Halving: líneas verticales solo en semanal + cripto
    mostrar_ciclo = temporalidad == "Semanal" and "Cripto" in tipo_mercado
    if mostrar_ciclo:
        _dibujar_ciclo_halving(fig)

    # Limites y Layout
    if mostrar_ciclo:
        # 🔮 Modo Oráculo: 7 años hacia atrás (cubre ciclo 2020) y
        # 10 meses hacia el futuro para ver el próximo DCA Start
        fecha_inicio = max(hist.index[-1] - pd.DateOffset(years=7), hist.index[0])
        fecha_fin    = hist.index[-1] + pd.DateOffset(months=10)
    else:
        fecha_inicio, fecha_fin = hist.index[max(0, len(hist)-100)], hist.index[-1] + pd.Timedelta(days=5) # Zoom inteligente automático
    
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#131722", plot_bgcolor="#131722", 
        xaxis_rangeslider_visible=False, height=800 if show_smi else 650, margin=dict(l=10, r=10, t=30, b=10), 
        showlegend=False, dragmode='pan', modebar_add=['drawline', 'drawrect', 'eraseshape']
    )
   # Limpiar líneas de fondo blancas en TODOS los sub-gráficos
    fig.update_xaxes(range=[fecha_inicio, fecha_fin], showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False) # Aplica a todas las filas automáticamente
    fig.update_yaxes(side="right", row=1, col=1)     # Mantiene el precio a la derecha
    
    return fig