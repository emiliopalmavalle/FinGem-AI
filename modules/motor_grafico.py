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

    Usa add_shape (no add_vline): en Plotly 6.x add_vline es un no-op sobre
    figuras make_subplots. La línea va de y0=0 a y1=1 en 'y domain' (todo el
    alto de la fila de precio); la fecha ISO se ubica en el eje x de tiempo.
    """
    for ciclo in CICLOS_HALVING:
        for evento, fecha in ciclo.items():
            fig.add_shape(
                type="line", x0=fecha, x1=fecha, xref="x",
                y0=0, y1=1, yref="y domain",
                line=dict(
                    color=_COLORES_CICLO[evento], width=2,
                    dash="dot" if evento == "halving" else "solid",
                ),
                opacity=0.7, row=1, col=1,
            )


def _calcular_niveles_sr(hist, lookback=6, tol_pct=0.010, max_por_lado=3, ventana=120):
    """Soportes y Resistencias estilo LuxAlgo: pivotes swing confirmados.

    Un pivote de resistencia es una vela cuyo High es el máximo de la ventana
    [i-lookback, i+lookback]; uno de soporte, cuyo Low es el mínimo. Se agrupan
    niveles cercanos (dentro de tol_pct) y se cuentan los toques: más toques =
    nivel más respetado. Devuelve los más cercanos al precio actual (hasta
    max_por_lado por lado).

    Solo mira las últimas `ventana` velas: los niveles quedan relevantes a lo
    que se ve en pantalla (el gráfico hace zoom a las ~100 más recientes), no
    a mínimos históricos lejanos que ya no operan.
    """
    if len(hist) > ventana:
        hist = hist.iloc[-ventana:]

    highs = hist['High'].values
    lows  = hist['Low'].values
    n = len(highs)
    if n < 2 * lookback + 1:
        return {'soportes': [], 'resistencias': []}

    crudos = []
    for i in range(lookback, n - lookback):
        if highs[i] == highs[i - lookback:i + lookback + 1].max():
            crudos.append(highs[i])
        if lows[i] == lows[i - lookback:i + lookback + 1].min():
            crudos.append(lows[i])

    if not crudos:
        return {'soportes': [], 'resistencias': []}

    # Agrupar niveles cercanos (clustering simple) y contar toques
    crudos.sort()
    grupos, actual = [], [crudos[0]]
    for nivel in crudos[1:]:
        if abs(nivel - actual[-1]) <= actual[-1] * tol_pct:
            actual.append(nivel)
        else:
            grupos.append(actual)
            actual = [nivel]
    grupos.append(actual)

    niveles = [(sum(g) / len(g), len(g)) for g in grupos]

    precio = hist['Close'].iloc[-1]
    resistencias = sorted([x for x in niveles if x[0] > precio], key=lambda x: x[0])[:max_por_lado]
    soportes     = sorted([x for x in niveles if x[0] <= precio], key=lambda x: -x[0])[:max_por_lado]
    return {'soportes': soportes, 'resistencias': resistencias}


def _dibujar_soportes_resistencias(fig, hist, ann_bg="rgba(19,23,34,0.65)"):
    """Dibuja las líneas horizontales de S/R en la fila de precio.

    Rojo = resistencia (sobre el precio), verde = soporte (bajo el precio).
    La opacidad crece con el número de toques: nivel más tocado = más visible.

    Usa add_shape/add_annotation (no add_hline): en Plotly 6.x add_hline es
    un no-op sobre figuras make_subplots, no dibuja nada. `ann_bg` es el fondo
    de la etiqueta (se adapta al tema claro/oscuro).
    """
    niveles = _calcular_niveles_sr(hist)

    def _nivel(precio, toques, color, etiqueta, yanchor):
        opacidad = min(0.35 + toques * 0.15, 0.95)
        fig.add_shape(
            type="line", x0=0, x1=1, xref="x domain",
            y0=precio, y1=precio, yref="y",
            line=dict(color=color, width=1.3), opacity=opacidad,
            row=1, col=1,
        )
        fig.add_annotation(
            x=1, y=precio, xref="x domain", yref="y",
            text=f"{etiqueta} {precio:,.2f} ({toques})",
            showarrow=False, xanchor="right", yanchor=yanchor,
            font=dict(color=color, size=10),
            bgcolor=ann_bg,
            row=1, col=1,
        )

    for precio, toques in niveles['resistencias']:
        _nivel(precio, toques, "#F23645", "R", "bottom")
    for precio, toques in niveles['soportes']:
        _nivel(precio, toques, "#089981", "S", "top")


def construir_grafico_tecnico(hist, ha_df, ema_200, temporalidad, tipo_mercado, toggles):
    show_emas = toggles.get("EMAs", True)
    show_sr = toggles.get("SR", False)
    show_smi = toggles.get("SMI", False)
    fondo_oscuro = toggles.get("Fondo_Oscuro", True)
    show_halving = toggles.get("Halving", True)

    # 🎨 Paleta según tema. Las líneas que estaban hardcodeadas en blanco
    # (EMA 200, ADX) se vuelven invisibles en fondo claro, así que su color
    # (y el fondo de las etiquetas S/R) dependen del tema.
    if fondo_oscuro:
        tema_template = "plotly_dark"
        color_fondo   = "#131722"
        color_linea   = "white"
        ann_bg        = "rgba(19,23,34,0.65)"
    else:
        tema_template = "plotly_white"
        color_fondo   = "#FFFFFF"
        color_linea   = "#131722"
        ann_bg        = "rgba(255,255,255,0.80)"

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
        if ema_200 > 0: fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA_200'], mode='lines', line=dict(color=color_linea, width=2), name='EMA 200'), row=1, col=1)

    # 🎚️ TOGGLE: Soportes y Resistencias (pivotes swing estilo LuxAlgo)
    if show_sr:
        _dibujar_soportes_resistencias(fig, hist, ann_bg=ann_bg)

    # Fila 2: Monitor y ADX (Base)
    colores_monitor = ['#089981' if (val >= 0 and val > hist['Monitor'].iloc[i-1]) else '#006400' if val >= 0 else '#F23645' if (val < 0 and val < hist['Monitor'].iloc[i-1]) else '#8B0000' for i, val in enumerate(hist['Monitor'])]
    colores_monitor[0] = 'gray'
    
    fig.add_trace(go.Bar(x=hist.index, y=hist['Monitor'], marker_color=colores_monitor, name='Monitor', opacity=0.8), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['ADX'], mode='lines', line=dict(color=color_linea, width=1.5), name='ADX'), row=2, col=1, secondary_y=True)
    # Umbral de tendencia ADX=23. add_shape (no add_hline, que es no-op en subplots).
    fig.add_shape(type="line", x0=0, x1=1, xref="x domain", y0=23, y1=23,
                  line=dict(color="gray", dash="dot", width=1),
                  row=2, col=1, secondary_y=True)

    # 🎚️ TOGGLE: Fila 3 SMI
    if show_smi:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMI'], mode='lines', line=dict(color='#2962FF', width=2), name='SMI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMI_Signal'], mode='lines', line=dict(color='#F23645', width=1.5), name='SMI Signal'), row=3, col=1)
        # Zonas de sobrecompra/sobreventa SMI (±40). add_shape por el no-op de add_hline.
        fig.add_shape(type="line", x0=0, x1=1, xref="x domain", y0=40, y1=40,
                      line=dict(color="red", dash="dash", width=1), opacity=0.5, row=3, col=1)
        fig.add_shape(type="line", x0=0, x1=1, xref="x domain", y0=-40, y1=-40,
                      line=dict(color="green", dash="dash", width=1), opacity=0.5, row=3, col=1)

    # 🗺️ Ciclo Halving: líneas verticales solo en semanal + cripto, y solo si
    # el usuario deja el toggle encendido (pueden ensuciar el análisis).
    mostrar_ciclo = show_halving and temporalidad == "Semanal" and "Cripto" in tipo_mercado
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
        template=tema_template, paper_bgcolor=color_fondo, plot_bgcolor=color_fondo,
        xaxis_rangeslider_visible=False, height=800 if show_smi else 650, margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False, dragmode='pan', modebar_add=['drawline', 'drawrect', 'eraseshape']
    )
   # Limpiar líneas de fondo blancas en TODOS los sub-gráficos
    fig.update_xaxes(range=[fecha_inicio, fecha_fin], showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False) # Aplica a todas las filas automáticamente
    fig.update_yaxes(side="right", row=1, col=1)     # Mantiene el precio a la derecha
    
    return fig