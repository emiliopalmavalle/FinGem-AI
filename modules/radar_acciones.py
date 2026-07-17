"""
radar_acciones.py — Motor de escaneo dual (Value + Momentum)
=============================================================

Escanea universos de Wall Street en paralelo (ThreadPoolExecutor)
y clasifica oportunidades en dos motores:
  - Value:    acciones con descuento fuerte desde máximo 52 semanas
  - Momentum: fuerza de precio semanal confirmada por volumen relativo

Estrategia de llamadas a Yahoo Finance (minimiza latencia):
  history() siempre → fast_info (ligero) → .info completo solo si
  el activo ya pasó un filtro técnico.
"""

import yfinance as yf
import pandas as pd
import warnings
import concurrent.futures
from typing import Optional

warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ CONFIGURACIÓN DEL MOTOR
# ==========================================
MAX_WORKERS   = 10   # Hilos paralelos
MIN_VELAS     = 20   # Mínimo de velas para análisis válido
PERIODO_HIST  = "2mo"

# Umbrales de señales
UMBRAL_CAIDA_VALUE    = -15.0   # % de caída desde máximo 52 semanas
UMBRAL_PE_MAX         = 30.0
UMBRAL_CAMBIO_SEMANAL = 1.0     # % mínimo de subida semanal
UMBRAL_RVOL           = 1.2     # Volumen relativo mínimo


def _obtener_info_fundamental(empresa: yf.Ticker) -> dict:
    """Obtiene datos base usando fast_info (llamada HTTP ligera).

    Atributos reales de yfinance FastInfo: year_high (máximo 52 semanas)
    y two_hundred_day_average (SMA 200). Nota: fifty_two_week_high y
    short_name NO existen en fast_info — el nombre sale de .info solo
    para candidatos value (llamada pesada diferida).

    Args:
        empresa: objeto yf.Ticker.

    Returns:
        dict con tipo, max_52 (máximo 52 semanas real) y sma200.
    """
    try:
        fi = empresa.fast_info  # ← Más rápido: evita parsear ~120 campos
        tipo   = getattr(fi, 'quote_type', 'EQUITY')
        max_52 = getattr(fi, 'year_high', None)
        sma200 = getattr(fi, 'two_hundred_day_average', None)
        return {
            'tipo':   tipo if tipo else 'EQUITY',
            'max_52': max_52,
            'sma200': sma200,
        }
    except Exception:
        return {'tipo': 'EQUITY', 'max_52': None, 'sma200': None}


def _obtener_fundamentales_completos(empresa: yf.Ticker) -> dict:
    """Fallback a .info solo cuando el activo ya pasó el filtro técnico.

    Llamada más pesada — se ejecuta en el ~20-30% de los casos.

    Args:
        empresa: objeto yf.Ticker con sesión inyectada.

    Returns:
        dict con pe_ratio, eps y nombre.
    """
    try:
        info = empresa.info
        return {
            'pe_ratio': info.get('trailingPE', 0) or 0,
            'eps':      info.get('trailingEps', 0) or 0,
            'nombre':   (info.get('shortName', '')[:20] or None),
        }
    except Exception:
        return {'pe_ratio': 0, 'eps': 0, 'nombre': None}


def procesar_un_ticker(ticker: str) -> Optional[dict]:
    """Analiza un solo activo. Diseñado para ejecución en hilos paralelos.

    Estrategia de llamadas API (reduce latencia total ~40%):
      1. history() — siempre necesario
      2. fast_info — ligero, obtiene tipo y max_52w
      3. .info completo — SOLO si el activo pasó el filtro técnico previo

    Args:
        ticker: símbolo del activo a analizar.

    Returns:
        dict con resultados value/momentum o None si no cumple filtros.
    """
    try:
        empresa = yf.Ticker(ticker)
        hist = empresa.history(period=PERIODO_HIST)

        if hist.empty or len(hist) < MIN_VELAS:
            return None

        # --- Cálculos técnicos base (sin API extra) ---
        precio_actual   = hist['Close'].iloc[-1]
        cambio_semanal  = ((precio_actual - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6]) * 100
        vol_promedio    = hist['Volume'].iloc[-21:-1].mean()

        # RVOL robusto a sesión en curso: el volumen de "hoy" está
        # incompleto si el mercado sigue abierto, lo que subestima el RVOL
        # y descarta momentum reales. Tomamos el mayor entre la vela de hoy
        # y la última vela completa (ayer).
        if vol_promedio > 0:
            rvol_hoy  = hist['Volume'].iloc[-1] / vol_promedio
            rvol_ayer = hist['Volume'].iloc[-2] / vol_promedio if len(hist) >= 2 else 0.0
            rvol = max(rvol_hoy, rvol_ayer)
        else:
            rvol = 0.0

        # --- Llamada ligera (fast_info) ---
        info_base = _obtener_info_fundamental(empresa)
        tipo_activo = info_base['tipo']

        # Máximo 52 semanas real (year_high); fallback al máximo del
        # historial de 2 meses solo si fast_info falla
        max_52w = info_base.get('max_52') or hist['Close'].max()
        caida_pct = ((precio_actual - max_52w) / max_52w) * 100

        # Posición vs SMA 200: contexto anti-cuchillo (¿ya se estabilizó?)
        sma200 = info_base.get('sma200') or 0
        vs_sma200_pct = ((precio_actual - sma200) / sma200) * 100 if sma200 > 0 else None

        # --- Filtros técnicos previos ---
        candidato_value    = (tipo_activo == 'EQUITY' and caida_pct <= UMBRAL_CAIDA_VALUE)
        candidato_momentum = (cambio_semanal > UMBRAL_CAMBIO_SEMANAL and rvol > UMBRAL_RVOL)


        if not candidato_value and not candidato_momentum:
            return None  # Descartar sin llamar a .info completo

        # --- Solo aquí hacemos la llamada pesada ---
        # El activo ya pasó al menos un filtro, vale la pena investigarlo
        nombre = info_base.get('nombre') or ticker
        pe_ratio, eps = 0, 0

        if candidato_value:
            fundamentales = _obtener_fundamentales_completos(empresa)
            pe_ratio = fundamentales['pe_ratio']
            eps      = fundamentales['eps']
            nombre   = fundamentales.get('nombre') or nombre

        resultado = {
            "ticker": ticker,
            "tipo":   tipo_activo,
            "value":  None,
            "momentum": None,
        }

        # Formato de la posición vs SMA200 (contexto de estabilización)
        if vs_sma200_pct is None:
            sma200_txt = "N/A"
        elif vs_sma200_pct >= 0:
            sma200_txt = f"+{vs_sma200_pct:.1f}% ✅"   # sobre SMA200: estabilizada
        else:
            sma200_txt = f"{vs_sma200_pct:.1f}% ⚠️"    # bajo SMA200: aún cayendo

        # Motor 1: VALUE (solo acciones con descuento real y fundamentales sanos)
        if candidato_value and 0 < pe_ratio < UMBRAL_PE_MAX and eps > 0:
            # Score compuesto: profundidad del descuento + P/E barato +
            # bono si el precio ya recuperó la SMA200 (más caída NO siempre
            # es mejor: un cuchillo cayendo puntúa menos que uno estabilizado)
            score = (
                min(abs(caida_pct), 60) * 0.6          # descuento (tope 60%)
                + max(0.0, UMBRAL_PE_MAX - pe_ratio)   # qué tan barato es el P/E
                + (15 if (vs_sma200_pct or -1) >= 0 else 0)  # estabilización
            )
            resultado["value"] = {
                "Ticker":    ticker,
                "Nombre":    nombre,
                "Precio":    f"${precio_actual:.2f}",
                "Score_Raw": score,
                "Score":     int(score),
                "Caida_Raw": caida_pct,
                "P/E":       round(pe_ratio, 2),
                "EPS":       f"${eps:.2f}",
                "vs SMA200": sma200_txt,
            }

        # Motor 2: MOMENTUM (fuerza de precio + confirmación de volumen)
        if candidato_momentum:
            resultado["momentum"] = {
                "Ticker":        ticker,
                "Nombre":        nombre,
                "Fuerza_Raw":    cambio_semanal,
                "Subida Semanal": f"+{cambio_semanal:.2f}% 🚀",
                "RVOL":          f"{rvol:.1f}x Vol 🐋",
                "vs SMA200":     sma200_txt,
            }

        # Si ningún motor produjo resultado útil, descartamos
        if resultado["value"] is None and resultado["momentum"] is None:
            return None

        return resultado

    except Exception:
        return None


def buscar_joyas_ocultas(max_resultados: int = 25) -> list[str]:
    """Búsqueda profunda de "joyas ocultas" con el screener nativo de Yahoo.

    Reemplaza el scraping de Finviz (su tabla ahora se renderiza con
    JavaScript y el HTML ya no trae tickers). Mismos filtros Quant:
    P/E < 20, crecimiento EPS positivo, volumen > 500K, precio > USD 5
    y solo NYSE/NASDAQ (evita penny stocks OTC).

    Returns:
        Lista de símbolos ordenados por volumen (vacía si Yahoo falla).
    """
    try:
        consulta = yf.EquityQuery('and', [
            yf.EquityQuery('lt', ['peratio.lasttwelvemonths', 20]),
            yf.EquityQuery('gt', ['epsgrowth.lasttwelvemonths', 0]),
            yf.EquityQuery('gt', ['dayvolume', 500_000]),
            yf.EquityQuery('gt', ['intradayprice', 5]),
            yf.EquityQuery('is-in', ['exchange', 'NMS', 'NYQ']),
        ])
        respuesta = yf.screen(
            consulta, sortField='dayvolume', sortAsc=False, size=max_resultados
        )
        return [q['symbol'] for q in respuesta.get('quotes', []) if q.get('symbol')]
    except Exception:
        return []


def escaneo_institucional_dual(
    lista_tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Motor principal multihilo.

    Lanza MAX_WORKERS hilos en paralelo para maximizar throughput
    limitado por latencia de red (I/O-bound).

    Args:
        lista_tickers: lista de símbolos a escanear.

    Returns:
        Tupla de tres DataFrames:
          - df_val:      Ángeles Caídos (acciones value en descuento)
          - df_mom_acc:  Momentum Acciones
          - df_mom_etf:  Momentum ETFs
    """
    angeles_caidos    = []
    despegues_acciones = []
    despegues_etfs    = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        resultados = list(executor.map(procesar_un_ticker, lista_tickers))

    for res in resultados:
        if res is None:
            continue
        if res["value"]:
            angeles_caidos.append(res["value"])
        if res["momentum"]:
            if res["tipo"] == 'ETF':
                despegues_etfs.append(res["momentum"])
            else:
                despegues_acciones.append(res["momentum"])

    # --- Construcción de DataFrames de salida ---
    def _construir_df_value(datos: list) -> pd.DataFrame:
        if not datos:
            return pd.DataFrame()
        # Ranking por score compuesto (descuento + P/E + estabilización),
        # no por caída pura: más caída no siempre es mejor oportunidad
        df = pd.DataFrame(datos).sort_values("Score_Raw", ascending=False).head(10)
        df['Descuento'] = df['Caida_Raw'].apply(lambda x: f"{x:.2f}% 🩸")
        columnas = ['Ticker', 'Nombre', 'Precio', 'Score', 'Descuento', 'P/E', 'EPS', 'vs SMA200']
        return df[columnas]

    def _construir_df_momentum(datos: list) -> pd.DataFrame:
        if not datos:
            return pd.DataFrame()
        df = pd.DataFrame(datos).sort_values("Fuerza_Raw", ascending=False).head(10)
        return df.drop(columns=['Fuerza_Raw'])

    df_val      = _construir_df_value(angeles_caidos)
    df_mom_acc  = _construir_df_momentum(despegues_acciones)
    df_mom_etf  = _construir_df_momentum(despegues_etfs)


    return df_val, df_mom_acc, df_mom_etf
