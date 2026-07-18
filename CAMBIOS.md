# 📋 CAMBIOS.md — Control de Cambios · FinGem Terminal

Registro de versiones del proyecto. Cada entrada referencia su commit en Git
(`git show <hash>` muestra el detalle completo). Orden: más reciente arriba.

---

## v4.4 — 2026-07-18 · Auditoría externa aplicada + parches del auditor

| Commit | Cambio |
|---|---|
| _(actual)_ | **Parches finales del auditor (veredicto 9/10):** el JSON del plan separa `sesgo` (mercado) de `direccion` (operación largo/corto/fuera) — un largo táctico con sesgo bajista ya no es falso positivo, se marca como contra-tendencia informativa. Chequeo de ATR solo en 1H/4H/Diario. Tarjeta de métricas muestra "Operación: Largo (sesgo mercado: bajista)". **Fundamentales de la tabla del auditor en las salidas:** P/E, P/S (ventas), ROE (eficiencia), Deuda/EBITDA (riesgo) y FCF — como fila de métricas visible y dentro del prompt. Se crea este CAMBIOS.md. |
| `31764cc` | **Auditoría externa completa (9 hallazgos corregidos):** numeración del prompt (plan en punto 5), horizonte parametrizado por temporalidad, stop 1.5×ATR solo en marcos cortos, vencimiento de opciones según temporalidad (45d Semanal / 90d Mensual), "volumen inusual" con dirección desconocida (honestidad en 3 módulos), etiqueta ADX sin ambigüedad, **nuevo `validador_plan.py`** (la IA emite JSON, Python verifica coherencia y recalcula R/B), fundamentales + earnings próximos al prompt, fecha/timestamp de generación, caché por día, FVG solo no-mitigados, rate-limit del login, `$` seguro con regex. |

## v4.3 — 2026-07-17/18 · Análisis Top-Down y calidad de reportes

| Commit | Cambio |
|---|---|
| `4e9304f` | **Análisis Top-Down completo:** nuevo `contexto_macro.py` — VIX con régimen, tasas 10a/3m con curva de rendimientos (lectura de ciclo), petróleo/oro/dólar como proxies de inflación. Reporte de bolsa pasa a 5 puntos: Macro → Sentimiento → Cualitativo (moat/gestión/regulatorio) → Técnico → Veredicto. |
| `93dde81` | Etiqueta de autoría en cada análisis ("Análisis generado por: Claude Opus 4.8...") y spinner con el proveedor real. |
| `0ee04b0` | Conclusión final obligatoria anclada a la temporalidad solicitada (dirección + convicción, nivel que confirma/invalida, acción concreta). |
| `8a773d3` | Contexto operable para la IA: ATR(14), RSI(14), RVOL, estructura de precio (máx/mín 5-20 velas), divergencia bajista nueva, niveles de opciones automáticos para tickers opcionables. |

## v4.2 — 2026-07-17 · Módulos alineados a day trading 1-3 días

| Commit | Cambio |
|---|---|
| `6abb2c4` | BMV ampliada: lista IPC de 36 emisoras + screener dinámico mexicano (`buscar_universo_bmv`). |
| `0fc3ab1` | Escáner Global: revisión de universos (Top50/ETFs/BMV/personalizada), moneda MXN correcta, nombres en momentum, ETFs a 30. |
| `fa2a4a2` | Radar de Opciones: columnas Soporte/Resistencia/Max Pain/Vol Inusual por ticker, bono de volumen inusual al score, IA con niveles operables. |
| `bd93425` | Flujo de Opciones para 1-3 días: salta 0DTE, Put/Call Wall + Max Pain + PCR del vencimiento, detección de volumen inusual (vol > OI), muros por OI cerca del spot. |
| `0683605` | Escáner Global: **fix crítico** del máximo 52 semanas (`year_high`; antes usaba máx de 2 meses y perdía NKE -44%), score compuesto de valor, RVOL robusto, columna vs SMA200. Título "Oportunidades de Valor". |
| `2e591d8` | Joyas Ocultas: Finviz (muerto para scraping) reemplazado por screener nativo de Yahoo (`yf.screen`). |

## v4.1 — 2026-07-16 · Infraestructura: IA multi-proveedor y seguridad

| Commit | Cambio |
|---|---|
| `4495958` | Manejo amable de errores de Yahoo (símbolo inválido / rate limit) con mensajes y formato de símbolos. |
| `48197b5` | **IA multi-proveedor** (`ai_client.py`): Claude Opus 4.8 principal → Gemini respaldo → reporte local; caché global 24h. **Login** (`auth.py`): 3 usuarios con hash SHA-256 en secrets. |
| `cb54592` | Ciclo del halving restaurado (banner + líneas 2020/2024 + modo oráculo), radar con umbral ajustable + señales LEAN, nuevo módulo CALLs Baratos (capital pequeño). |

## v4.0 — 2026-07-15 · Arquitectura modular

| Commit | Cambio |
|---|---|
| `6c907b9` | Migración de monolito a arquitectura modular (`modules/`), corrección de APIs (yfinance 1.x, Etherscan V2, noticias anidadas), secrets fuera del repo, requirements pineados. |
| `78670c4`…`9aff86e` | Versiones históricas del monolito FINGEM 1.0 (pre-modular). |

---

### Convención para nuevas entradas
Al hacer push de un cambio relevante, añadir una fila arriba con: hash corto,
qué cambió y por qué importa. Los detalles técnicos completos viven en el
mensaje del commit (`git log`), aquí va el resumen ejecutivo.
