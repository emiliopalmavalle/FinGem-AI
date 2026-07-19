# FinGem — Mejoras al módulo "CALLs Baratos"

Instrucciones para aplicar con **Claude Code**. Cada cambio es independiente:
puedes aplicarlos por separado y probar entre uno y otro.

> **Revisión pre-aplicación aplicada.** Se descartó el pre-filtro de tickers
> "optionables" que figuraba antes como Cambio 1: el worker
> `_procesar_calls_baratos_ticker` ya sale de inmediato (`return []`) cuando un
> ticker no tiene `.options`, así que un pre-filtro solo añadía una segunda
> llamada a `.options` por ticker vivo — gasto de red contra el rate limit de
> Yahoo (nuestro punto débil en la nube) a cambio de beneficio marginal. Los
> cambios válidos son los tres siguientes, más la actualización de docstrings.

**Archivos afectados**
- `modules/radar_opciones.py` — motor de escaneo (toda la lógica real)
- `FINGEM.py` — solo la UI (sección `💸 MÓDULO: CALLS BARATOS`)

**Contexto para el agente:** el escáner ya filtra por operabilidad (bid/ask
vivos, spread ≤ 35%, OI mínimo, delta ≥ 0.10) y puntúa premiando delta ≈ 0.40.
Eso está bien y NO debe romperse. Estos cambios añaden protección de earnings,
mejoran el score con IV, y hacen el spread configurable. Mantén el estilo del
archivo (docstrings, comentarios en español, `try/except` defensivo).

**Costo de red:** el Cambio 1 (earnings) añade 1 llamada `.calendar` por ticker.
Para universos de ~20 símbolos es asumible; si algún día expandes el universo,
considera cachear o paralelizar esa lectura.

---

## Cambio 1 — Aviso de EARNINGS dentro del vencimiento (el más crítico)

**Problema:** un CALL con DTE de 30–45 días casi siempre cruza un reporte de
resultados. El *IV crush* posterior puede evaporar la prima aunque el precio
suba. El buscador ignora esto hoy, y es el mayor destructor de valor para
capital pequeño. (La app ya avisa de earnings en el análisis individual — esto
lo hace consistente.)

**Dónde:** `modules/radar_opciones.py`, función `_procesar_calls_baratos_ticker`.

### 1a. Obtener la fecha de earnings una vez por ticker

Justo después de obtener `spot` (tras el bloque
`spot = getattr(t.fast_info, "last_price", None) or 0.0` y su `if not spot`),
añade:

```python
    # Fecha del próximo reporte de resultados (para marcar riesgo de IV crush).
    # Una sola lectura por ticker; si falla, no bloquea el escaneo.
    fecha_earnings = None
    try:
        cal = t.calendar
        fechas_e = (cal or {}).get("Earnings Date", []) if isinstance(cal, dict) else []
        if fechas_e:
            fecha_earnings = pd.Timestamp(fechas_e[0]).to_pydatetime().replace(tzinfo=None)
    except Exception:
        fecha_earnings = None
```

### 1b. Marcar cada contrato cuyo vencimiento cae DESPUÉS del reporte

Dentro del bucle `for _, fila in c.iterrows():`, después de calcular
`breakeven` y `pct_be` y **antes** del `contratos.append({...})`, añade:

```python
                # Riesgo de earnings: el contrato cruza el reporte si la fecha
                # de resultados es futura Y cae en/antes del vencimiento.
                # El guardia `hoy <=` evita falsos positivos cuando Yahoo
                # devuelve una fecha de earnings ya pasada (aún sin actualizar).
                venc_dt = datetime.strptime(fecha, "%Y-%m-%d")
                cruza_earnings = bool(fecha_earnings and hoy <= fecha_earnings <= venc_dt)
```

Luego añade la clave `"cruza_earnings": cruza_earnings,` dentro del dict
`contratos.append({...})`.

### 1c. Exponer la columna en el DataFrame de salida

**Dónde:** `escanear_calls_baratos`, en el `pd.DataFrame({...})` final que arma
las columnas para la UI. Añade una columna (por ejemplo tras `"IV"`):

```python
        "⚠️ Earnings": df["cruza_earnings"].map(lambda x: "SÍ — IV crush" if x else "—"),
```

### 1d. (Opcional) Nota en la UI

**Dónde:** `FINGEM.py`, en el `st.caption(...)` que explica cómo leer la tabla
(sección CALLs Baratos). Añade al final del texto:

```
⚠️ Earnings = 'SÍ' significa que el contrato vence DESPUÉS del próximo reporte:
la prima puede desplomarse por IV crush aunque el precio suba. Trátalo como
riesgo, no como oportunidad.
```

---

## Cambio 2 — Penalizar IV alta en el Score (mejora de calidad)

**Problema:** el score premia delta, spread, OI y cercanía al breakeven, pero
ignora el NIVEL de IV. Dos contratos idénticos salvo la IV (40% vs 90%) reciben
el mismo score, cuando el de IV 90% es peor compra (pagas volatilidad inflada).

**Dónde:** `modules/radar_opciones.py`, `_procesar_calls_baratos_ticker`, en el
bloque `# ── Score de viabilidad 0-100`.

**Qué hacer:** añadir un sub-score de IV y reponderar. Reemplaza el bloque de
score actual por este (mismos componentes + IV, pesos re-normalizados):

```python
                # ── Score de viabilidad 0-100
                s_delta  = max(0, 100 - abs(delta - 0.40) * 250)       # sweet spot ~0.40
                sp       = fila["spread_pct"]
                s_spread = 90 if sp <= 10 else 60 if sp <= 20 else 35 if sp <= 30 else 10
                oi       = fila["openInterest"]
                s_oi     = 90 if oi >= 1000 else 70 if oi >= 300 else 50 if oi >= 100 else 30
                s_be     = 90 if pct_be <= 3 else 70 if pct_be <= 6 else 45 if pct_be <= 10 else 20
                # IV absoluta como proxy de "caro/barato en volatilidad":
                # sin IV Rank por ticker aquí, se usa un umbral absoluto simple.
                iv_p     = iv * 100
                s_iv     = 90 if iv_p <= 40 else 65 if iv_p <= 60 else 40 if iv_p <= 90 else 20
                score    = int(s_delta * 0.25 + s_spread * 0.15 + s_oi * 0.15
                               + s_be * 0.25 + s_iv * 0.20)
```

> Nota: lo ideal sería usar IV *Rank* (percentil histórico) en vez de IV
> absoluta, que ya existe en la Capa 2 de este módulo. Si quieres precisión,
> un segundo paso sería reutilizar ese cálculo aquí. El umbral absoluto es una
> aproximación suficiente para un primer filtro.

---

## Cambio 3 — Spread default más estricto y tope por ticker (refinamiento UI)

**Problema:** spread ≤ 35% es generoso para capital pequeño (35% de fuga al
entrar/salir es enorme). Y `.head(top_n)` global puede llenar la tabla con un
solo ticker muy líquido.

### 3a. Spread como slider en la UI

**Dónde:** `FINGEM.py`, sidebar de la sección CALLs Baratos, junto a los otros
sliders (`presupuesto`, `dte_rango`, `oi_minimo`). Añade:

```python
    spread_max = st.sidebar.slider(
        "📏 Spread bid/ask máximo (%):",
        min_value=10, max_value=40, value=25, step=5,
        help="Diferencia entre compra y venta. Menor = menos fuga al entrar y salir. "
             "Para capital pequeño, 25% o menos es lo sano.",
    )
```

Y pásalo a la llamada `escanear_calls_baratos(...)` como nuevo argumento
`spread_max=spread_max`.

### 3b. Aceptar el parámetro en el motor

**Dónde:** `modules/radar_opciones.py`.

- En la firma de `escanear_calls_baratos`, añade `spread_max: float = 25.0,`.
- Propágalo en `args`: `(sym, presupuesto_max, dte_min, dte_max, oi_min, spread_max)`.
- En `_procesar_calls_baratos_ticker`, desempaqueta el nuevo valor y usa
  `spread_max` en lugar del literal `35` en el filtro:

```python
    ticker, presupuesto, dte_min, dte_max, oi_min, spread_max = args
    ...
            c = c[
                (c["costo"] <= presupuesto) &
                (c["openInterest"] >= oi_min) &
                (c["spread_pct"] <= spread_max)
            ]
```

### 3c. (Opcional) Máximo N contratos por ticker

**Dónde:** `escanear_calls_baratos`, antes del `.head(top_n)` global. Para
diversificar la tabla en lugar de concentrarla en 1–2 símbolos:

```python
    df = pd.DataFrame(todos).sort_values("score", ascending=False)
    df = df.groupby("ticker", group_keys=False).head(3)   # máx 3 por ticker
    df = df.head(top_n)
```

---

## Cambio 4 — Actualizar docstrings desfasados (obligatorio al aplicar 2 y 3)

Aplicar los cambios de score y de spread deja dos docstrings mintiendo. Hay que
actualizarlos en el mismo commit:

- **`escanear_calls_baratos`** (docstring): la línea que describe el score
  ("pondera: cercanía a delta 0.40 (30%)... spread (20%) y Open Interest (20%)")
  debe reflejar los pesos nuevos: delta 25%, breakeven 25%, IV 20%, spread 15%,
  OI 15%. Y si mencionaba "spread ≤ 35%", cambiarlo a "spread ≤ `spread_max`
  (configurable, default 25%)".
- **`_procesar_calls_baratos_ticker`** (sección `Args:`): documentar el nuevo
  parámetro `spread_max` que ahora llega dentro de la tupla `args`.

No es cosmético: estos docstrings son lo que lee el próximo que toque el módulo
(humano o agente), y un peso equivocado ahí propaga errores.

---

## Verificación tras aplicar

1. `python -m py_compile FINGEM.py modules/radar_opciones.py` — debe pasar.
2. Correr la app y escanear el universo "Acciones baratas líquidas": la tabla
   debe mostrar la columna **⚠️ Earnings**. (Nota: un ticker sin cadena de
   opciones simplemente no aparece — el worker ya lo descarta solo, no hace
   falta pre-filtro.)
3. Confirmar que un contrato con IV muy alta baja de posición en el ranking
   frente a uno equivalente con IV baja.
4. Bajar el slider de spread a 15% y confirmar que la tabla se reduce.

## Qué NO tocar

- Los filtros de operabilidad existentes (bid/ask vivos, delta ≥ 0.10).
- El sweet spot de delta ≈ 0.40 en el score.
- `_delta_call_bs`: la delta es una aproximación Black-Scholes intencional
  (Yahoo no publica greeks). Si en el futuro quieres precisión para pagadores
  de dividendos, restar el dividend yield en `d1` — pero es refinamiento, no
  bug.

