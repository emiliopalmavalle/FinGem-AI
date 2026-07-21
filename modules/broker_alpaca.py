"""
broker_alpaca.py — Ejecución en cuenta PAPEL de Alpaca (simulada)
==================================================================

Cierra el ciclo de la terminal: descubrir → discriminar → EJECUTAR Y MEDIR.
Sin un registro de lo que pasó después, no hay forma de distinguir un
sistema que funciona de una racha con suerte.

⚠️ SEGURIDAD — este módulo SOLO opera contra la cuenta de PAPEL.
La URL está fijada en código (_BASE_PAPEL) y no se lee de configuración:
ninguna combinación de credenciales o parámetros puede hacer que envíe
una orden real. Para operar en real haría falta editar este archivo a
conciencia, que es exactamente la fricción que queremos.

Límites reales de Alpaca en OPCIONES (verificado en su documentación):
  · Tipos permitidos: market, limit, stop, stop_limit
  · NO existen órdenes bracket / OCO / OTO ni trailing stops
  · time_in_force solo 'day' o 'gtc'
Consecuencia de diseño: en opciones se envía SOLO la entrada, y el stop
y el objetivo quedan registrados en el plan para vigilarlos desde la
terminal. En ACCIONES sí se usa bracket nativo (entrada+stop+TP atómico).
"""

import requests
import pandas as pd
from typing import Optional

# ==========================================
# 🔒 CONFIGURACIÓN (solo papel)
# ==========================================
_BASE_PAPEL = "https://paper-api.alpaca.markets/v2"   # ⚠️ NO cambiar a api.alpaca.markets
_TIMEOUT    = 20

_credenciales = {"key": "", "secret": ""}


def configurar_broker(key_id: str, secret: str) -> None:
    """Inyecta credenciales desde FINGEM (ningún módulo lee st.secrets)."""
    _credenciales["key"]    = key_id or ""
    _credenciales["secret"] = secret or ""


def broker_activo() -> bool:
    """True si hay credenciales cargadas."""
    return bool(_credenciales["key"] and _credenciales["secret"])


def _cabeceras() -> dict:
    return {
        "APCA-API-KEY-ID":     _credenciales["key"],
        "APCA-API-SECRET-KEY": _credenciales["secret"],
        "Content-Type":        "application/json",
    }


def _pedir(metodo: str, ruta: str, **kwargs) -> tuple[bool, object]:
    """Llamada HTTP con manejo uniforme de errores.

    Returns:
        (exito, datos) — datos es el JSON, o el mensaje de error si falló.
    """
    if not broker_activo():
        return False, "No hay credenciales de Alpaca configuradas."
    try:
        resp = requests.request(
            metodo, f"{_BASE_PAPEL}{ruta}",
            headers=_cabeceras(), timeout=_TIMEOUT, **kwargs
        )
        if resp.status_code >= 400:
            try:
                msg = resp.json().get("message", resp.text)
            except Exception:
                msg = resp.text
            return False, f"[{resp.status_code}] {msg}"
        return True, (resp.json() if resp.text else {})
    except Exception as e:
        return False, f"Error de conexión: {e}"


# ==========================================
# 📊 LECTURA: ESTADO DE LA CUENTA
# ==========================================
def estado_cuenta() -> dict:
    """Resumen de la cuenta de papel.

    Returns:
        dict con equity, efectivo, poder de compra, P&L del día y estado;
        {'ok': False, 'error': ...} si falla.
    """
    ok, d = _pedir("GET", "/account")
    if not ok:
        return {"ok": False, "error": d}

    equity     = float(d.get("equity") or 0)
    last_eq    = float(d.get("last_equity") or 0)
    pl_dia     = equity - last_eq
    pl_dia_pct = (pl_dia / last_eq * 100) if last_eq else 0.0

    return {
        "ok":             True,
        "estado":         d.get("status", "?"),
        "equity":         equity,
        "efectivo":       float(d.get("cash") or 0),
        "poder_compra":   float(d.get("buying_power") or 0),
        "poder_opciones": float(d.get("options_buying_power") or 0),
        "nivel_opciones": d.get("options_approved_level", 0),
        "pl_dia":         pl_dia,
        "pl_dia_pct":     pl_dia_pct,
    }


def mercado_abierto() -> dict:
    """Estado del mercado: si está abierto y cuándo abre/cierra."""
    ok, d = _pedir("GET", "/clock")
    if not ok:
        return {"abierto": False, "error": d}
    return {
        "abierto":       bool(d.get("is_open")),
        "proxima_apertura": d.get("next_open", ""),
        "proximo_cierre":   d.get("next_close", ""),
    }


def posiciones_abiertas() -> pd.DataFrame:
    """Posiciones vivas con su P&L no realizado.

    Returns:
        DataFrame ordenado por P&L (peor primero, para ver riesgos antes).
    """
    ok, d = _pedir("GET", "/positions")
    if not ok or not isinstance(d, list) or not d:
        return pd.DataFrame()

    filas = []
    for p in d:
        cant    = float(p.get("qty") or 0)
        entrada = float(p.get("avg_entry_price") or 0)
        actual  = float(p.get("current_price") or 0)
        pl      = float(p.get("unrealized_pl") or 0)
        pl_pct  = float(p.get("unrealized_plpc") or 0) * 100
        es_op   = p.get("asset_class") == "us_option"
        filas.append({
            "Símbolo":    p.get("symbol", ""),
            "Tipo":       "Opción" if es_op else "Acción",
            "Cantidad":   cant,
            "Entrada":    f"USD {entrada:,.2f}",
            "Actual":     f"USD {actual:,.2f}",
            "Valor":      f"USD {float(p.get('market_value') or 0):,.2f}",
            "P&L":        f"USD {pl:+,.2f}",
            "P&L %":      f"{pl_pct:+.1f}%",
            "_pl_raw":    pl,
        })
    df = pd.DataFrame(filas).sort_values("_pl_raw")
    return df.drop(columns=["_pl_raw"])


def ordenes(estado: str = "all", limite: int = 25) -> pd.DataFrame:
    """Órdenes recientes (abiertas, cerradas o todas)."""
    ok, d = _pedir("GET", "/orders", params={
        "status": estado, "limit": limite, "direction": "desc", "nested": "true",
    })
    if not ok or not isinstance(d, list) or not d:
        return pd.DataFrame()

    filas = []
    for o in d:
        filas.append({
            "Fecha":     (o.get("submitted_at") or "")[:16].replace("T", " "),
            "Símbolo":   o.get("symbol", ""),
            "Lado":      "Compra" if o.get("side") == "buy" else "Venta",
            "Cantidad":  o.get("qty") or o.get("filled_qty") or "",
            "Tipo":      o.get("type", ""),
            "Estado":    o.get("status", ""),
            "Precio med.": (f"USD {float(o['filled_avg_price']):,.2f}"
                            if o.get("filled_avg_price") else "—"),
            "id":        o.get("id", ""),
        })
    return pd.DataFrame(filas)


# ==========================================
# 📏 DIMENSIONAMIENTO DE POSICIÓN
# ==========================================
def calcular_tamano(
    capital: float,
    riesgo_pct: float,
    precio_entrada: float,
    precio_stop: Optional[float] = None,
    es_opcion: bool = False,
) -> dict:
    """Cuánto comprar para arriesgar solo un % definido del capital.

    Es la decisión más determinante de la operativa y la que la terminal
    no cubría: el plan decía dónde entrar y salir, pero no cuánto.

    Lógica según instrumento:
      · OPCIÓN comprada: la pérdida máxima es la prima pagada, así que el
        riesgo por contrato es prima × 100 (el multiplicador estándar).
      · ACCIÓN: el riesgo por título es la distancia entrada-stop.

    Args:
        capital: equity de la cuenta.
        riesgo_pct: % del capital a arriesgar (1.0 = 1%).
        precio_entrada: prima de la opción o precio de la acción.
        precio_stop: solo para acciones.
        es_opcion: True si el instrumento es un contrato de opciones.

    Returns:
        dict con cantidad, riesgo en USD, costo total y un aviso si aplica.
    """
    riesgo_usd = capital * (riesgo_pct / 100.0)

    if precio_entrada <= 0:
        return {"cantidad": 0, "riesgo_usd": 0, "costo": 0,
                "aviso": "Precio de entrada inválido."}

    if es_opcion:
        riesgo_unitario = precio_entrada * 100.0   # prima × multiplicador
        costo_unitario  = riesgo_unitario
    else:
        if not precio_stop or precio_stop <= 0:
            return {"cantidad": 0, "riesgo_usd": riesgo_usd, "costo": 0,
                    "aviso": "Para acciones hace falta un stop para dimensionar."}
        riesgo_unitario = abs(precio_entrada - precio_stop)
        costo_unitario  = precio_entrada
        if riesgo_unitario <= 0:
            return {"cantidad": 0, "riesgo_usd": riesgo_usd, "costo": 0,
                    "aviso": "El stop coincide con la entrada: riesgo indefinido."}

    cantidad = int(riesgo_usd // riesgo_unitario)
    aviso = ""
    if cantidad < 1:
        aviso = (f"Con {riesgo_pct}% de riesgo (USD {riesgo_usd:,.2f}) no alcanza "
                 f"ni para 1 unidad: costaría USD {riesgo_unitario:,.2f}. "
                 f"Sube el riesgo o busca un contrato más barato.")

    return {
        "cantidad":   max(cantidad, 0),
        "riesgo_usd": riesgo_usd,
        "riesgo_real": cantidad * riesgo_unitario,
        "costo":      cantidad * costo_unitario,
        "aviso":      aviso,
    }


# ==========================================
# 🚀 ESCRITURA: ENVÍO DE ÓRDENES (SOLO PAPEL)
# ==========================================
def enviar_orden_opcion(
    simbolo_occ: str,
    cantidad: int,
    lado: str = "buy",
    tipo: str = "limit",
    precio_limite: Optional[float] = None,
    tif: str = "day",
) -> tuple[bool, str]:
    """Envía una orden de opciones a la cuenta de PAPEL.

    Alpaca no admite bracket en opciones, así que esto manda ÚNICAMENTE la
    entrada. El stop y el objetivo se vigilan desde la terminal.

    Args:
        simbolo_occ: contrato en formato OCC (ej. AAPL260724C00330000).
        cantidad: número de contratos.
        lado: 'buy' o 'sell'.
        tipo: 'market' o 'limit'.
        precio_limite: obligatorio si tipo='limit'.
        tif: 'day' o 'gtc'.

    Returns:
        (exito, mensaje descriptivo o id de la orden).
    """
    if cantidad < 1:
        return False, "La cantidad debe ser al menos 1 contrato."

    cuerpo = {
        "symbol": simbolo_occ, "qty": str(int(cantidad)),
        "side": lado, "type": tipo, "time_in_force": tif,
    }
    if tipo == "limit":
        if not precio_limite or precio_limite <= 0:
            return False, "Una orden límite necesita un precio límite válido."
        cuerpo["limit_price"] = str(round(float(precio_limite), 2))

    ok, d = _pedir("POST", "/orders", json=cuerpo)
    if not ok:
        return False, str(d)
    return True, f"Orden enviada · id {d.get('id', '?')[:8]} · estado {d.get('status', '?')}"


def enviar_orden_accion(
    simbolo: str,
    cantidad: int,
    lado: str = "buy",
    tipo: str = "market",
    precio_limite: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    tif: str = "day",
) -> tuple[bool, str]:
    """Envía una orden de acciones, con bracket nativo si hay stop y TP.

    A diferencia de las opciones, en acciones Alpaca SÍ admite bracket:
    entrada, stop y objetivo viajan como una sola orden atómica y el
    broker cancela la contraria cuando una se ejecuta.
    """
    if cantidad < 1:
        return False, "La cantidad debe ser al menos 1."

    cuerpo = {
        "symbol": simbolo, "qty": str(int(cantidad)),
        "side": lado, "type": tipo, "time_in_force": tif,
    }
    if tipo == "limit":
        if not precio_limite or precio_limite <= 0:
            return False, "Una orden límite necesita un precio límite válido."
        cuerpo["limit_price"] = str(round(float(precio_limite), 2))

    if stop_loss and take_profit:
        cuerpo["order_class"] = "bracket"
        cuerpo["stop_loss"]   = {"stop_price": str(round(float(stop_loss), 2))}
        cuerpo["take_profit"] = {"limit_price": str(round(float(take_profit), 2))}
        # El bracket exige GTC: con 'day' el stop moriría al cierre
        cuerpo["time_in_force"] = "gtc"

    ok, d = _pedir("POST", "/orders", json=cuerpo)
    if not ok:
        return False, str(d)
    clase = d.get("order_class") or "simple"
    return True, f"Orden {clase} enviada · id {d.get('id', '?')[:8]} · estado {d.get('status', '?')}"


def cancelar_orden(id_orden: str) -> tuple[bool, str]:
    """Cancela una orden pendiente."""
    ok, d = _pedir("DELETE", f"/orders/{id_orden}")
    return (True, "Orden cancelada.") if ok else (False, str(d))


def cerrar_posicion(simbolo: str) -> tuple[bool, str]:
    """Cierra una posición completa a mercado."""
    ok, d = _pedir("DELETE", f"/positions/{simbolo}")
    return (True, f"Cierre enviado para {simbolo}.") if ok else (False, str(d))
