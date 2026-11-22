"""Lógica de monitoreo y simulación de drift, independiente de Prefect y MLflow."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.modelo import CATEGORICAS, NUMERICAS, OBJETIVO, VARIABLES

# Cambio máximo simulado en la proporción de destino_usa (30 puntos porcentuales).
SUBIDA_TASA_MAXIMA = 0.30


def validar_lote(lote):
    """Rechaza lotes vacíos, sin target, con nulos o con etiquetas no binarias."""
    if not isinstance(lote, pd.DataFrame):
        raise TypeError("El lote debe ser un DataFrame de pandas")
    if lote.empty:
        raise ValueError("El lote está vacío")
    if OBJETIVO not in lote.columns:
        raise ValueError(f"El lote no contiene la etiqueta '{OBJETIVO}'")
    if lote.isna().any().any():
        raise ValueError("El lote contiene valores nulos")
    if not lote[OBJETIVO].isin([0, 1]).all():
        raise ValueError(f"'{OBJETIVO}' debe contener únicamente etiquetas 0 y 1")


def aplicar_drift(lote, nivel, semilla=42):
    """Simula drift sin alterar el lote original.

    Desplaza hacia arriba las distribuciones de pnk y fobdol en 2 * nivel
    desviaciones estándar (sin admitir valores negativos). Convierte al azar
    etiquetas 0 en 1 hasta subir la tasa de destino_usa en 0.30 * nivel
    puntos de proporción. Conserva las etiquetas observadas originales.
    La semilla fija hace reproducible la selección de etiquetas modificadas.
    """
    try:
        nivel = float(nivel)
    except (TypeError, ValueError) as exc:
        raise ValueError("El nivel de drift debe estar entre 0.0 y 1.0") from exc
    if not np.isfinite(nivel) or not 0 <= nivel <= 1:
        raise ValueError("El nivel de drift debe estar entre 0.0 y 1.0")

    validar_lote(lote)
    faltantes = [columna for columna in NUMERICAS if columna not in lote.columns]
    if faltantes:
        raise ValueError(f"Faltan variables numéricas para emular drift: {faltantes}")

    numericas = lote[NUMERICAS].apply(pd.to_numeric, errors="coerce")
    if numericas.isna().any().any() or not np.isfinite(numericas.to_numpy(dtype=float)).all():
        raise ValueError("pnk y fobdol deben contener valores numéricos finitos")
    if (numericas < 0).any().any():
        raise ValueError("pnk y fobdol no pueden contener valores negativos")

    salida = lote.copy(deep=True)
    salida[f"{OBJETIVO}_original"] = lote[OBJETIVO].copy()
    if nivel == 0.0:
        return salida

    for columna in NUMERICAS:
        valores = numericas[columna].astype(float)
        desviacion = float(valores.std(ddof=0))
        # En una variable constante también se garantiza un cambio controlado.
        escala = desviacion if desviacion > 0 else max(1.0, float(valores.mean()))
        salida[columna] = (valores + 2.0 * nivel * escala).clip(lower=0)

    rng = np.random.default_rng(semilla)
    etiquetas = lote[OBJETIVO].to_numpy(dtype=int, copy=True)
    cantidad_positivos = int(etiquetas.sum())
    tasa_objetivo = min(1.0, cantidad_positivos / len(etiquetas) + nivel * SUBIDA_TASA_MAXIMA)
    faltan = max(0, round(tasa_objetivo * len(etiquetas)) - cantidad_positivos)
    indices_negativos = np.flatnonzero(etiquetas == 0)
    if faltan:
        elegidos = rng.choice(indices_negativos, size=min(faltan, len(indices_negativos)), replace=False)
        etiquetas[elegidos] = 1
    salida[OBJETIVO] = etiquetas
    return salida


def _extraer_etiquetas(datos, nombre):
    """Normaliza una serie o un DataFrame con target a etiquetas binarias."""
    if isinstance(datos, pd.DataFrame):
        if OBJETIVO not in datos.columns:
            raise ValueError(f"{nombre} no contiene la columna '{OBJETIVO}'")
        etiquetas = datos[OBJETIVO]
    elif isinstance(datos, pd.Series):
        etiquetas = datos
    else:
        raise TypeError(f"{nombre} debe ser una Serie o un DataFrame de pandas")
    if etiquetas.empty or etiquetas.isna().any():
        raise ValueError(f"{nombre} no puede estar vacío ni contener nulos")
    if not etiquetas.isin([0, 1]).all():
        raise ValueError(f"{nombre} debe contener únicamente etiquetas 0 y 1")
    return etiquetas.astype(int)


def drift_de_target(referencia, lote, alfa=0.05):
    """Compara proporciones binarias mediante chi-cuadrado y su p-valor."""
    if not 0 < alfa < 1:
        raise ValueError("alfa debe estar entre 0 y 1, sin incluir los extremos")
    ref = _extraer_etiquetas(referencia, "La referencia")
    actual = _extraer_etiquetas(lote, "El lote")
    positivos_ref = int(ref.sum())
    positivos_lote = int(actual.sum())
    observados = [
        [positivos_ref, len(ref) - positivos_ref],
        [positivos_lote, len(actual) - positivos_lote],
    ]
    # Si ambos conjuntos tienen una sola clase idéntica, no hay cambio.
    if sum(fila[0] for fila in observados) == 0 or sum(fila[1] for fila in observados) == 0:
        p_valor = 1.0
    else:
        _, p_valor, _, _ = chi2_contingency(observados)
    return {
        "tasa_referencia": positivos_ref / len(ref),
        "tasa_lote": positivos_lote / len(actual),
        "p_valor": float(p_valor),
        "drift": bool(p_valor < alfa),
    }


def fraccion_desde_evidently(resultado):
    """Extrae el share de DriftedColumnsCount de un snapshot de Evidently."""
    if not isinstance(resultado, dict):
        if not hasattr(resultado, "dict"):
            raise TypeError("El resultado debe ser un diccionario o snapshot de Evidently")
        resultado = resultado.dict()
    for metrica in resultado.get("metrics", []):
        if str(metrica.get("metric_name", "")).startswith("DriftedColumnsCount"):
            valor = metrica.get("value", {})
            if not isinstance(valor, dict):
                break
            fraccion = valor.get("share", valor.get("drift_share"))
            if fraccion is None:
                break
            fraccion = float(fraccion)
            if not np.isfinite(fraccion) or not 0 <= fraccion <= 1:
                raise ValueError("La fracción de drift debe estar entre 0 y 1")
            return fraccion
    raise ValueError("Evidently no incluye la fracción DriftedColumnsCount")
