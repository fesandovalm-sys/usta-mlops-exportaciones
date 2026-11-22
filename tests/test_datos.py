"""Contrato de datos probado exclusivamente con exportaciones sintéticas."""

import numpy as np
import pandas as pd
import pytest

from src.modelo import CATEGORICAS, NUMERICAS, OBJETIVO, VARIABLES
from src.monitoreo import validar_lote


@pytest.fixture
def datos():
    return pd.DataFrame({
        "periodo": ["2025-08", "2025-09", "2026-06", "2026-08"],
        "pnk": [0.0, 120.5, 50.0, 2500.0],
        "fobdol": [100.0, 0.0, 930.25, 25000.0],
        "capitulo": ["01", "09", "27", "90"],
        "modad": ["1", "2", "1", "2"],
        "dpto_origen": ["05", "11", "76", "08"],
        OBJETIVO: [0, 1, 0, 1],
    })


def _comprobar_contrato(datos):
    obligatorias = {"periodo", *VARIABLES, OBJETIVO}
    assert obligatorias.issubset(datos.columns), "Faltan columnas obligatorias"
    assert not datos[list(obligatorias)].isna().any().any(), "Hay valores nulos"
    for columna in NUMERICAS:
        assert pd.api.types.is_numeric_dtype(datos[columna]), f"Tipo numérico incorrecto: {columna}"
        assert np.isfinite(datos[columna].to_numpy(dtype=float)).all(), f"No finitos: {columna}"
        assert (datos[columna] >= 0).all(), f"Valor negativo en {columna}"
    for columna in CATEGORICAS:
        assert datos[columna].map(lambda x: isinstance(x, str) and bool(x.strip())).all(), (
            f"Categoría inválida: {columna}"
        )
    assert datos["periodo"].str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all(), "Periodo inválido"
    assert pd.api.types.is_integer_dtype(datos[OBJETIVO]), "Target no entero"
    assert set(datos[OBJETIVO].unique()) <= {0, 1}, "Target fuera de rango"


def test_columnas_obligatorias_y_esquema(datos):
    _comprobar_contrato(datos)
    for columna in ["periodo", *VARIABLES, OBJETIVO]:
        with pytest.raises(AssertionError, match="columnas obligatorias"):
            _comprobar_contrato(datos.drop(columns=[columna]))


def test_tipos_de_features_y_target(datos):
    _comprobar_contrato(datos)
    with pytest.raises(AssertionError, match="Tipo numérico"):
        _comprobar_contrato(datos.assign(pnk="valor-invalido"))
    with pytest.raises(AssertionError, match="Categoría inválida"):
        _comprobar_contrato(datos.assign(capitulo=123))


@pytest.mark.parametrize("columna", NUMERICAS)
def test_rangos_no_negativos_y_valores_finitos(datos, columna):
    with pytest.raises(AssertionError, match="negativo"):
        _comprobar_contrato(datos.assign(**{columna: -1.0}))
    with pytest.raises(AssertionError, match="No finitos"):
        _comprobar_contrato(datos.assign(**{columna: np.inf}))


def test_nulos_rechazados(datos):
    validar_lote(datos)
    for columna in [*NUMERICAS, *CATEGORICAS, OBJETIVO]:
        alterado = datos.copy()
        alterado.loc[0, columna] = np.nan
        with pytest.raises(ValueError, match="nulos"):
            validar_lote(alterado)


def test_etiquetas_validas_y_fuera_de_rango(datos):
    validar_lote(datos)
    for etiqueta in [-1, 2, 0.5]:
        alterado = datos.copy()
        alterado[OBJETIVO] = etiqueta
        with pytest.raises(ValueError, match="0 y 1"):
            validar_lote(alterado)


def test_formato_de_periodo(datos):
    with pytest.raises(AssertionError, match="Periodo inválido"):
        _comprobar_contrato(datos.assign(periodo="2026-13"))
