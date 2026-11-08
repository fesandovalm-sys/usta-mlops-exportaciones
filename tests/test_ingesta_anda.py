"""Pruebas automatizadas de la ingesta EXPO (DANE/ANDA), sin datos reales.

Ejecutar desde la raíz del repositorio:
    poetry run python -m pytest tests/test_ingesta_anda.py -q
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.ingesta_anda import (
    COLUMNAS_SALIDA,
    conjunto,
    numero_colombiano,
    procesar_bloque,
)


def fila(**cambios: str) -> dict[str, str]:
    """Registro sintético coherente con agosto de 2025."""
    valores = {
        "FECH": "2508",
        "PAIS": "840",
        "COD_PAI4": "USA",
        "POSAR": "0101210000",
        "MODAD": "198",
        "DPTO1": "5",
        "PNK": "1.234,50",
        "FOBDOL": "33.716,76",
    }
    valores.update(cambios)
    return valores


def test_conversion_numerica_colombiana() -> None:
    numeros = numero_colombiano(pd.Series(["33.716,76", "1.234", "0", "-5", "NO ES NUMERO"]))
    assert numeros.iloc[0] == pytest.approx(33716.76)
    assert numeros.iloc[1] == pytest.approx(1234.0)
    assert numeros.iloc[2] == 0
    assert numeros.iloc[3] == -5
    assert pd.isna(numeros.iloc[4])


def test_etiqueta_binaria_y_ausencia_de_fuga_directa() -> None:
    bloque = pd.DataFrame([fila(), fila(PAIS="170", COD_PAI4="COL")])
    salida, motivos = procesar_bloque(bloque, "2025-08")
    assert motivos.total() == 0
    assert salida.columns.tolist() == COLUMNAS_SALIDA
    assert salida["destino_usa"].tolist() == [1, 0]
    assert not {"PAIS", "COD_PAI4", "POSAR"}.intersection(salida.columns)
    assert salida["destino_usa"].dtype.kind in "iu"


def test_ceros_a_la_izquierda_y_numeros() -> None:
    salida, _ = procesar_bloque(pd.DataFrame([fila()]), "2025-08")
    assert salida.iloc[0]["capitulo"] == "01"
    assert salida.iloc[0]["dpto_origen"] == "05"
    assert salida.iloc[0]["modad"] == "198"
    assert salida.iloc[0]["pnk"] == pytest.approx(1234.50)
    assert salida.iloc[0]["fobdol"] == pytest.approx(33716.76)
    assert salida.iloc[0]["periodo"] == "2025-08"


@pytest.mark.parametrize(
    "periodo,esperado",
    [
        ("2025-08", "train"), ("2026-05", "train"),
        ("2026-06", "valid"), ("2026-08", "valid"),
        ("2026-09", "sept"),
    ],
)
def test_particiones_temporales(periodo: str, esperado: str) -> None:
    assert conjunto(periodo) == esperado


@pytest.mark.parametrize("periodo", ["2025-07", "2026-10", "2027-01"])
def test_periodo_fuera_del_contrato(periodo: str) -> None:
    with pytest.raises(ValueError, match="Periodo fuera del contrato"):
        conjunto(periodo)


def test_columnas_obligatorias() -> None:
    bloque = pd.DataFrame([fila()]).drop(columns="POSAR")
    with pytest.raises(ValueError, match="Faltan columnas.*POSAR"):
        procesar_bloque(bloque, "2025-08")


@pytest.mark.parametrize(
    "cambios,motivo",
    [
        ({"PAIS": "08", "COD_PAI4": ""}, "pais_invalido"),
        ({"PAIS": "840", "COD_PAI4": "COL"}, "pais_invalido"),
        ({"PAIS": "170", "COD_PAI4": "USA"}, "pais_invalido"),
        ({"PNK": "0"}, "peso_invalido"),
        ({"PNK": ""}, "peso_invalido"),
        ({"FOBDOL": "-1"}, "fob_invalido"),
        ({"POSAR": "123"}, "posar_invalida"),
        ({"FECH": "2509"}, "periodo_incorrecto"),
    ],
)
def test_exclusion_documentada(cambios: dict[str, str], motivo: str) -> None:
    salida, motivos = procesar_bloque(pd.DataFrame([fila(**cambios)]), "2025-08")
    assert salida.empty
    assert motivos[motivo] == 1


def test_reproducibilidad_sin_modificar_entrada() -> None:
    original = pd.DataFrame([fila(), fila(PAIS="170", COD_PAI4="COL")])
    resultado_1, motivos_1 = procesar_bloque(original.copy(), "2025-08")
    resultado_2, motivos_2 = procesar_bloque(original.copy(), "2025-08")
    assert_frame_equal(resultado_1, resultado_2)
    assert motivos_1 == motivos_2
