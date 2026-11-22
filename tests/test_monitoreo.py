"""Pruebas unitarias de monitoreo sin servicios ni microdatos externos."""

import numpy as np
import pandas as pd
import pytest

from src.modelo import CATEGORICAS, NUMERICAS, OBJETIVO, VARIABLES
from src.monitoreo import (
    SUBIDA_TASA_MAXIMA,
    aplicar_drift,
    drift_de_target,
    fraccion_desde_evidently,
    validar_lote,
)


@pytest.fixture
def lote():
    rng = np.random.default_rng(2026)
    n = 500
    datos = pd.DataFrame({
        "pnk": rng.gamma(shape=2.0, scale=800, size=n),
        "fobdol": rng.lognormal(mean=8.0, sigma=1.0, size=n),
        "capitulo": rng.choice(["01", "09", "27"], n),
        "modad": rng.choice(["1", "2"], n),
        "dpto_origen": rng.choice(["05", "11", "76"], n),
        OBJETIVO: rng.choice([0, 1], n, p=[0.78, 0.22]),
    })
    return datos


@pytest.mark.parametrize("nivel", [0.0, 0.3, 0.6])
def test_tres_niveles_preservan_originales_y_no_generan_negativos(lote, nivel):
    original = lote.copy(deep=True)
    resultado = aplicar_drift(lote, nivel)
    pd.testing.assert_frame_equal(lote, original)
    pd.testing.assert_series_equal(
        resultado[f"{OBJETIVO}_original"], original[OBJETIVO], check_names=False
    )
    assert set(resultado[OBJETIVO].unique()) <= {0, 1}
    assert (resultado[NUMERICAS] >= 0).all().all()
    esperada = min(1, original[OBJETIVO].mean() + nivel * SUBIDA_TASA_MAXIMA)
    assert resultado[OBJETIVO].mean() == pytest.approx(esperada, abs=1 / len(lote))


def test_nivel_cero_no_cambia_variables_ni_objetivo(lote):
    salida = aplicar_drift(lote, 0.0)
    pd.testing.assert_frame_equal(salida[lote.columns], lote)


def test_mayor_nivel_desplaza_distribuciones_numericas(lote):
    leve = aplicar_drift(lote, 0.3)
    alto = aplicar_drift(lote, 0.6)
    assert NUMERICAS == ["pnk", "fobdol"]
    assert set(NUMERICAS + CATEGORICAS) == set(VARIABLES)
    for columna in NUMERICAS:
        assert lote[columna].mean() < leve[columna].mean() < alto[columna].mean()
        assert not np.allclose(lote[columna], alto[columna])
    assert lote[OBJETIVO].mean() < leve[OBJETIVO].mean() < alto[OBJETIVO].mean()


def test_reproducibilidad_con_semilla_y_cambio_de_semilla(lote):
    uno = aplicar_drift(lote, 0.6, semilla=42)
    dos = aplicar_drift(lote, 0.6, semilla=42)
    otro = aplicar_drift(lote, 0.6, semilla=123)
    pd.testing.assert_frame_equal(uno, dos)
    assert not uno[OBJETIVO].equals(otro[OBJETIVO])


@pytest.mark.parametrize("nivel", [-0.1, 1.1, float("nan"), "no valido"])
def test_rechaza_niveles_invalidos(lote, nivel):
    with pytest.raises(ValueError, match="nivel de drift"):
        aplicar_drift(lote, nivel)


def test_rechaza_valores_negativos_de_entrada(lote):
    lote.loc[lote.index[0], "fobdol"] = -1.0
    with pytest.raises(ValueError, match="negativos"):
        aplicar_drift(lote, 0.6)


def test_validar_lote_rechaza_vacios_nulos_etiqueta_ausente_e_invalida(lote):
    validar_lote(lote)
    casos = [
        lote.iloc[0:0],
        lote.drop(columns=[OBJETIVO]),
        lote.assign(pnk=np.nan),
        lote.assign(**{OBJETIVO: 2}),
    ]
    for caso in casos:
        with pytest.raises(ValueError):
            validar_lote(caso)


def test_target_detecta_cambio_significativo():
    referencia = pd.Series([0] * 800 + [1] * 200)
    lote = pd.Series([0] * 400 + [1] * 600)
    resultado = drift_de_target(referencia, lote)
    assert set(resultado) == {"tasa_referencia", "tasa_lote", "p_valor", "drift"}
    assert resultado["tasa_referencia"] == pytest.approx(0.2)
    assert resultado["tasa_lote"] == pytest.approx(0.6)
    assert resultado["p_valor"] < 0.05
    assert resultado["drift"] is True


def test_target_no_detecta_cambio_si_mantiene_proporcion():
    referencia = pd.Series([0] * 800 + [1] * 200)
    lote = pd.DataFrame({OBJETIVO: [0] * 80 + [1] * 20})
    resultado = drift_de_target(referencia, lote)
    assert resultado["drift"] is False
    assert resultado["tasa_lote"] == pytest.approx(0.2)


def test_target_degenerado_no_falla():
    resultado = drift_de_target(pd.Series([0] * 20), pd.Series([0] * 10))
    assert resultado["p_valor"] == 1.0
    assert resultado["drift"] is False


def test_fraccion_desde_evidently_y_ausencia_de_metrica():
    reporte = {"metrics": [
        {"metric_name": "ValueDrift(column=pnk)", "value": 0.3},
        {"metric_name": "DriftedColumnsCount(drift_share=0.4)",
         "value": {"count": 2, "share": 0.4}},
    ]}
    assert fraccion_desde_evidently(reporte) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="DriftedColumnsCount"):
        fraccion_desde_evidently({"metrics": []})
