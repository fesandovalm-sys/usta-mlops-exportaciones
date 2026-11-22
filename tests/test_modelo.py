
"""Pruebas del modelo EXPO sin MLflow, Prefect ni datos reales."""

import numpy as np
import pandas as pd
import pytest

from src.modelo import (
    CATEGORICAS,
    NUMERICAS,
    OBJETIVO,
    VARIABLES,
    calcular_metricas,
    construir_modelo,
    resolver_variables,
)


@pytest.fixture
def datos_sinteticos():
    """Datos artificiales para comprobar el pipeline."""
    n = 40
    return pd.DataFrame({
        "pnk": [100.0 + i * 10 for i in range(n)],
        "fobdol": [200.0 + i * 50 for i in range(n)],
        "capitulo": ["09" if i % 2 else "01" for i in range(n)],
        "modad": ["1" if i % 3 else "2" for i in range(n)],
        "dpto_origen": ["11" if i % 2 else "05" for i in range(n)],
        OBJETIVO: [i % 2 for i in range(n)],
    })


def test_variables_permitidas_sin_fuga_del_target():
    assert OBJETIVO not in VARIABLES
    assert "PAIS" not in VARIABLES
    assert "COD_PAI4" not in VARIABLES
    assert len(VARIABLES) == 5


def test_seleccion_all():
    numericas, categoricas = resolver_variables(
        {"seleccion": "ALL"}
    )
    assert numericas == NUMERICAS
    assert categoricas == CATEGORICAS


def test_seleccion_numeric():
    numericas, categoricas = resolver_variables(
        {"seleccion": "NUMERIC"}
    )
    assert numericas == NUMERICAS
    assert categoricas == []


def test_seleccion_custom():
    numericas, categoricas = resolver_variables({
        "seleccion": "CUSTOM",
        "custom": ["pnk", "capitulo"],
    })
    assert numericas == ["pnk"]
    assert categoricas == ["capitulo"]


@pytest.mark.parametrize("config", [
    {"seleccion": "INVALIDA"},
    {"seleccion": "CUSTOM", "custom": []},
    {"seleccion": "CUSTOM", "custom": ["PAIS"]},
    {"seleccion": "CUSTOM", "custom": ["pnk", "pnk"]},
])
def test_rechaza_selecciones_invalidas(config):
    with pytest.raises(ValueError):
        resolver_variables(config)


@pytest.mark.parametrize("tipo,parametros", [
    ("logistic_regression", {"max_iter": 500, "random_state": 42}),
    ("random_forest", {"n_estimators": 10, "random_state": 42}),
])
def test_modelos_entrenan_y_predicen(
    datos_sinteticos, tipo, parametros
):
    modelo = construir_modelo(
        tipo,
        parametros,
        NUMERICAS,
        CATEGORICAS,
    )

    X = datos_sinteticos[VARIABLES]
    y = datos_sinteticos[OBJETIVO]

    modelo.fit(X, y)
    probabilidades = modelo.predict_proba(X)

    assert probabilidades.shape == (len(X), 2)
    assert np.isfinite(probabilidades).all()
    assert (probabilidades >= 0).all()
    assert (probabilidades <= 1).all()
    np.testing.assert_allclose(
        probabilidades.sum(axis=1),
        np.ones(len(X)),
    )


def test_modelo_rechaza_algoritmo_desconocido():
    with pytest.raises(ValueError, match="no soportado"):
        construir_modelo(
            "algoritmo_inexistente", {}, NUMERICAS, CATEGORICAS
        )


def test_metricas_clasificacion_correcta():
    y = np.array([0, 0, 1, 1])
    probabilidades = np.array([0.1, 0.2, 0.8, 0.9])

    metricas = calcular_metricas(y, probabilidades)

    assert metricas["auc"] == pytest.approx(1.0)
    assert metricas["accuracy"] == pytest.approx(1.0)
    assert metricas["precision"] == pytest.approx(1.0)
    assert metricas["recall"] == pytest.approx(1.0)
    assert metricas["f1"] == pytest.approx(1.0)


def test_metricas_rechazan_una_sola_clase():
    with pytest.raises(ValueError, match="dos clases"):
        calcular_metricas([1, 1, 1], [0.8, 0.9, 0.7])


@pytest.mark.parametrize("umbral", [-0.1, 1.1])
def test_metricas_rechazan_umbral_invalido(umbral):
    with pytest.raises(ValueError, match="umbral"):
        calcular_metricas(
            [0, 1], [0.2, 0.8], umbral=umbral
        )
