
"""Pruebas de FastAPI sin conexiones a MLflow ni servicios externos."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import main as modulo


class ModeloFalso:
    """Simula el contrato del modelo champion."""

    feature_names_in_ = np.array([
        "pnk", "fobdol", "capitulo",
        "modad", "dpto_origen",
    ])
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        assert list(X.columns) == list(self.feature_names_in_)
        assert "destino_usa" not in X.columns
        assert len(X) == 1

        probabilidad = (
            0.8 if float(X["pnk"].iloc[0]) >= 200 else 0.2
        )
        return np.array([[1 - probabilidad, probabilidad]])


@pytest.fixture
def registro():
    return {
        "capitulo": "09",
        "modad": "1",
        "dpto_origen": "11",
        "pnk": 250.0,
        "fobdol": 6250.0,
    }


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.setattr(
        modulo,
        "cargar_champion",
        lambda: (ModeloFalso(), "2"),
    )
    return TestClient(modulo.app)


def test_health_con_champion_disponible(cliente):
    respuesta = cliente.get("/health")

    assert respuesta.status_code == 200
    datos = respuesta.json()

    assert datos["estado"] == "listo"
    assert datos["alias"] == "champion"
    assert datos["version"] == "2"


def test_health_sin_champion(monkeypatch):
    def modelo_no_disponible():
        raise RuntimeError("Modelo no disponible")

    monkeypatch.setattr(
        modulo, "cargar_champion", modelo_no_disponible
    )

    cliente = TestClient(modulo.app)
    respuesta = cliente.get("/health")

    assert respuesta.status_code == 200
    assert (
        respuesta.json()["estado"]
        == "sin_champion_disponible"
    )
    assert respuesta.json()["version"] is None


def test_predict_clase_positiva(cliente, registro):
    respuesta = cliente.post("/predict", json=registro)

    assert respuesta.status_code == 200
    datos = respuesta.json()

    assert datos["version"] == "2"
    assert datos["alias"] == "champion"
    assert datos["prediccion_destino_usa"] == 1
    assert datos["probabilidad_destino_usa"] == pytest.approx(0.8)
    assert datos["umbral_clasificacion"] == pytest.approx(0.5)


def test_predict_clase_negativa(cliente, registro):
    registro["pnk"] = 100.0

    respuesta = cliente.post("/predict", json=registro)

    assert respuesta.status_code == 200
    assert respuesta.json()["prediccion_destino_usa"] == 0
    assert (
        respuesta.json()["probabilidad_destino_usa"]
        == pytest.approx(0.2)
    )


@pytest.mark.parametrize("campo", [
    "pnk",
    "fobdol",
    "capitulo",
    "modad",
    "dpto_origen",
])
def test_predict_rechaza_campos_faltantes(
    cliente, registro, campo
):
    registro.pop(campo)

    respuesta = cliente.post("/predict", json=registro)

    assert respuesta.status_code == 422


def test_predict_rechaza_variable_no_permitida(
    cliente, registro
):
    registro["destino_usa"] = 1

    respuesta = cliente.post("/predict", json=registro)

    assert respuesta.status_code == 422


def test_predict_rechaza_numero_invalido(cliente, registro):
    registro["pnk"] = "no-es-un-numero"

    respuesta = cliente.post("/predict", json=registro)

    assert respuesta.status_code == 422


def test_predict_sin_champion_devuelve_503(
    monkeypatch, registro
):
    def modelo_no_disponible():
        raise RuntimeError("No se encuentra champion")

    monkeypatch.setattr(
        modulo, "cargar_champion", modelo_no_disponible
    )

    cliente = TestClient(modulo.app)
    respuesta = cliente.post("/predict", json=registro)

    assert respuesta.status_code == 503


def test_predict_rechaza_feature_faltante_en_modelo(
    monkeypatch, registro
):
    class ModeloIncompatible(ModeloFalso):
        feature_names_in_ = np.array([
            "pnk", "fobdol", "variable_inexistente",
        ])

    monkeypatch.setattr(
        modulo,
        "cargar_champion",
        lambda: (ModeloIncompatible(), "2"),
    )

    cliente = TestClient(modulo.app)
    respuesta = cliente.post("/predict", json=registro)

    assert respuesta.status_code == 422


def test_predict_error_de_procesamiento(
    monkeypatch, registro
):
    class ModeloConError(ModeloFalso):
        def predict_proba(self, X):
            raise ValueError("Prediccion imposible")

    monkeypatch.setattr(
        modulo,
        "cargar_champion",
        lambda: (ModeloConError(), "2"),
    )

    cliente = TestClient(modulo.app)
    respuesta = cliente.post("/predict", json=registro)

    assert respuesta.status_code == 422


def test_predict_rechaza_probabilidad_fuera_de_rango(
    monkeypatch, registro
):
    class ModeloInvalido(ModeloFalso):
        def predict_proba(self, X):
            return np.array([[-0.2, 1.2]])

    monkeypatch.setattr(
        modulo,
        "cargar_champion",
        lambda: (ModeloInvalido(), "2"),
    )

    cliente = TestClient(modulo.app)
    respuesta = cliente.post("/predict", json=registro)

    assert respuesta.status_code == 500
