
"""Pruebas de inferencia EXPO sin servidores Prefect ni MLflow."""

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from flows import inferencia_flow as modulo


@pytest.fixture
def fecha():
    return datetime(
        2026, 11, 22, 14, 0,
        tzinfo=ZoneInfo("America/Bogota"),
    )


@pytest.fixture
def datos():
    n = 30
    return pd.DataFrame({
        "periodo": ["2026-09"] * n,
        "pnk": [100.0 + i * 10 for i in range(n)],
        "fobdol": [500.0 + i * 100 for i in range(n)],
        "capitulo": ["09" if i % 2 else "01" for i in range(n)],
        "modad": ["1" if i % 3 else "2" for i in range(n)],
        "dpto_origen": ["11" if i % 2 else "05" for i in range(n)],
        "destino_usa": [int(i % 4 == 0) for i in range(n)],
    })


@pytest.fixture
def config(tmp_path):
    return {
        "datos": {
            "inferencia": "datos_prueba.csv",
        },
        "inferencia": {
            "tamano_lote": 20,
            "nivel_drift": 0.0,
            "semilla_drift": 42,
            "umbral_clasificacion": 0.5,
        },
        "monitoreo": {
            "carpeta_lotes": "salidas/lotes",
            "lote_actual": "salidas/lotes/lote_actual.csv",
        },
        "registro": {
            "nombre_modelo": "modelo-prueba",
            "alias": "champion",
        },
    }


def test_cargar_lote_reproducible(
    monkeypatch, tmp_path, datos, config, fecha
):
    datos.to_csv(tmp_path / "datos_prueba.csv", index=False)
    monkeypatch.setattr(modulo, "RAIZ", tmp_path)

    lote_uno = modulo.cargar_lote.fn(config, fecha)
    lote_dos = modulo.cargar_lote.fn(config, fecha)

    pd.testing.assert_frame_equal(lote_uno, lote_dos)
    assert len(lote_uno) == 20
    assert lote_uno["periodo"].eq("2026-09").all()
    assert "destino_usa" in lote_uno.columns


def test_cargar_lote_rechaza_columnas_faltantes(
    monkeypatch, tmp_path, datos, config, fecha
):
    datos.drop(columns=["fobdol"]).to_csv(
        tmp_path / "datos_prueba.csv", index=False
    )
    monkeypatch.setattr(modulo, "RAIZ", tmp_path)

    with pytest.raises(ValueError, match="Columnas faltantes"):
        modulo.cargar_lote.fn(config, fecha)


def test_cargar_lote_rechaza_tamano_invalido(
    monkeypatch, tmp_path, datos, config, fecha
):
    datos.to_csv(tmp_path / "datos_prueba.csv", index=False)
    config["inferencia"]["tamano_lote"] = 0
    monkeypatch.setattr(modulo, "RAIZ", tmp_path)

    with pytest.raises(ValueError, match="mayor que cero"):
        modulo.cargar_lote.fn(config, fecha)


@pytest.mark.parametrize("nivel", [0.0, 0.3, 0.6])
def test_simulacion_guarda_lote_y_conserva_original(
    monkeypatch, tmp_path, datos, config, fecha, nivel
):
    monkeypatch.setattr(modulo, "RAIZ", tmp_path)
    config["inferencia"]["nivel_drift"] = nivel
    original = datos.copy(deep=True)

    simulado, ruta = modulo.simular_y_guardar_lote.fn(
        datos, config, fecha
    )

    pd.testing.assert_frame_equal(datos, original)

    assert len(simulado) == len(original)
    assert (tmp_path / config["monitoreo"]["lote_actual"]).is_file()

    archivo = pd.read_csv(ruta)
    assert len(archivo) == len(original)
    assert "destino_usa_original" in archivo.columns

    assert (
        simulado["destino_usa_original"].to_numpy()
        == original["destino_usa"].to_numpy()
    ).all()

    if nivel == 0.0:
        assert (
            simulado["destino_usa"]
            == simulado["destino_usa_original"]
        ).all()
    else:
        assert (
            simulado["destino_usa"].mean()
            > original["destino_usa"].mean()
        )


def test_simulacion_rechaza_nivel_invalido(
    monkeypatch, tmp_path, datos, config, fecha
):
    monkeypatch.setattr(modulo, "RAIZ", tmp_path)
    config["inferencia"]["nivel_drift"] = 1.5

    with pytest.raises(ValueError, match="nivel_drift"):
        modulo.simular_y_guardar_lote.fn(
            datos, config, fecha
        )


def test_inferencia_champion_con_mock(
    monkeypatch, datos, config, fecha
):
    """Comprueba predicciones sin conectarse a MLflow."""

    class ModeloFalso:
        feature_names_in_ = np.array([
            "pnk", "fobdol", "capitulo",
            "modad", "dpto_origen",
        ])

        def predict_proba(self, X):
            assert "destino_usa" not in X.columns
            assert "destino_usa_original" not in X.columns

            probabilidades = np.where(
                X["pnk"].to_numpy() > 200, 0.8, 0.2
            )
            return np.column_stack([
                1 - probabilidades,
                probabilidades,
            ])

    cliente = SimpleNamespace(
        get_model_version_by_alias=lambda nombre, alias:
            SimpleNamespace(version="2")
    )

    monkeypatch.setattr(
        modulo, "configurar_mlflow", lambda cfg: None
    )
    monkeypatch.setattr(
        modulo, "MlflowClient", lambda: cliente
    )
    monkeypatch.setattr(
        modulo.mlflow.sklearn,
        "load_model",
        lambda uri: ModeloFalso(),
    )

    lote = datos.copy()
    lote["destino_usa_original"] = lote["destino_usa"]

    salida, version, variables = (
        modulo.predecir_con_champion.fn(
            lote, config, fecha
        )
    )

    assert len(salida) == len(lote)
    assert version == "2"
    assert len(variables) == 5
    assert salida["probabilidad_destino_usa"].between(0, 1).all()
    assert set(salida["prediccion_destino_usa"]) <= {0, 1}
    assert salida["modelo_version"].eq("2").all()

    assert (
        salida["destino_usa_observado"].to_numpy()
        == lote["destino_usa_original"].to_numpy()
    ).all()


def test_champion_rechaza_umbral_invalido(
    monkeypatch, datos, config, fecha
):
    class ModeloFalso:
        feature_names_in_ = np.array([
            "pnk", "fobdol", "capitulo",
            "modad", "dpto_origen",
        ])

        def predict_proba(self, X):
            return np.tile([0.4, 0.6], (len(X), 1))

    cliente = SimpleNamespace(
        get_model_version_by_alias=lambda nombre, alias:
            SimpleNamespace(version="2")
    )

    monkeypatch.setattr(
        modulo, "configurar_mlflow", lambda cfg: None
    )
    monkeypatch.setattr(
        modulo, "MlflowClient", lambda: cliente
    )
    monkeypatch.setattr(
        modulo.mlflow.sklearn,
        "load_model",
        lambda uri: ModeloFalso(),
    )

    config["inferencia"]["umbral_clasificacion"] = 1.5

    with pytest.raises(ValueError, match="umbral_clasificacion"):
        modulo.predecir_con_champion.fn(
            datos, config, fecha
        )
