
"""Pruebas de los flujos de drift sin servidores externos."""

import json
from pathlib import Path

import pandas as pd
import pytest

from flows import drift_evento_flow as evento
from flows import drift_features_flow as features
from src.modelo import OBJETIVO, VARIABLES


@pytest.fixture
def config(tmp_path):
    return {
        "inferencia": {"nivel_drift": 0.6},
        "monitoreo": {
            "alfa_evento": 0.05,
            "umbral_drift": 0.30,
            "carpeta_reportes": "reportes_prueba",
        },
    }


@pytest.fixture
def datos_features():
    return pd.DataFrame({
        "pnk": [100.0, 200.0, 300.0, 400.0],
        "fobdol": [500.0, 600.0, 700.0, 800.0],
        "capitulo": ["01", "09", "27", "90"],
        "modad": ["1", "2", "1", "2"],
        "dpto_origen": ["05", "11", "76", "08"],
    })


def test_evento_rechaza_lote_sin_target(tmp_path):
    ruta = tmp_path / "sin_etiquetas.csv"
    pd.DataFrame({"pnk": [100.0]}).to_csv(ruta, index=False)

    with pytest.raises(
        ValueError, match="no contiene la columna"
    ):
        evento.leer_target(ruta)


def test_evento_rechaza_lote_vacio(tmp_path):
    ruta = tmp_path / "vacio.csv"
    pd.DataFrame({OBJETIVO: []}).to_csv(ruta, index=False)

    with pytest.raises(ValueError, match="sin registros"):
        evento.leer_target(ruta)


def test_evento_rechaza_archivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        evento.leer_target(tmp_path / "no_existe.csv")


@pytest.mark.parametrize(
    "positivos_lote,espera_alerta",
    [(20, False), (60, True)],
)
def test_flujo_target_genera_resultado_y_alerta(
    monkeypatch, tmp_path, config,
    positivos_lote, espera_alerta
):
    monkeypatch.setattr(evento, "RAIZ", tmp_path)

    referencia = pd.Series([0] * 80 + [1] * 20)
    lote = pd.Series(
        [0] * (100 - positivos_lote)
        + [1] * positivos_lote
    )

    resultado = evento.evaluar_drift.fn(
        referencia, lote, config
    )

    assert resultado["drift"] is espera_alerta
    assert resultado["tasa_referencia"] == pytest.approx(0.2)
    assert resultado["tasa_lote"] == pytest.approx(
        positivos_lote / 100
    )

    archivos = list(
        (tmp_path / "reportes_prueba").glob("drift_target_*.json")
    )
    assert len(archivos) == 1

    contenido = json.loads(
        archivos[0].read_text(encoding="utf-8")
    )
    assert contenido["drift"] is espera_alerta


def test_flujo_target_rechaza_alfa_invalido(
    monkeypatch, tmp_path, config
):
    monkeypatch.setattr(evento, "RAIZ", tmp_path)
    config["monitoreo"]["alfa_evento"] = 1.5

    referencia = pd.Series([0, 1, 0, 1])
    lote = pd.Series([0, 0, 1, 1])

    with pytest.raises(ValueError, match="alfa"):
        evento.evaluar_drift.fn(
            referencia, lote, config
        )


def test_features_preserva_codigos_categoricos(
    tmp_path, datos_features
):
    ruta = tmp_path / "features.csv"
    datos_features.to_csv(ruta, index=False)

    cargados = features.leer_features(ruta)

    assert cargados.columns.tolist() == VARIABLES
    assert cargados["capitulo"].iloc[0] == "01"
    assert cargados["dpto_origen"].iloc[0] == "05"


@pytest.mark.parametrize(
    "fraccion,espera_alerta",
    [(0.2, False), (0.4, True), (0.6, True)],
)
def test_flujo_features_respeta_umbral_y_genera_html(
    monkeypatch, tmp_path, config, datos_features,
    fraccion, espera_alerta
):
    """Simula Evidently y comprueba la logica del flujo."""
    monkeypatch.setattr(features, "RAIZ", tmp_path)

    ejecuciones = []

    def preset_falso(drift_share):
        return drift_share

    class ReporteFalso:
        def __init__(self, presets):
            assert presets == [0.30]

        def run(self, reference_data, current_data):
            ejecuciones.append(1)
            assert len(reference_data) == 4
            assert len(current_data) == 4
            return self

        def dict(self):
            return {
                "metrics": [{
                    "metric_name": "DriftedColumnsCount",
                    "value": {
                        "count": round(fraccion * 5),
                        "share": fraccion,
                    },
                }]
            }

        def save_html(self, ruta):
            Path(ruta).write_text(
                "Reporte sintetico de prueba",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        features, "DataDriftPreset", preset_falso
    )
    monkeypatch.setattr(
        features, "Report", ReporteFalso
    )

    resultado = features.analizar_drift.fn(
        datos_features, datos_features, config
    )

    assert resultado["alerta"] is espera_alerta
    assert resultado["fraccion_con_drift"] == pytest.approx(
        fraccion
    )
    assert resultado["umbral"] == pytest.approx(0.30)

    # Evidently debe ejecutarse una sola vez.
    assert len(ejecuciones) == 1

    ruta_html = Path(resultado["reporte_html"])
    assert ruta_html.is_file()

    resumenes = list(
        (tmp_path / "reportes_prueba").glob(
            "drift_features_*.json"
        )
    )
    assert len(resumenes) == 1


@pytest.mark.parametrize("umbral", [-0.1, 1.1])
def test_features_rechaza_umbral_invalido(
    monkeypatch, tmp_path, config, datos_features, umbral
):
    monkeypatch.setattr(features, "RAIZ", tmp_path)
    config["monitoreo"]["umbral_drift"] = umbral

    with pytest.raises(ValueError, match="umbral_drift"):
        features.analizar_drift.fn(
            datos_features, datos_features, config
        )
