
"""Actividad 3: deteccion de drift de caracteristicas EXPO."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset
from prefect import flow, task

from src.modelo import VARIABLES, CATEGORICAS
from src.monitoreo import fraccion_desde_evidently


RAIZ = Path(__file__).resolve().parents[1]


def leer_configuracion():
    with (RAIZ / "configs/config.yaml").open(
        encoding="utf-8"
    ) as archivo:
        return yaml.safe_load(archivo)


def leer_features(ruta):
    """Lee las variables predictoras sin perder ceros iniciales."""
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe: {ruta}")

    datos = pd.read_csv(
        ruta,
        usecols=VARIABLES,
        dtype={col: str for col in CATEGORICAS},
    )

    if datos.empty:
        raise ValueError(f"Archivo vacio: {ruta}")

    return datos[VARIABLES].copy()


@task(name="referencia-drift-features")
def cargar_referencia(config):
    ruta = RAIZ / config["datos"]["entrenamiento"]
    referencia = leer_features(ruta)

    tamano = int(config["monitoreo"]["tamano_referencia"])
    if tamano < 1:
        raise ValueError("tamano_referencia debe ser positivo")

    referencia = referencia.sample(
        n=min(tamano, len(referencia)),
        random_state=42,
    ).reset_index(drop=True)

    print(f"Referencia: {len(referencia)} registros")
    return referencia


@task(name="lote-drift-features")
def cargar_lote(config):
    ruta = RAIZ / config["monitoreo"]["lote_actual"]
    lote = leer_features(ruta)

    print(f"Lote monitoreado: {len(lote)} registros")
    return lote


@task(name="reporte-evidently-features")
def analizar_drift(referencia, lote, config):
    """Genera el reporte y determina si corresponde una alerta."""
    umbral = float(config["monitoreo"]["umbral_drift"])
    if not 0 <= umbral <= 1:
        raise ValueError("umbral_drift debe estar entre 0 y 1")

    resultado = Report([DataDriftPreset(drift_share=umbral)]).run(
        reference_data=referencia,
        current_data=lote,
    )

    fraccion = fraccion_desde_evidently(resultado.dict())
    alerta = fraccion > umbral

    carpeta = RAIZ / config["monitoreo"]["carpeta_reportes"]
    carpeta.mkdir(parents=True, exist_ok=True)

    nivel = float(config["inferencia"]["nivel_drift"])
    marca = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nombre = f"drift_features_{nivel:.1f}_{marca}"

    ruta_html = carpeta / f"{nombre}.html"
    ruta_json = carpeta / f"{nombre}.json"

    resultado.save_html(str(ruta_html))

    resumen = {
        "nivel_drift": nivel,
        "registros_referencia": len(referencia),
        "registros_lote": len(lote),
        "variables_analizadas": VARIABLES,
        "fraccion_con_drift": fraccion,
        "umbral": umbral,
        "alerta": bool(alerta),
        "reporte_html": str(ruta_html),
    }

    with ruta_json.open("w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, indent=2)

    if alerta:
        print(
            f"ALERTA: drift detectado en {fraccion:.1%} "
            f"de las variables. Umbral: {umbral:.1%}"
        )
    else:
        print(
            f"SIN ALERTA: drift en {fraccion:.1%} "
            f"de las variables. Umbral: {umbral:.1%}"
        )

    print(f"Reporte Evidently: {ruta_html}")
    print(f"Resumen JSON: {ruta_json}")

    return resumen


@flow(name="drift-features-expo", log_prints=True)
def drift_features_flow():
    """Compara el lote de inferencia con el entrenamiento."""
    config = leer_configuracion()
    referencia = cargar_referencia(config)
    lote = cargar_lote(config)

    return analizar_drift(referencia, lote, config)


if __name__ == "__main__":
    drift_features_flow()
