
"""Actividad 3: deteccion de drift del target destino_usa."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from prefect import flow, task

from src.modelo import OBJETIVO
from src.monitoreo import drift_de_target

RAIZ = Path(__file__).resolve().parents[1]


def leer_configuracion():
    with (RAIZ / "configs/config.yaml").open(
        encoding="utf-8"
    ) as archivo:
        return yaml.safe_load(archivo)


def leer_target(ruta):
    """Carga etiquetas binarias y rechaza archivos sin target."""
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe: {ruta}")

    try:
        datos = pd.read_csv(ruta, usecols=[OBJETIVO])
    except ValueError as exc:
        raise ValueError(
            f"El lote no contiene la columna '{OBJETIVO}'"
        ) from exc

    if datos.empty:
        raise ValueError(f"Archivo sin registros: {ruta}")

    return datos[OBJETIVO]


@task(name="referencia-drift-target")
def cargar_referencia(config):
    ruta = RAIZ / config["datos"]["entrenamiento"]
    referencia = leer_target(ruta)

    tamano = int(config["monitoreo"]["tamano_referencia"])
    if tamano < 1:
        raise ValueError("tamano_referencia debe ser positivo")

    referencia = referencia.sample(
        n=min(tamano, len(referencia)),
        random_state=42,
    ).reset_index(drop=True)

    print(f"Referencia target: {len(referencia)} registros")
    return referencia


@task(name="lote-drift-target")
def cargar_lote(config):
    ruta = RAIZ / config["monitoreo"]["lote_actual"]
    lote = leer_target(ruta)

    print(f"Lote target: {len(lote)} registros")
    return lote


@task(name="evaluar-y-alertar-drift-target")
def evaluar_drift(referencia, lote, config):
    alfa = float(config["monitoreo"]["alfa_evento"])

    resultado = drift_de_target(
        referencia, lote, alfa=alfa
    )

    nivel = float(config["inferencia"]["nivel_drift"])
    carpeta = RAIZ / config["monitoreo"]["carpeta_reportes"]
    carpeta.mkdir(parents=True, exist_ok=True)

    marca = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ruta = carpeta / f"drift_target_{nivel:.1f}_{marca}.json"

    resumen = {
        "nivel_drift_simulado": nivel,
        "n_referencia": len(referencia),
        "n_lote": len(lote),
        "target": OBJETIVO,
        "alfa": alfa,
        **resultado,
    }

    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, indent=2)

    print(
        f"Tasa de destino USA: "
        f"referencia={resultado['tasa_referencia']:.2%}; "
        f"lote={resultado['tasa_lote']:.2%}"
    )
    print(f"p-valor: {resultado['p_valor']:.6f}")

    if resultado["drift"]:
        print("ALERTA: drift estadistico del target detectado")
    else:
        print("SIN ALERTA: drift estadistico del target no detectado")

    print(f"Resumen target: {ruta}")
    return resumen


@flow(name="drift-evento-expo", log_prints=True)
def drift_evento_flow():
    config = leer_configuracion()
    referencia = cargar_referencia(config)
    lote = cargar_lote(config)
    return evaluar_drift(referencia, lote, config)


if __name__ == "__main__":
    drift_evento_flow()
