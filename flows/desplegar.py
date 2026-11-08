"""Actividad 2: registrar dos despliegues locales de Prefect.

Desde la raiz del repositorio (PowerShell):
    $env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    $env:PYTHONUTF8 = "1"
    poetry run python -m flows.desplegar

Este proceso debe permanecer activo para ejecutar los despliegues.
"""

from pathlib import Path

import yaml
from prefect import serve
from prefect.schedules import Cron

from flows.entrenamiento_flow import entrenamiento_flow
from flows.inferencia_flow import inferencia_flow

RAIZ = Path(__file__).resolve().parents[1]


def obtener_programacion():
    """Lee el cron y la zona horaria desde la configuracion externa."""
    with (RAIZ / "configs" / "config.yaml").open(encoding="utf-8") as archivo:
        config = yaml.safe_load(archivo)

    despliegues = config.get("despliegues", {})
    cron = despliegues.get("inferencia_cron", "0 21 * * *")
    zona = despliegues.get("zona_horaria", "America/Bogota")
    return cron, zona


def crear_despliegues():
    """Entrenamiento manual e inferencia diaria por hora local."""
    cron, zona = obtener_programacion()

    entrenamiento = entrenamiento_flow.to_deployment(
        name="entrenamiento-manual",
        description="Entrenamiento y evaluacion del clasificador EXPO; ejecucion manual.",
    )
    inferencia = inferencia_flow.to_deployment(
        name="inferencia-diaria",
        description=f"Inferencia diaria con champion MLflow, cron {cron} ({zona}).",
        schedules=[Cron(cron, timezone=zona)],
    )
    return entrenamiento, inferencia


if __name__ == "__main__":
    entrenamiento, inferencia = crear_despliegues()
    print("Registrando dos despliegues en Prefect...")
    print("Entrenamiento: manual, sin horario")
    cron, zona = obtener_programacion()
    print(f"Inferencia: cron={cron}, zona_horaria={zona}")
    print("Mantener esta terminal abierta para que se ejecuten los despliegues.")
    serve(entrenamiento, inferencia)
