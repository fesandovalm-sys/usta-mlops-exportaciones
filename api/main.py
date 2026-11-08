"""API de inferencia para clasificar exportaciones con el champion de MLflow.

Desde la raiz del repositorio:
    $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    poetry run uvicorn api.main:app --host 127.0.0.1 --port 8000

La API carga el modelo de forma diferida y lo mantiene en memoria. Si cambia
el alias champion, reiniciar Uvicorn para cargar la nueva version.
"""

import os
from functools import lru_cache
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

RAIZ = Path(__file__).resolve().parents[1]
with (RAIZ / "configs" / "config.yaml").open(encoding="utf-8") as archivo:
    CONFIG = yaml.safe_load(archivo)

NOMBRE_MODELO = CONFIG["registro"]["nombre_modelo"]
ALIAS = CONFIG["registro"].get("alias", "champion")
UMBRAL = float(CONFIG["inferencia"]["umbral_clasificacion"])
if not 0.0 <= UMBRAL <= 1.0:
    raise ValueError("umbral_clasificacion debe estar entre 0 y 1")


class Exportacion(BaseModel):
    """Caracteristicas de UNA operacion; no incluye el destino observado."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{
                "capitulo": "09",
                "modad": "1",
                "dpto_origen": "11",
                "pnk": 850.0,
                "fobdol": 6250.0,
            }]
        },
    )

    capitulo: str | None = Field(..., description="Capitulo arancelario; conservar ceros iniciales")
    modad: str | None = Field(..., description="Codigo de modalidad de exportacion")
    dpto_origen: str | None = Field(..., description="Codigo de departamento de origen")
    pnk: float | None = Field(..., allow_inf_nan=False, description="Peso neto en kilogramos")
    fobdol: float | None = Field(..., allow_inf_nan=False, description="Valor FOB en dolares")


class RespuestaPrediccion(BaseModel):
    modelo: str
    alias: str
    version: str
    probabilidad_destino_usa: float
    prediccion_destino_usa: int
    umbral_clasificacion: float


@lru_cache(maxsize=1)
def cargar_champion():
    """Carga una sola vez el modelo vigente, desde su ALIAS (no version fija)."""

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", CONFIG["mlflow"]["tracking_uri"])
    )
    version = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, ALIAS)
    uri = f"models:/{NOMBRE_MODELO}@{ALIAS}"
    modelo = mlflow.sklearn.load_model(uri)

    if not hasattr(modelo, "predict_proba"):
        raise RuntimeError("El champion no permite calcular probabilidades")
    if not hasattr(modelo, "feature_names_in_"):
        raise RuntimeError("El champion no conserva los nombres de las variables")
    if not hasattr(modelo, "classes_") or 1 not in modelo.classes_:
        raise RuntimeError("El champion no contiene la clase positiva 1")

    return modelo, str(version.version)


app = FastAPI(
    title="API MLOps - Exportaciones destino USA",
    description=(
        "Actividad 2, Universidad Santo Tomas. Clasificacion binaria: "
        "probabilidad de que el pais de destino sea Estados Unidos. "
        "La variable destino_usa y los codigos PAIS/COD_PAI4 no se usan como entradas. "
        "Los valores del ejemplo son ilustrativos y no provienen de una operacion real."
    ),
    version="1.0.0",
)


@app.get("/health", summary="Estado de la API y disponibilidad del champion")
def health():
    """La API responde aunque todavia no exista un champion registrado."""

    try:
        _, version = cargar_champion()
    except Exception:
        return {
            "estado": "sin_champion_disponible",
            "modelo": NOMBRE_MODELO,
            "alias": ALIAS,
            "version": None,
        }
    return {
        "estado": "listo",
        "modelo": NOMBRE_MODELO,
        "alias": ALIAS,
        "version": version,
    }


@app.post(
    "/predict",
    response_model=RespuestaPrediccion,
    summary="Predecir si el destino de una exportacion es Estados Unidos",
)
def predict(registro: Exportacion):
    """Recibe una operacion, devuelve la probabilidad y la clasificacion 0/1."""

    try:
        modelo, version = cargar_champion()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Champion no disponible. Inicia MLflow y ejecuta primero el entrenamiento.",
        ) from exc

    datos = pd.DataFrame([{
        nombre: (np.nan if valor is None else valor)
        for nombre, valor in registro.model_dump().items()
    }])
    variables = list(modelo.feature_names_in_)
    faltantes = [columna for columna in variables if columna not in datos.columns]
    if faltantes:
        raise HTTPException(
            status_code=422,
            detail=f"El modelo requiere variables no disponibles: {faltantes}",
        )

    try:
        indice_usa = list(modelo.classes_).index(1)
        probabilidad = float(modelo.predict_proba(datos[variables])[0, indice_usa])
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"No fue posible procesar los valores de entrada: {exc}",
        ) from exc

    if not 0.0 <= probabilidad <= 1.0:
        raise HTTPException(status_code=500, detail="Probabilidad fuera de rango")

    return RespuestaPrediccion(
        modelo=NOMBRE_MODELO,
        alias=ALIAS,
        version=version,
        probabilidad_destino_usa=probabilidad,
        prediccion_destino_usa=int(probabilidad >= UMBRAL),
        umbral_clasificacion=UMBRAL,
    )
