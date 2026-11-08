"""Actividad 2: entrenamiento supervisado, evaluación y promoción en MLflow.

Ejecutar desde la raíz del repositorio (PowerShell):
  $env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
  $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
  $env:PYTHONUTF8 = "1"
  poetry run python -m flows.entrenamiento_flow

Requiere los CSV preparados por la ingesta de Liliana.
"""

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, task

from src.modelo import (
    CATEGORICAS,
    NUMERICAS,
    OBJETIVO,
    calcular_metricas,
    construir_modelo,
    resolver_variables,
)

RAIZ = Path(__file__).resolve().parents[1]


def leer_yaml(nombre):
    with (RAIZ / "configs" / nombre).open(encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


@task(name="ingesta-y-validacion")
def cargar_datos(config):
    """Lee ambos conjuntos, valida etiquetas y verifica la separación temporal."""
    columnas = ["periodo", *NUMERICAS, *CATEGORICAS, OBJETIVO]
    conjuntos = []
    rangos = [("entrenamiento", "2025-08", "2026-05"),
              ("validacion", "2026-06", "2026-08")]

    for clave, inicio, fin in rangos:
        ruta = RAIZ / config["datos"][clave]
        if not ruta.is_file():
            raise FileNotFoundError(
                f"No existe {ruta}. Primero ejecuta la ingesta de datos de Lili."
            )

        # dtype evita perder ceros iniciales en codigos categoricos.
        df = pd.read_csv(
            ruta,
            dtype={"periodo": str, **{c: str for c in CATEGORICAS}},
        )
        faltan = sorted(set(columnas) - set(df.columns))
        if faltan:
            raise ValueError(f"Faltan columnas en {ruta.name}: {faltan}")
        df = df[columnas].copy()

        if df.empty:
            raise ValueError(f"El conjunto {clave} esta vacio")
        periodos = df["periodo"].astype(str).str.strip()
        if not periodos.str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all():
            raise ValueError(f"Formato de periodo invalido en {clave}; esperado YYYY-MM")
        if not periodos.between(inicio, fin).all():
            raise ValueError(f"Se encontraron periodos fuera de {inicio} a {fin}")

        for columna in NUMERICAS:
            df[columna] = pd.to_numeric(df[columna], errors="coerce")
            if not np.isfinite(df[columna].dropna().to_numpy(dtype=float)).all():
                raise ValueError(f"Valores numericos no finitos en {columna}")
        for columna in CATEGORICAS:
            # Compatible con SimpleImputer(missing_values=np.nan).
            df[columna] = df[columna].astype(object)
            df[columna] = df[columna].where(df[columna].notna(), np.nan)

        df[OBJETIVO] = pd.to_numeric(df[OBJETIVO], errors="coerce")
        if not df[OBJETIVO].isin([0, 1]).all():
            raise ValueError(f"{OBJETIVO} debe contener solo 0 y 1 en {clave}")
        df[OBJETIVO] = df[OBJETIVO].astype(int)
        if df[OBJETIVO].nunique() != 2:
            raise ValueError(f"Se necesitan ambas clases para AUC en {clave}")

        print(f"{clave}: {len(df):,} registros; USA={df[OBJETIVO].mean():.2%}")
        conjuntos.append(df)

    return tuple(conjuntos)


@task(name="entrenar-modelo")
def entrenar_modelo(candidato, numericas, categoricas, train):
    variables = numericas + categoricas
    modelo = construir_modelo(
        candidato["tipo"], candidato["parametros"], numericas, categoricas
    )
    modelo.fit(train[variables], train[OBJETIVO])
    return modelo


@task(name="evaluar-modelo")
def evaluar_modelo(modelo, valid, variables, umbral):
    probabilidades = modelo.predict_proba(valid[variables])[:, 1]
    return calcular_metricas(valid[OBJETIVO], probabilidades, umbral)


@task(name="registrar-experimento")
def registrar_experimento(modelo, candidato, config, features, metricas, valid, variables, n_train):
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    )
    mlflow.set_experiment(config["mlflow"]["experimento"])
    with mlflow.start_run(run_name=candidato["nombre"]) as run:
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.log_params({
            "modelo": candidato["tipo"],
            "candidato": candidato["nombre"],
            "features": str(features["seleccion"]).upper(),
            "n_features": len(variables),
            "filas_entrenamiento": n_train,
            "filas_validacion": len(valid),
            **candidato["parametros"],
        })
        mlflow.log_metrics(metricas)
        ejemplo = valid[variables].head(3)
        
        info = mlflow.sklearn.log_model(
            sk_model=modelo,
            name="model",
            input_example=ejemplo,
            signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
            serialization_format="skops",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        run_id = run.info.run_id
    return {"run_id": run_id, "model_uri": info.model_uri,
            "auc": metricas["auc"], "candidato": candidato["nombre"]}


@task(name="promover-champion")
def promover_champion(mejor, config):
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    )
    client = MlflowClient()
    nombre = config["registro"]["nombre_modelo"]
    alias = config["registro"].get("alias", "champion")
    minimo = float(config["registro"]["umbral_auc"])
    auc = mejor["auc"]

    if auc <= minimo:
        print(f"AUC {auc:.4f} no supera minimo {minimo:.4f}; no se promueve")
        return None

    try:
        actual = client.get_model_version_by_alias(nombre, alias)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        actual = None

    if actual is not None:
        auc_actual = client.get_run(actual.run_id).data.metrics["auc"]
        if auc <= auc_actual:
            print(f"AUC {auc:.4f} <= champion {auc_actual:.4f}; alias sin cambios")
            return actual.version

    version = mlflow.register_model(mejor["model_uri"], nombre)
    client.set_registered_model_alias(nombre, alias, version.version)
    client.set_model_version_tag(nombre, version.version, "estado", "aprobado")
    print(f"Champion actualizado: {nombre}, version {version.version}, AUC {auc:.4f}")
    return version.version


@flow(name="entrenamiento-expo-destino-usa", log_prints=True)
def entrenamiento_flow():
    config = leer_yaml("config.yaml")
    features = leer_yaml("features.yaml")
    numericas, categoricas = resolver_variables(features)
    variables = numericas + categoricas
    train, valid = cargar_datos(config)

    resultados = []
    for candidato in config["modelos"]:
        modelo = entrenar_modelo(candidato, numericas, categoricas, train)
        metricas = evaluar_modelo(
            modelo, valid, variables,
            config["inferencia"]["umbral_clasificacion"],
        )
        resultado = registrar_experimento(
            modelo, candidato, config, features, metricas, valid,
            variables, len(train),
        )
        print(f"{candidato['nombre']}: AUC={metricas['auc']:.4f}, "
              f"F1={metricas['f1']:.4f}")
        resultados.append(resultado)

    mejor = max(resultados, key=lambda r: r["auc"])
    version = promover_champion(mejor, config)
    print(f"Mejor: {mejor['candidato']} (AUC={mejor['auc']:.4f}); "
          f"version champion: {version}")
    return mejor


if __name__ == "__main__":
    entrenamiento_flow()
