"""Actividad 2: inferencia por lote con el modelo champion de MLflow.

Desde la raiz del repositorio, con Prefect y MLflow encendidos:
    $env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    $env:PYTHONUTF8 = "1"
    poetry run python -m flows.inferencia_flow

Antes debe existir un modelo registrado con alias champion.
"""

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from prefect import flow, task

from src.modelo import CATEGORICAS, NUMERICAS, OBJETIVO
from src.monitoreo import aplicar_drift

RAIZ = Path(__file__).resolve().parents[1]
ZONA_HORARIA = ZoneInfo("America/Bogota")


def leer_configuracion():
    with (RAIZ / "configs" / "config.yaml").open(encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def configurar_mlflow(config):
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    )


@task(name="ingesta-lote-inferencia")
def cargar_lote(config, fecha_ejecucion):
    """Lee el conjunto temporal y escoge un lote reproducible por dia."""
    ruta = RAIZ / config["datos"]["inferencia"]
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero src/ingesta_anda.py "
            "con los archivos del ANDA preparados por Lili."
        )

    columnas_requeridas = ["periodo", *NUMERICAS, *CATEGORICAS, OBJETIVO]
    datos = pd.read_csv(
        ruta, dtype={"periodo": str, **{c: str for c in CATEGORICAS}}
    )
    faltan = sorted(set(columnas_requeridas) - set(datos.columns))
    if faltan:
        raise ValueError(f"Columnas faltantes en {ruta.name}: {faltan}")
    if datos.empty:
        raise ValueError("El archivo de inferencia esta vacio")

    periodos = datos["periodo"].astype(str).str.strip()
    if not periodos.str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all():
        raise ValueError("El periodo debe tener el formato YYYY-MM")
    datos["periodo"] = periodos

    for columna in NUMERICAS:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")
        if np.isinf(datos[columna].dropna().to_numpy(dtype=float)).any():
            raise ValueError(f"Valores infinitos en {columna}")

    for columna in CATEGORICAS:
        datos[columna] = datos[columna].astype(object)
        datos[columna] = datos[columna].where(datos[columna].notna(), np.nan)

    tamano = int(config["inferencia"]["tamano_lote"])
    if tamano < 1:
        raise ValueError("tamano_lote debe ser mayor que cero")

    # Mismo lote dentro del dia; en dias distintos se prueba otra muestra.
    semilla = int(fecha_ejecucion.strftime("%Y%m%d"))
    lote = datos.sample(n=min(tamano, len(datos)), random_state=semilla)
    print(
        f"Ingesta: {len(datos):,} filas; lote={len(lote)}; "
        f"periodos={', '.join(sorted(lote['periodo'].unique()))}"
    )
    return lote.reset_index(drop=True)


@task(name="simular-y-guardar-lote-drift")
def simular_y_guardar_lote(lote, config, fecha_ejecucion):
    """Aplica drift y conserva el lote usado en inferencia."""
    nivel = float(config["inferencia"]["nivel_drift"])
    semilla = int(config["inferencia"]["semilla_drift"])

    if not 0.0 <= nivel <= 1.0:
        raise ValueError("nivel_drift debe estar entre 0 y 1")

    lote_simulado = aplicar_drift(
        lote, nivel=nivel, semilla=semilla
    )

    if "destino_usa_original" not in lote_simulado.columns:
        raise ValueError(
            "La simulacion debe conservar destino_usa_original"
        )

    carpeta = RAIZ / config["monitoreo"]["carpeta_lotes"]
    carpeta.mkdir(parents=True, exist_ok=True)

    marca = fecha_ejecucion.strftime("%Y%m%d_%H%M%S_%f")
    ruta = carpeta / f"lote_drift_{nivel:.1f}_{marca}.csv"
    lote_simulado.to_csv(
        ruta, index=False, encoding="utf-8-sig"
    )

    ruta_actual = RAIZ / config["monitoreo"]["lote_actual"]
    ruta_actual.parent.mkdir(parents=True, exist_ok=True)
    lote_simulado.to_csv(
        ruta_actual, index=False, encoding="utf-8-sig"
    )

    print(
        f"Drift emulado: nivel={nivel:.1f}; "
        f"registros={len(lote_simulado)}; archivo={ruta}"
    )

    return lote_simulado, str(ruta)


@task(name="cargar-y-aplicar-champion")
def predecir_con_champion(lote, config, fecha_ejecucion):
    """Carga por alias (nunca por version fija) y estima probabilidades."""
    configurar_mlflow(config)
    nombre = config["registro"]["nombre_modelo"]
    alias = config["registro"].get("alias", "champion")
    uri = f"models:/{nombre}@{alias}"

    client = MlflowClient()
    try:
        version_modelo = client.get_model_version_by_alias(nombre, alias)
    except MlflowException as exc:
        raise RuntimeError(
            f"No existe un {alias} disponible para '{nombre}'. "
            "Ejecuta y valida primero el flujo de entrenamiento."
        ) from exc

    modelo = mlflow.sklearn.load_model(uri)
    if not hasattr(modelo, "feature_names_in_"):
        raise ValueError(
            "El modelo no expone feature_names_in_. "
            "Verifica el entrenamiento usando un DataFrame con nombres de columnas."
        )

    # Se usa el contrato guardado en el modelo, no una configuracion que
    # pudo haber cambiado despues de su entrenamiento.
    variables = list(modelo.feature_names_in_)
    faltan = [c for c in variables if c not in lote.columns]
    if faltan:
        raise ValueError(f"Faltan variables exigidas por champion: {faltan}")

    probabilidades = modelo.predict_proba(lote[variables])[:, 1]
    umbral = float(config["inferencia"]["umbral_clasificacion"])
    if not 0 <= umbral <= 1:
        raise ValueError("umbral_clasificacion debe estar entre 0 y 1")

    salida = lote[["periodo", *variables]].copy()
    salida["probabilidad_destino_usa"] = probabilidades
    salida["prediccion_destino_usa"] = (probabilidades >= umbral).astype(int)
    salida["modelo_version"] = str(version_modelo.version)
    salida["fecha_ejecucion_bogota"] = fecha_ejecucion.isoformat()

    # Etiqueta de septiembre SOLO para comprobacion retrospectiva,
    # nunca incluida en las variables usadas para la prediccion.
    if OBJETIVO in lote.columns:
        salida["destino_usa_simulado"] = lote[OBJETIVO].to_numpy()

    if "destino_usa_original" in lote.columns:
        salida["destino_usa_observado"] = (
            lote["destino_usa_original"].to_numpy()
        )

    salida["nivel_drift"] = float(
        config["inferencia"]["nivel_drift"]
    )

    print(
        f"MLflow: {uri}, version {version_modelo.version}; "
        f"predicciones={len(salida)}"
    )
    return salida, str(version_modelo.version), variables


@task(name="guardar-y-registrar-inferencia")
def guardar_y_registrar(
    salida, version, variables, config, fecha_ejecucion, ruta_lote
):
    """Guarda CSV fechado y registra el lote como run en MLflow."""
    configurar_mlflow(config)
    carpeta = RAIZ / config["inferencia"]["carpeta_salida"]
    carpeta.mkdir(parents=True, exist_ok=True)
    marca = fecha_ejecucion.strftime("%Y%m%d_%H%M%S_%f")
    ruta = carpeta / f"predicciones_destino_usa_{marca}.csv"
    salida.to_csv(ruta, index=False, encoding="utf-8-sig")

    mlflow.set_experiment(config["mlflow"]["experimento"])
    with mlflow.start_run(run_name=f"inferencia_{marca}") as run:
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("alias", config["registro"].get("alias", "champion"))
        mlflow.log_params({
            "model_name": config["registro"]["nombre_modelo"],
            "model_version": version,
            "n_features": len(variables),
            "features": ",".join(variables),
            "tamano_lote": len(salida),
            "umbral_clasificacion": config["inferencia"]["umbral_clasificacion"],
            "nivel_drift": config["inferencia"]["nivel_drift"],
            "semilla_drift": config["inferencia"]["semilla_drift"],
        })
        mlflow.log_metric("n_predicciones", len(salida))
        mlflow.log_metric(
            "proporcion_predicha_usa", salida["prediccion_destino_usa"].mean()
        )

        # La metrica temporal no afecta el champion: septiembre es posterior.
        if "destino_usa_observado" in salida.columns:
            from sklearn.metrics import roc_auc_score
            reales = pd.to_numeric(salida["destino_usa_observado"], errors="coerce")
            validos = reales.isin([0, 1])
            if reales[validos].nunique() == 2:
                auc_temporal = roc_auc_score(
                    reales[validos].astype(int),
                    salida.loc[validos, "probabilidad_destino_usa"],
                )
                mlflow.log_metric("auc_temporal_septiembre", float(auc_temporal))
                print(f"AUC temporal (solo evaluacion): {auc_temporal:.4f}")

        mlflow.log_artifact(str(ruta), artifact_path="predicciones")
        mlflow.log_artifact(
            ruta_lote, artifact_path="lotes_monitoreo"
        )
        run_id = run.info.run_id

    print(f"CSV: {ruta}")
    print(f"Run MLflow de inferencia: {run_id}")
    return {"archivo": str(ruta), "model_version": version, "run_id": run_id}




@flow(name="inferencia-expo-destino-usa", log_prints=True)
def inferencia_flow():
    """Inferencia con drift emulado y trazabilidad en MLflow."""
    config = leer_configuracion()
    fecha = datetime.now(ZONA_HORARIA)

    lote_original = cargar_lote(config, fecha)

    lote_simulado, ruta_lote = simular_y_guardar_lote(
        lote_original, config, fecha
    )

    salida, version, variables = predecir_con_champion(
        lote_simulado, config, fecha
    )

    return guardar_y_registrar(
        salida, version, variables,
        config, fecha, ruta_lote
    )


if __name__ == "__main__":
    inferencia_flow()