# Actividad 2 — Pipeline y empaquetado

**Curso:** MLOps · Maestría en Ciencia de Datos, Universidad Santo Tomás  
**Integrantes:** Fredy Ernesto Sandoval Méndez y Liliana Carolina Herrera Prieto  
**Repositorio:** [usta-mlops-exportaciones](https://github.com/fesandovalm-sys/usta-mlops-exportaciones)  
**Guía:** [Actividad 2 — Henry Cala](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/actividad-2.md)

## Objetivo y alcance

Implementar un flujo MLOps para estimar la **probabilidad de que una operación de exportación colombiana tenga como destino Estados Unidos**, a partir de microdatos públicos EXPO del DANE/ANDA. La variable objetivo es `destino_usa` (1: Estados Unidos; 0: otros destinos válidos).

La solución incorpora preparación de datos, entrenamiento y evaluación con Scikit-learn, orquestación mediante Prefect, seguimiento y registro de modelos en MLflow y predicción con FastAPI. La promoción del alias **`champion`** exige superar tanto el AUC mínimo configurado como el AUC del modelo vigente. La API utiliza el alias, sin fijar un número de versión en el código.

## Archivos principales

| Componente | Ruta |
|---|---|
| Ingesta y validación de datos | [`src/ingesta_anda.py`](../../src/ingesta_anda.py) |
| Entrenamiento y promoción | [`flows/entrenamiento_flow.py`](../../flows/entrenamiento_flow.py) |
| Inferencia y registro de predicciones | [`flows/inferencia_flow.py`](../../flows/inferencia_flow.py) |
| Despliegues de Prefect | [`flows/desplegar.py`](../../flows/desplegar.py) |
| Modelos y métricas | [`src/modelo.py`](../../src/modelo.py) |
| Servicio de predicción | [`api/main.py`](../../api/main.py) |
| Parámetros y selección de variables | [`configs/config.yaml`](../../configs/config.yaml) y [`configs/features.yaml`](../../configs/features.yaml) |
| Pruebas de ingesta | [`tests/test_ingesta_anda.py`](../../tests/test_ingesta_anda.py) |
| Evidencias de ejecución | [`evidencias/`](evidencias/) |
| Memoria | [`Memoria2.md`](Memoria2.md) · [PDF](Memoria2.pdf) |

## 1. Requisitos y servicios

Trabajar desde la **raíz del repositorio** en Windows/PowerShell. Requiere Python 3.12 o una versión compatible con `pyproject.toml`, Poetry 2.x y los servicios de Prefect y MLflow. La instalación inicial y las alternativas de puesta en marcha se describen en el [README principal](../../README.md).

```powershell
poetry install
```

Para iniciar los servicios con Docker Desktop:

```powershell
docker compose -f .\services\docker-compose.yml up -d
docker compose -f .\services\docker-compose.yml ps
```

Interfaces: **Prefect** `http://127.0.0.1:4200` y **MLflow** `http://127.0.0.1:5000`. Usar una sola modalidad de servidores a la vez, evitando ejecutar servidores locales y Docker simultáneamente en los mismos puertos.

En las terminales donde se ejecuten los flujos:

```powershell
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:PYTHONUTF8 = "1"
```

## 2. Preparar los microdatos

Los datos originales provienen de [DANE/ANDA — microdatos EXPO](https://microdatos.dane.gov.co/index.php/catalog/859/get-microdata). Los archivos originales y procesados **no se versionan** en GitHub.

Ubicar los ZIP originales en:

```text
data/raw/Expo_2025.zip
data/raw/Expo_2026.zip
```

Ejecutar:

```powershell
poetry run python src/ingesta_anda.py
```

El proceso valida los 14 meses comprendidos entre agosto de 2025 y septiembre de 2026 y genera:

```text
data/processed/train_202508_202605.csv.gz
data/processed/valid_202606_202608.csv.gz
data/processed/sept_202609.csv.gz
data/processed/reporte_ingesta.json
```

Los conjuntos contienen `periodo`, `capitulo`, `modad`, `dpto_origen`, `pnk`, `fobdol` y `destino_usa`. **Ni el período ni la etiqueta son variables predictoras**; tampoco se utilizan los identificadores directos del país. Las reglas y los resultados de ingesta se detallan en [`fuentes_y_datos.md`](fuentes_y_datos.md).

## 3. Configurar y ejecutar el entrenamiento

Ajustar los parámetros en [`configs/config.yaml`](../../configs/config.yaml) y la selección de variables en [`configs/features.yaml`](../../configs/features.yaml): `ALL`, `NUMERIC` o `CUSTOM`. El proyecto compara candidatos de regresión logística y Random Forest mediante AUC-ROC de validación.

Con Prefect y MLflow activos y los conjuntos de datos preparados:

```powershell
poetry run python -m flows.entrenamiento_flow
```

La ejecución realiza ingesta/validación, entrenamiento, evaluación y registro de runs en MLflow con el tag `fase = entrenamiento` y los parámetros `features` y `n_features`. El mejor candidato **solo obtiene una nueva versión registrada y el alias `champion`** cuando su AUC supera el umbral y el AUC del `champion` anterior. Los modelos no promovidos quedan como runs de experimentación.

## 4. Ejecutar la inferencia y los despliegues

Para una ejecución manual de inferencia:

```powershell
poetry run python -m flows.inferencia_flow
```

La inferencia carga `models:/expo-destino-usa@champion`, selecciona un lote del conjunto temporal y guarda predicciones fechadas en `data/outputs/`. Registra asimismo un run de MLflow con `fase = inferencia` y la versión empleada.

Para publicar los dos despliegues de Prefect:

```powershell
poetry run python -m flows.desplegar
```

El programa registra `entrenamiento-manual` y `inferencia-diaria`. El horario predeterminado de la inferencia es **21:00, zona America/Bogota**, modificable mediante YAML. **Mantener activa la terminal del proceso `serve`** para que se ejecuten los despliegues programados.

## 5. Probar FastAPI

Con MLflow disponible y un modelo `champion` registrado, abrir otra terminal en la raíz:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
poetry run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000/docs` para acceder a la documentación interactiva y probar `GET /health` y `POST /predict`. Ejemplo de cuerpo de solicitud:

```json
{
  "capitulo": "09",
  "modad": "1",
  "dpto_origen": "11",
  "pnk": 850.0,
  "fobdol": 6250.0
}
```

La respuesta de `/predict` contiene el nombre del modelo, el alias, la **versión**, la probabilidad y la predicción binaria. La API conserva el modelo en memoria: **al promover un nuevo `champion`, reiniciar Uvicorn** y repetir `/health` y `/predict` para comprobar que se utiliza la nueva versión **sin modificar el código**.

## 6. Pruebas y evidencias

```powershell
poetry run python -m pytest -q
```

Las 21 pruebas del módulo de ingesta se ejecutaron satisfactoriamente. La evaluación realizada con datos de validación registró un AUC aproximado de **0,7160** para la regresión logística promovida inicialmente como `champion` v1 y de **0,729074** para Random Forest. Las evidencias de la promoción posterior y la inferencia con el nuevo modelo están disponibles en [`evidencias/`](evidencias/), entre ellas:

- [`14_prefect_entrenamiento_random_forest.png`](evidencias/14_prefect_entrenamiento_random_forest.png) — entrenamiento con el nuevo candidato.
- [`15_mlflow_champion_v2.png`](evidencias/15_mlflow_champion_v2.png) — alias `champion` asociado con la versión 2.
- [`16_fastapi_health_champion_v2.png`](evidencias/16_fastapi_health_champion_v2.png) y [`17_fastapi_predict_champion_v2.png`](evidencias/17_fastapi_predict_champion_v2.png) — comprobación de la nueva versión desde FastAPI.
- [`11_prefect_inferencia_programada.png`](evidencias/11_prefect_inferencia_programada.png) — programación diaria.
- [`18_prefect_inferencia_champion_v2.png`](evidencias/18_prefect_inferencia_champion_v2.png) y [`20_mlflow_inferencia_champion_v2.png`](evidencias/20_mlflow_inferencia_champion_v2.png) — inferencia y registro en MLflow.

**Nota de reproducibilidad:** los modelos, experimentos y ejecuciones de los servidores locales que se realizaron desde el equipo de Fredy Sandoval no se sincronizan por Git. Un clon nuevo debe preparar sus datos y ejecutar el entrenamiento sobre su propio servidor MLflow. El código y las capturas versionados permiten comprobar el diseño y consultar las evidencias del equipo sin transferir sus bases locales.

Para decisiones metodológicas, dificultades, análisis y contribuciones individuales, consultar la [Memoria 2](Memoria2.md).
