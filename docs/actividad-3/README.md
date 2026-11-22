# Actividad 3 — Monitoreo, drift y CI

**Asignatura:** MLOps · Maestría en Ciencia de Datos, Universidad Santo Tomás

**Integrantes:** Liliana Carolina Herrera Prieto y Fredy Ernesto Sandoval Méndez

**Caso de estudio:** clasificación del destino de exportaciones colombianas (EXPO, DANE/ANDA)

**Documentación:** [Memoria 3](Memoria3.md) · [Guía oficial de la Actividad 3](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/actividad-3.md)

## 1. Objetivo y alcance

Se amplía el sistema de clasificación de la [Actividad 2](../actividad-2/README.md) con simulación reproducible de *data drift*, monitoreo de características (*feature drift*), monitoreo de la variable objetivo (*target drift*) y alertas configurables. El caso de estudio utiliza microdatos EXPO del DANE/ANDA para clasificar si una operación de exportación tiene como destino Estados Unidos (`destino_usa = 1`) u otro país válido (`destino_usa = 0`). En las ejecuciones documentadas se utilizó el modelo Random Forest registrado en MLflow con el alias `champion`, versión 2.

Las cinco variables predictoras son `pnk`, `fobdol`, `capitulo`, `modad` y `dpto_origen`. El período y la variable objetivo no se utilizan como predictores.

## 2. Componentes implementados

| Componente | Ubicación | Función |
|---|---|---|
| Simulación y validaciones estadísticas | [`src/monitoreo.py`](../../src/monitoreo.py) | Simulación reproducible, validación de lotes, chi-cuadrado y extracción de la fracción de drift de Evidently |
| Inferencia | [`flows/inferencia_flow.py`](../../flows/inferencia_flow.py) | Selección de lote, perturbación controlada, predicciones con `champion`, conservación de lotes y registro en MLflow |
| Monitoreo de características | [`flows/drift_features_flow.py`](../../flows/drift_features_flow.py) | Comparación de cinco variables con Evidently y reporte HTML/JSON |
| Monitoreo del objetivo | [`flows/drift_evento_flow.py`](../../flows/drift_evento_flow.py) | Comparación de proporciones con chi-cuadrado y resumen JSON |
| Parámetros | [`configs/config.yaml`](../../configs/config.yaml) | Niveles de simulación, tamaños de lote y referencia, umbrales y rutas |
| Pruebas de monitoreo | [`tests/test_monitoreo.py`](../../tests/test_monitoreo.py) | Pruebas de funciones de drift con datos sintéticos |
| Pruebas de datos | [`tests/test_datos.py`](../../tests/test_datos.py) y [`tests/test_ingesta_anda.py`](../../tests/test_ingesta_anda.py) | Esquema, tipos, rangos, nulos, etiquetas e ingesta |
| Pruebas del modelo | [`tests/test_modelo.py`](../../tests/test_modelo.py) | Selección de variables, entrenamiento y métricas |
| Pruebas de inferencia | [`tests/test_inferencia.py`](../../tests/test_inferencia.py) | Lotes, drift simulado y predicción con modelo sustituido |
| Pruebas de API | [`tests/test_api.py`](../../tests/test_api.py) | Validación de `/health` y `/predict` mediante `TestClient` y *mocks* |
| Pruebas de flujos de drift | [`tests/test_drift_flows.py`](../../tests/test_drift_flows.py) | Alertas, umbrales, manejo de etiquetas y archivos de salida |
| Evidencias | [`evidencias/`](evidencias/) | Catorce capturas de Prefect, Evidently y MLflow |

Se conservan el entrenamiento, la API FastAPI y los servicios de Prefect y MLflow de la Actividad 2.

## 3. Simulación y criterios de monitoreo

La función `aplicar_drift` recibe un nivel entre `0.0` y `1.0`. Se evaluaron los escenarios `0.0`, `0.3` y `0.6`, con semilla `42`. La perturbación desplaza las distribuciones de `pnk` y `fobdol` de forma proporcional al nivel, sin generar valores negativos; asimismo, incrementa de manera controlada la proporción simulada de `destino_usa = 1`. Se conserva `destino_usa_original`, lo que permite distinguir las etiquetas observadas de las utilizadas en la simulación. **Las modificaciones inducidas no representan cambios reales en las exportaciones.** El AUC temporal se evalúa frente a las etiquetas originales.

Ambos flujos de monitoreo utilizan una muestra reproducible de **10.000 registros de entrenamiento** como referencia y lotes de **500 registros de septiembre de 2026**:

- **Características:** Evidently (`DataDriftPreset`) compara las cinco variables. El umbral configurable es `monitoreo.umbral_drift: 0.30`, aplicado al reporte con `DataDriftPreset(drift_share=umbral)`. Se genera alerta cuando la fracción de características con drift supera el 30 %: con cinco variables, esto equivale a detectar drift en al menos dos.
- **Objetivo:** una prueba de chi-cuadrado contrasta la distribución binaria de `destino_usa` en el lote simulado frente a la referencia. Con `monitoreo.alfa_evento: 0.05`, se genera alerta cuando `p < 0.05`. El lote debe contener etiquetas; su ausencia produce un error de validación.

Los flujos guardan los resultados en `data/outputs/reportes_drift/`: reporte HTML y resumen JSON para características; resumen JSON para el objetivo. La inferencia guarda los lotes por nivel y fecha en `data/outputs/lotes/` y actualiza `lote_actual.csv`, utilizado por los flujos de monitoreo.

## 4. Reproducción de las ejecuciones

Desde la raíz del repositorio, con los datos procesados y el modelo `champion` disponible en MLflow (véase [Actividad 2](../actividad-2/README.md)):

```powershell
poetry install --with dev
docker compose -f .\services\docker-compose.yml up -d
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:PYTHONUTF8 = "1"
```

Seleccionar en `configs/config.yaml` el valor `inferencia.nivel_drift` (`0.0`, `0.3` o `0.6`). Ejecutar, en este orden:

```powershell
poetry run python -m flows.inferencia_flow
poetry run python -m flows.drift_features_flow
poetry run python -m flows.drift_evento_flow
```

Repetir la secuencia para cada escenario. Es necesario ejecutar los flujos de monitoreo antes de iniciar la siguiente inferencia, ya que esta reemplaza `data/outputs/lotes/lote_actual.csv`. Los archivos de datos, modelos y resultados locales no se versionan; las capturas de las ejecuciones sí se incluyen en el repositorio.

Si se desea ejecutar adicionalmente la API de inferencia de la Actividad 2, con el modelo `champion` registrado en MLflow, utilizar otra terminal:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
poetry run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

La documentación interactiva de la API se encuentra en `http://127.0.0.1:8000/docs`. Esta ejecución no es necesaria para correr la batería de pruebas automatizadas, que sustituye las dependencias externas cuando corresponde.

## 5. Resultados experimentales

La tasa `destino_usa = 1` de referencia fue **12,90 %** en los tres escenarios. Se obtuvieron los siguientes resultados:

| Nivel simulado | AUC temporal | Características con drift | Tasa USA del lote | Alerta características | Alerta objetivo | Valor p del objetivo |
|---|---:|---:|---:|---|---|---:|
| `0.0` | 0,7464 | 20 % (1/5) | 13,40 % | No | No | 0,797220 |
| `0.3` | 0,7029 | 40 % (2/5) | 22,40 % | Sí | Sí | < 0,000001 |
| `0.6` | 0,6752 | 60 % (3/5) | 31,40 % | Sí | Sí | < 0,000001 |

En `0.0` no hubo perturbación artificial: se conservaron las 67 etiquetas positivas y no se modificó ninguna. Sin embargo, Evidently detectó drift en una variable (20 %), sin activar la alerta global del 30 %. En `0.3` y `0.6`, aumentaron las diferencias detectadas y se activaron ambas alertas. La reducción del AUC corresponde a estos **escenarios de simulación**, no a una medición de desempeño en producción ni, por sí sola, a una demostración causal.

## 6. Evidencias de ejecución

| Escenario | Inferencia Prefect | Drift de características | Drift del objetivo | Reporte HTML de Evidently |
|---|---|---|---|---|
| `0.0` | [01 — Inferencia](evidencias/01_prefect_inferencia_drift_0.png) | [02 — Evidently](evidencias/02_evidently_features_drift_0.png) | [03 — Chi-cuadrado](evidencias/03_prefect_target_drift_0.png) | [10 — Reporte HTML](evidencias/10_evidently_reporte_html_drift_0.png) |
| `0.3` | [04 — Inferencia](evidencias/04_prefect_inferencia_drift_03.png) | [05 — Evidently](evidencias/05_evidently_features_drift_03.png) | [06 — Chi-cuadrado](evidencias/06_prefect_target_drift_03.png) | [11 — Reporte HTML](evidencias/11_evidently_reporte_html_drift_03.png) |
| `0.6` | [07 — Inferencia](evidencias/07_prefect_inferencia_drift_06.png) | [08 — Evidently](evidencias/08_evidently_features_drift_06.png) | [09 — Chi-cuadrado](evidencias/09_prefect_target_drift_06.png) | [12 — Reporte HTML](evidencias/12_evidently_reporte_html_drift_06.png) |

Las nueve ejecuciones definitivas finalizaron en estado `Completed`. Las doce primeras capturas documentan esas ejecuciones y los reportes HTML regenerados después de alinear el umbral de Evidently con el 30 % establecido en la configuración. El [PR #19](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/19) añadió las evidencias de MLflow: [13 — Parámetros y métricas de inferencia](evidencias/13_mlflow_inferencia_parametros_metricas.png) y [14 — Artefactos registrados](evidencias/14_mlflow_inferencia_artefactos.png). **Se encuentran publicadas 14 evidencias en total.**

## 7. Pruebas, alcance y trazabilidad

```powershell
poetry run python -m pytest -q
```

La batería ampliada mediante el PR #19 registró **95 pruebas automatizadas aprobadas**, sin requerir servidores MLflow o Prefect durante su ejecución. Las pruebas de API utilizan `TestClient` y *mocks*; las de inferencia y monitoreo aíslan funciones y simulan servicios cuando corresponde.

| Grupo de pruebas | Cantidad | Archivos |
|---|---:|---|
| Ingesta, datos y monitoreo inicial | 44 | `test_ingesta_anda.py`, `test_datos.py`, `test_monitoreo.py` |
| Modelo | 15 | `test_modelo.py` |
| Inferencia | 9 | `test_inferencia.py` |
| API FastAPI | 15 | `test_api.py` |
| Flujos de drift | 12 | `test_drift_flows.py` |
| **Total** | **95** | |

Se incluyen pruebas de validación, reproducción de lotes, detección de errores, alertas, generación de reportes y respuestas de API. **No se implementó GitHub Actions**, dado que la guía considera opcional la CI.

- [PR #16](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/16): simulación, validaciones y pruebas iniciales.
- [PR #17](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/17): flujos de inferencia y monitoreo, configuración y evidencias.
- [PR #18](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/18): aplicación del umbral configurado al reporte de Evidently.
- [PR #19](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/19): ampliación de 95 pruebas, corrección de ejecución redundante de Evidently y evidencias de MLflow.

## Referencias

- [Memoria 3](Memoria3.md).
- [Actividad 2: documentación](../actividad-2/README.md).
- [Guía oficial de la Actividad 3](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/actividad-3.md).
- [Plantilla de memoria](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/plantilla-memoria.md).
