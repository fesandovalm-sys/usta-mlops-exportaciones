# Memoria 3 — Monitoreo, drift y CI

**Asignatura:** MLOps · Maestría en Ciencia de Datos, Universidad Santo Tomás

**Integrantes:** Liliana Carolina Herrera Prieto y Fredy Ernesto Sandoval Méndez

**Caso de estudio:** clasificación del destino de exportaciones colombianas (EXPO, DANE/ANDA)

## 1. Objetivo

Ampliar el sistema de clasificación de la Actividad 2 mediante simulación reproducible de cambios en los datos, monitoreo de características (*feature drift*) y de la variable objetivo (*target drift*), alertas configurables y pruebas automatizadas. Se conserva el objetivo binario `destino_usa` (1: Estados Unidos; 0: otros destinos válidos) y el modelo identificado con el alias `champion` en MLflow.

## 2. Descripción del trabajo

Se implementó `src/monitoreo.py` con validación de lotes, emulación de drift, prueba de chi-cuadrado para el objetivo y extracción de resultados de Evidently. La lógica se mantiene independiente de Prefect y MLflow para facilitar las pruebas. El flujo `flows/inferencia_flow.py` selecciona lotes de 500 registros de septiembre de 2026, aplica niveles configurables de perturbación (`0.0`, `0.3` y `0.6`), conserva `destino_usa_original` y registra predicciones y artefactos con el modelo Random Forest `champion` v2.

Dos flujos de Prefect complementan la inferencia: `flows/drift_features_flow.py` compara las cinco variables predictoras con una muestra reproducible de 10.000 registros de entrenamiento mediante Evidently; `flows/drift_evento_flow.py` contrasta las proporciones de `destino_usa` mediante chi-cuadrado. Los flujos generan reportes y registros de alertas. Las nueve ejecuciones de inferencia y monitoreo de los tres escenarios terminaron en estado `Completed`.

## 3. Decisiones técnicas

La emulación modifica las distribuciones de `pnk` y `fobdol` y, de forma controlada, la proporción del objetivo; utiliza semilla `42` para reproducibilidad y preserva las etiquetas originales. Los cambios son **artificiales** y no representan una variación observada de las exportaciones.

Se utilizó Evidently por ser la herramienta indicada en la guía, sin comparación experimental con alternativas. El contraste chi-cuadrado se seleccionó por la naturaleza binaria del objetivo y los tamaños de las muestras; se aplicó una significancia de `α = 0,05`, sin evaluación formal de otras pruebas estadísticas.

El umbral global de características se fijó en **30 %** como criterio operativo configurable: con cinco variables, una con drift equivale al 20 % y dos al 40 %, por lo que se alerta a partir de dos variables afectadas. No se realizó optimización estadística de este valor. La configuración se aplica tanto a la alerta como al reporte con `DataDriftPreset(drift_share=umbral)`.

## 4. Evidencias

El código de monitoreo y las pruebas iniciales se integraron mediante el [PR #16](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/16); los flujos y nueve capturas, mediante el [PR #17](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/17); y el ajuste del umbral con tres reportes HTML adicionales, mediante el [PR #18](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/18).

| Nivel | Inferencia (Prefect) | Características (Evidently) | Objetivo (Prefect) | Reporte HTML |
|---|---|---|---|---|
| `0.0` | [01](evidencias/01_prefect_inferencia_drift_0.png) | [02](evidencias/02_evidently_features_drift_0.png) | [03](evidencias/03_prefect_target_drift_0.png) | [10](evidencias/10_evidently_reporte_html_drift_0.png) |
| `0.3` | [04](evidencias/04_prefect_inferencia_drift_03.png) | [05](evidencias/05_evidently_features_drift_03.png) | [06](evidencias/06_prefect_target_drift_03.png) | [11](evidencias/11_evidently_reporte_html_drift_03.png) |
| `0.6` | [07](evidencias/07_prefect_inferencia_drift_06.png) | [08](evidencias/08_evidently_features_drift_06.png) | [09](evidencias/09_prefect_target_drift_06.png) | [12](evidencias/12_evidently_reporte_html_drift_06.png) |

## 5. Resultados

Se evaluaron lotes de **500 registros** contra **10.000 registros de referencia**. La tasa de `destino_usa = 1` en la referencia fue **12,90 %**.

| Nivel | AUC temporal | Características con drift | Tasa USA del lote | Valor p del objetivo | Alerta características | Alerta objetivo |
|---|---:|---:|---:|---:|---|---|
| `0.0` | 0,7464 | 20 % (1/5) | 13,40 % | 0,797220 | No | No |
| `0.3` | 0,7029 | 40 % (2/5) | 22,40 % | < 0,000001 | Sí | Sí |
| `0.6` | 0,6752 | 60 % (3/5) | 31,40 % | < 0,000001 | Sí | Sí |

Con nivel `0.0` no se modificaron las 67 etiquetas positivas del lote. Aun sin perturbación artificial, Evidently detectó drift en una característica, insuficiente para activar la alerta del 30 %. En `0.3` y `0.6` se activaron las dos alertas y aumentó la proporción simulada de destino USA. Los valores p inferiores a `0,000001` no deben confundirse con un valor exactamente nulo cuando Prefect los presenta redondeados como `0.000000`.

El AUC, calculado con respecto a las **etiquetas originales**, disminuyó de **0,7464** a **0,6752** entre los escenarios extremos. Esta asociación corresponde a un experimento de perturbación controlada, no a desempeño en producción ni a una relación causal generalizable.

En `main` se verificaron **44 pruebas automatizadas**. La ampliación de cobertura en una rama de desarrollo alcanzó **83 pruebas satisfactorias**, distribuidas en las 44 iniciales, 15 del modelo, 9 de inferencia y 15 de FastAPI con `TestClient` y mocks. El total definitivo se consolidará tras integrar y comprobar la batería completa, incluidos los flujos de drift.

## 6. Problemas encontrados

Se corrigió un error de indentación en la inferencia y se comprobó la sintaxis de Python. Asimismo, se identificó que Evidently generaba inicialmente reportes con un umbral predeterminado del **50 %**, distinto del **30 %** definido en YAML. Se ajustó `DataDriftPreset(drift_share=umbral)`, se regeneraron los tres reportes y se verificó el criterio aplicado; el cambio se integró en el PR #18. Las nueve ejecuciones definitivas terminaron sin fallos.

## 7. Conclusiones

La separación de la lógica estadística y la orquestación permitió verificar la simulación y el monitoreo de manera reproducible. El nivel `0.0` mostró que la ausencia de perturbación no implica igualdad estadística entre las distribuciones, y que una característica con drift no necesariamente desencadena una alerta global. Los niveles `0.3` y `0.6` activaron ambas alertas y registraron una reducción progresiva del AUC en el experimento emulado.

La ampliación de pruebas fortalece la confiabilidad del sistema; antes de una aplicación operativa sería necesario contrastar su comportamiento en otros períodos, evaluar la sensibilidad de los umbrales y consolidar la ejecución automatizada de la batería de pruebas. La integración continua con GitHub Actions se considera una extensión del proceso de validación, no un resultado ya acreditado.

## 8. Uso de herramientas de IA

Se utilizó **ChatGPT** como apoyo en la elaboración y revisión del módulo de monitoreo y sus pruebas, los flujos de Prefect y Evidently, el diagnóstico de errores, los comandos de PowerShell, el versionamiento y la documentación. Las propuestas se sometieron a validación mediante ejecución de pruebas, comprobación de sintaxis y revisión de reportes; se corrigieron los resultados que lo requirieron, incluido el ajuste del umbral de Evidently.

**Liliana Carolina Herrera Prieto** desarrolló y probó, con asistencia de IA, `src/monitoreo.py`, `tests/test_monitoreo.py` y `tests/test_datos.py`, integrados mediante el PR #16; verificó las pruebas, gestionó su rama y elaboró la documentación académica. **Fredy Ernesto Sandoval Méndez** desarrolló y verificó, también con asistencia de IA, la inferencia y los flujos de drift; ejecutó los escenarios, revisó los resultados en Prefect, MLflow y Evidently, recopiló las evidencias, gestionó los PR #17 y #18 y amplió las pruebas de modelo, inferencia y API. La elección del caso, la aceptación de cambios, la interpretación de los resultados y la comprobación de las ejecuciones correspondieron a los integrantes, no a decisiones delegadas a la IA.

**Referencias:** [Guía oficial](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/actividad-3.md) · [Plantilla de memoria](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/plantilla-memoria.md) · [Repositorio del equipo](https://github.com/fesandovalm-sys/usta-mlops-exportaciones).

<!-- Referencias reservadas para incorporación tras validación y publicación de las capturas de MLflow:
[Parámetros y métricas de inferencia](evidencias/13_mlflow_inferencia_parametros_metricas.png)
[Artefactos del experimento](evidencias/14_mlflow_inferencia_artefactos.png)
-->
