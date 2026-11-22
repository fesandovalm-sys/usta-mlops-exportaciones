# Memoria 3 — Monitoreo, drift y CI

**Asignatura:** MLOps · Maestría en Ciencia de Datos, Universidad Santo Tomás

**Docente:** Henry Cala

**Integrantes:** Liliana Carolina Herrera Prieto y Fredy Ernesto Sandoval Méndez

**Repositorio:** https://github.com/fesandovalm-sys/usta-mlops-exportaciones

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

La implementación de monitoreo y sus pruebas iniciales se integraron mediante el [PR #16](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/16); los flujos y las nueve primeras capturas, mediante el [PR #17](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/17); y la parametrización del umbral de Evidently con tres reportes HTML, mediante el [PR #18](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/18). El [PR #19](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/19) incorporó las pruebas de modelo, inferencia, API y flujos de drift, eliminó una ejecución redundante de Evidently y añadió dos capturas de MLflow. **Las 14 evidencias están publicadas en el repositorio.**

| Nivel | Inferencia (Prefect) | Características (Evidently) | Objetivo (Prefect) | Reporte HTML |
|---|---|---|---|---|
| `0.0` | [01](evidencias/01_prefect_inferencia_drift_0.png) | [02](evidencias/02_evidently_features_drift_0.png) | [03](evidencias/03_prefect_target_drift_0.png) | [10](evidencias/10_evidently_reporte_html_drift_0.png) |
| `0.3` | [04](evidencias/04_prefect_inferencia_drift_03.png) | [05](evidencias/05_evidently_features_drift_03.png) | [06](evidencias/06_prefect_target_drift_03.png) | [11](evidencias/11_evidently_reporte_html_drift_03.png) |
| `0.6` | [07](evidencias/07_prefect_inferencia_drift_06.png) | [08](evidencias/08_evidently_features_drift_06.png) | [09](evidencias/09_prefect_target_drift_06.png) | [12](evidencias/12_evidently_reporte_html_drift_06.png) |

Las evidencias de MLflow complementan la trazabilidad de las ejecuciones de inferencia: [13 — Parámetros y métricas](evidencias/13_mlflow_inferencia_parametros_metricas.png) y [14 — Artefactos registrados](evidencias/14_mlflow_inferencia_artefactos.png). El registro de inferencia almacena el nivel de drift, la versión del modelo, el tamaño del lote, las métricas de predicción y los archivos del lote y de las predicciones.

## 5. Resultados

Se evaluaron lotes de **500 registros** contra **10.000 registros de referencia**. La tasa de `destino_usa = 1` en la referencia fue **12,90 %**.

| Nivel | AUC temporal | Características con drift | Tasa USA del lote | Valor p del objetivo | Alerta características | Alerta objetivo |
|---|---:|---:|---:|---:|---|---|
| `0.0` | 0,7464 | 20 % (1/5) | 13,40 % | 0,797220 | No | No |
| `0.3` | 0,7029 | 40 % (2/5) | 22,40 % | < 0,000001 | Sí | Sí |
| `0.6` | 0,6752 | 60 % (3/5) | 31,40 % | < 0,000001 | Sí | Sí |

Con nivel `0.0` no se modificaron las 67 etiquetas positivas del lote. Aun sin perturbación artificial, Evidently detectó drift en una característica, insuficiente para activar la alerta del 30 %. En `0.3` y `0.6` se activaron las dos alertas y aumentó la proporción simulada de destino USA. Los valores p inferiores a `0,000001` no deben confundirse con un valor exactamente nulo cuando Prefect los presenta redondeados como `0.000000`.

El AUC, calculado con respecto a las **etiquetas originales**, disminuyó de **0,7464** a **0,6752** entre los escenarios extremos. Esta asociación corresponde a un experimento de perturbación controlada, no a desempeño en producción ni a una relación causal generalizable.

El [PR #19](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/19), fusionado en `main`, documentó **95 pruebas automatizadas aprobadas**: 44 iniciales de ingesta, datos y monitoreo; 15 del modelo; 9 de inferencia; 15 de FastAPI con `TestClient` y *mocks*; y 12 de los flujos de drift. Se verifican contratos de entrada, comportamiento del modelo y predicción, alertas, generación de reportes y manejo de errores, mediante funciones aisladas, datos sintéticos y sustitutos de servicios externos. El resultado reportado corresponde a esa batería automatizada y no constituye una medición de desempeño operativo en producción.

## 6. Problemas encontrados

Se corrigió un error de indentación en la inferencia y se comprobó la sintaxis de Python. Asimismo, se identificó que Evidently generaba inicialmente reportes con un umbral predeterminado del **50 %**, distinto del **30 %** definido en YAML. Se ajustó `DataDriftPreset(drift_share=umbral)`, se regeneraron los tres reportes y se verificó el criterio aplicado; el cambio se integró en el PR #18. En la ampliación de pruebas se detectó y corrigió una ejecución redundante del reporte Evidently: ahora el flujo genera un único reporte por lote con el umbral configurado, comportamiento verificado mediante las pruebas del PR #19. Las nueve ejecuciones experimentales definitivas terminaron sin fallos.

## 7. Conclusiones

La separación de la lógica estadística y la orquestación permitió verificar la simulación y el monitoreo de manera reproducible. El nivel `0.0` mostró que la ausencia de perturbación no implica igualdad estadística entre las distribuciones, y que una característica con drift no necesariamente desencadena una alerta global. Los niveles `0.3` y `0.6` activaron ambas alertas y registraron una reducción progresiva del AUC en el experimento emulado.

Las **95 pruebas aprobadas**, que incluyen datos, modelo, inferencia, API y flujos de monitoreo, fortalecen la verificación funcional de la solución; no sustituyen la validación con datos y condiciones reales de operación. Antes de su aplicación operativa convendría evaluar otros períodos, la estabilidad de los umbrales y la disponibilidad de etiquetas reales para el monitoreo del objetivo. No se configuró GitHub Actions, dado que la guía establece la ejecución de pruebas en CI como **opcional**; su implementación queda como posible mejora, sin presentarse como resultado obtenido.

## 8. Uso de herramientas de IA

Se utilizó **ChatGPT** como apoyo en la elaboración y revisión del módulo de monitoreo y sus pruebas, los flujos de Prefect y Evidently, el diagnóstico de errores, los comandos de PowerShell, el versionamiento y la documentación. Las propuestas se sometieron a validación mediante ejecución de pruebas, comprobación de sintaxis y revisión de reportes; se corrigieron los resultados que lo requirieron, incluido el ajuste del umbral de Evidently.

**Liliana Carolina Herrera Prieto** desarrolló y probó, con asistencia de IA, `src/monitoreo.py`, `tests/test_monitoreo.py` y `tests/test_datos.py`, integrados mediante el PR #16; verificó resultados, gestionó su rama y elaboró y revisó la documentación académica. **Fredy Ernesto Sandoval Méndez** desarrolló y verificó, también con asistencia de IA, la inferencia y ambos flujos de drift; ejecutó los tres escenarios, examinó los resultados en Prefect, MLflow y Evidently, recopiló las 14 evidencias y gestionó los PR #17, #18 y #19. Asimismo, implementó y verificó la ampliación de pruebas del modelo, la inferencia, FastAPI y los flujos de drift. La elección del caso, la aceptación de cambios, la interpretación de resultados y la comprobación de ejecuciones correspondieron a los integrantes; las propuestas de IA se revisaron y ajustaron frente al código y los resultados observados.

**Referencias:** [Guía oficial](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/actividad-3.md) · [Plantilla de memoria](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/plantilla-memoria.md) · [Repositorio del equipo](https://github.com/fesandovalm-sys/usta-mlops-exportaciones).

