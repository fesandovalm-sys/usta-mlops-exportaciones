# Memoria 3 — Monitoreo, drift y CI

**MLOps · Maestría en Ciencia de Datos · Universidad Santo Tomás**  
**Integrantes:** Liliana Carolina Herrera Prieto y Fredy Ernesto Sandoval Méndez  
**Estado:** borrador. Los campos **[PENDIENTE]** requieren resultados y evidencias verificables.

## 1. Objetivo

Extender el clasificador de destino de exportaciones colombianas de las actividades previas con simulación reproducible de drift, monitoreo de variables y del objetivo, alertas y pruebas automatizadas. El caso de estudio conserva los microdatos EXPO del DANE/ANDA y la clasificación `destino_usa`: Estados Unidos (`1`) frente a otros destinos válidos (`0`).

## 2. Descripción del trabajo

Se desarrolló `src/monitoreo.py` con cuatro funciones independientes de Prefect y MLflow: `aplicar_drift`, `validar_lote`, `drift_de_target` y `fraccion_desde_evidently`. La simulación contempla niveles `0.0`, `0.3` y `0.6`; altera `pnk`, `fobdol` y la proporción simulada del objetivo, conservando las etiquetas originales en `destino_usa_original`.

Se añadieron `tests/test_monitoreo.py` y `tests/test_datos.py` para validar la lógica mediante datos sintéticos. Estos archivos se integraron mediante el [PR #16](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/16). El entrenamiento y la API provienen de la Actividad 2.

Como primera comprobación de integración, se ejecutó la inferencia con nivel de drift `0.0`, utilizando el modelo Random Forest `champion` v2 y un lote de 500 registros de septiembre de 2026. Prefect registró la ejecución como **Completed**, con cuatro tareas finalizadas. La simulación conservó las 67 etiquetas positivas originales, sin modificaciones.

Para el monitoreo de variables de este escenario se comparó el lote de 500 registros con una **muestra reproducible de 10.000 registros de entrenamiento**. Evidently identificó drift estadístico en el **20 % de las cinco variables predictoras** (una de cinco). Al no superar el umbral configurado de **30 %**, el sistema **no generó alerta**. Este resultado se refiere a *feature drift*, no a *target drift*.

Fredy implementó dos flujos adicionales de Prefect: `flows/drift_features_flow.py`, para el monitoreo de características con Evidently, y `flows/drift_evento_flow.py`, para comparar la variable objetivo mediante chi-cuadrado. En el escenario `0.0`, este último obtuvo tasas de destino USA de **12,90 %** en entrenamiento y **13,40 %** en el lote (p = **0,797220**, α = **0,05**), por lo que **no generó alerta de target drift**.

**[PENDIENTE]** Incorporar resultados de los niveles `0.3` y `0.6`, los artefactos de monitoreo y la validación documental de las ejecuciones completas de ambos flujos.

## 3. Decisiones técnicas

La lógica de monitoreo se separó de la orquestación para permitir pruebas unitarias sin servicios levantados. La emulación aplica a `pnk` y `fobdol` un desplazamiento de `2 × nivel × escala`, con límite inferior de cero, y aumenta artificialmente la proporción de `destino_usa = 1` hasta `0.30 × nivel` puntos de proporción.

Una semilla fija permite reproducibilidad y `destino_usa_original` distingue las etiquetas observadas de las simuladas. **Estos cambios no representan una variación real de las exportaciones.**

Para el objetivo binario se utiliza una prueba de chi-cuadrado con significancia `0.05`; la fracción de variables con drift se extrae del resultado de Evidently. La alerta de características se condiciona a que dicha fracción **supere** el umbral configurado de `0.30`. Se distinguen así el nivel de perturbación artificial, el resultado del contraste estadístico y la activación de una alerta.

**[PENDIENTE]** Justificar el umbral final y documentar las alternativas efectivamente consideradas o descartadas durante la integración.

## 4. Evidencias

- **Pruebas locales:** `poetry run python -m pytest -q` → **`44 passed in 2.67s`**, después de instalar las dependencias y sincronizar la rama.
- **Versionamiento:** PR #16 integrado, commit `9323146`, merge `53da2e2`.
- **Primera inferencia, nivel 0.0:** Prefect `Completed`, cuatro tareas finalizadas, modelo Random Forest `champion` v2, lote de 500 registros y ninguna etiqueta modificada.
- **Inferencia y feature drift, nivel 0.0 (resultados comunicados por Fredy):** referencia reproducible de 10.000 registros y lote de 500; Evidently identificó drift en 1 de 5 variables (20 %), sin alerta al no superar el umbral de 30 %.
- **Target drift, nivel 0.0 (resultado comunicado por Fredy):** referencia 12,90 %, lote 13,40 %, chi-cuadrado p = 0,797220; sin alerta con α = 0,05.
- **[EVIDENCIAS POR INCORPORAR]** Capturas guardadas por Fredy, aún no incorporadas ni revisadas en esta memoria:
  - `evidencias/01_prefect_inferencia_drift_0.png` — inferencia.
  - `evidencias/02_evidently_features_drift_0.png` — reporte de características.
  - `evidencias/03_prefect_target_drift_0.png` — flujo del objetivo.
- **[PENDIENTE]** Evidencias de los niveles `0.3` y `0.6`, sus reportes de drift, comparaciones y alertas.
- **[PENDIENTE]** Evidencias adicionales de pruebas de modelo, API e inferencia, y de CI únicamente si se implementó.

## 5. Resultados

Se comprobaron **44 pruebas locales satisfactorias**. Las pruebas del módulo de monitoreo emplearon datos sintéticos, sin depender de MLflow, Prefect ni microdatos originales.

En la primera ejecución de inferencia con nivel `0.0`, Fredy verificó que el lote conservó sus 67 etiquetas originales de destino USA, sin modificaciones simuladas. El AUC temporal registrado fue **0,7464**. Este resultado corresponde al desempeño del modelo en el lote evaluado y no constituye una medición de drift estadístico.

| Criterio | Resultado |
|---|---|
| Inferencia nivel `0.0` | Completada; 500 registros, 0 etiquetas modificadas |
| AUC temporal, nivel `0.0` | 0,7464 |
| Referencia de monitoreo, nivel `0.0` | Muestra reproducible de 10.000 registros de entrenamiento |
| *Feature drift* con Evidently, nivel `0.0` | 20 % de cinco variables predictoras (1 de 5) |
| Alerta de características, nivel `0.0` | No se generó; 20 % no supera el umbral de 30 % |
| Tasa `destino_usa` referencia / lote, nivel `0.0` | 12,90 % / 13,40 % |
| Chi-cuadrado de target, nivel `0.0` | p = 0,797220; α = 0,05; **sin alerta** |
| Inferencias niveles `0.3` y `0.6` | **[PENDIENTE]** |
| Reportes de *feature drift* de otros niveles | **[PENDIENTE]** |
| Comparación estadística del target en `0.3` y `0.6` | **[PENDIENTE]** |
| Alerta para el nivel `0.6` | **[PENDIENTE]** |
| Comparación de reportes entre `0.0` y `0.6` | **[PENDIENTE]** |

La ejecución con nivel `0.0` confirma que no se introdujeron modificaciones mediante la simulación. **Esto no equivale a ausencia de diferencias estadísticas:** Evidently detectó drift en una de las cinco variables frente al entrenamiento, aunque la fracción quedó por debajo del umbral de alerta. En contraste, para el objetivo la diferencia entre las tasas de 12,90 % y 13,40 % no fue estadísticamente significativa al nivel `0,05` (p = `0,797220`); no se generó alerta. Aún no se dispone de los resultados de los escenarios `0.3` y `0.6`.

## 6. Problemas encontrados

Unas modificaciones locales previas de la Memoria 2 impedían la actualización mediante `git pull`. Se preservaron con `git stash`, se sincronizó el repositorio y, tras comprobar que los archivos coincidían con las versiones integradas, se descartó el respaldo temporal.

El PR #16 se fusionó sin conflictos; las pruebas volvieron a pasar después del merge.

**[PENDIENTE — Fredy]** Documentar únicamente incidentes reales en la integración de inferencia, Prefect y Evidently, con sus soluciones.

## 7. Conclusiones

Se obtuvo una base de monitoreo comprobable y reproducible, separada de los servicios de orquestación. Preservar la etiqueta original evita confundir una alteración simulada con un hecho observado.

La primera inferencia con drift emulado `0.0` se completó sin alterar las etiquetas. Evidently identificó drift en una de cinco variables (20 %), insuficiente para activar la alerta del 30 %. El contraste de la proporción del objetivo tampoco generó alerta (p = 0,797220 frente a α = 0,05). Por tanto, **no introducir perturbación artificial no implica que las distribuciones de entrada sean idénticas a las de entrenamiento**; además, encontrar drift en una variable no obliga a activar una alerta global.

**[PENDIENTE]** Contrastar los niveles `0.3` y `0.6`, comprobar sus reportes y alertas, y ampliar las conclusiones según la evidencia obtenida.

## 8. Uso de herramientas de IA

Se utilizó asistencia de IA para elaborar el módulo de drift y sus pruebas, orientar la solución de incidencias de Git y preparar esta documentación.

Liliana ejecutó y verificó las pruebas, gestionó su rama y el PR, y comprobó los resultados locales. Fredy coordina la integración de inferencia y los flujos de detección con apoyo de su IA.

**[PENDIENTE — declaración individual]** Precisar qué propuestas de IA revisó, corrigió o descartó cada integrante y qué partes realizó sin IA. Validar esta atribución con Fredy antes de entregar.

**Fuentes:** [guía oficial](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/actividad-3.md) · [plantilla de memoria](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/plantilla-memoria.md) · [repositorio del equipo](https://github.com/fesandovalm-sys/usta-mlops-exportaciones).
