# Memoria de actividad 2
## Pipeline y empaquetado

*Universidad:* Universidad Santo Tomás  
*Programa:* Maestría en Ciencia de Datos  
*Asignatura:* MLOps  
*Docente:* Henry Cala  
*Integrantes:* Fredy Ernesto Sandoval Méndez y Liliana Carolina Herrera Prieto  
*Repositorio:* https://github.com/fesandovalm-sys/usta-mlops-exportaciones

## 1. Objetivo

Construir un sistema MLOps para estimar la probabilidad de que un registro de exportación colombiana tenga como país de destino Estados Unidos, integrando preparación de datos, entrenamiento, evaluación, registro de modelos e inferencia. La actividad amplía el entorno configurado en la Memoria 1 mediante pipelines de Prefect, seguimiento y gestión de versiones en MLflow, configuración YAML y una API desarrollada con FastAPI.

## 2. Descripción del trabajo

### 2.1. Preparación de datos

Utilizamos microdatos públicos de exportaciones del DANE, disponibles en el Archivo Nacional de Datos (ANDA), para el período agosto de 2025–septiembre de 2026. Implementamos `src/ingesta_anda.py` para procesar los archivos CSV incluidos en ZIP mensuales, normalizar valores, validar registros y generar conjuntos comprimidos para entrenamiento, validación y aplicación temporal.

Definimos la etiqueta `destino_usa` (1: Estados Unidos; 0: otros países válidos) contrastando los campos `PAIS` y `COD_PAI4`. El conjunto procesado conserva cinco variables predictoras: `capitulo`, `modad`, `dpto_origen`, `pnk` y `fobdol`; además de `periodo` y `destino_usa`, que no se utilizan como predictores. Implementamos 21 pruebas automatizadas para verificar las transformaciones y reglas de calidad.

### 2.2. Modelado, orquestación e inferencia

El desarrollo se distribuyó en componentes independientes: preparación y pruebas de datos; construcción y evaluación de un clasificador con Scikit-learn; pipeline de entrenamiento en Prefect y registro en MLflow; pipeline de inferencia que consulta el alias `champion`; y API con FastAPI. Los parámetros del modelo, umbrales y selección de variables deben administrarse mediante archivos YAML.

**[PENDIENTE — Fredy]** Completar con los archivos y flujos efectivamente implementados, el funcionamiento de los dos despliegues de Prefect, la programación diaria de inferencia y la prueba de la API. No presentar estos componentes como ejecutados hasta disponer de sus evidencias.

## 3. Decisiones técnicas

### 3.1. Caso de uso y selección de variables

Inicialmente contemplamos detectar valores FOB potencialmente atípicos, en relación con nuestros intereses de investigación. Para esta actividad preferimos una clasificación binaria con etiqueta observada, que permitiera evaluar la discriminación del modelo mediante AUC-ROC sin definir previamente una metodología de etiquetado de anomalías.

Excluimos `PAIS` y `COD_PAI4` de las entradas predictoras para evitar una fuga directa de la respuesta. Conservamos los códigos categóricos como texto para no perder ceros iniciales.

### 3.2. Separación temporal y reproducibilidad

Elegimos una división cronológica, en lugar de una partición aleatoria: entrenamiento de agosto de 2025 a mayo de 2026; validación de junio a agosto de 2026; y aplicación temporal en septiembre de 2026. Esta decisión permite examinar el comportamiento sobre registros posteriores. La lectura por bloques evita extraer manualmente todos los archivos y facilita repetir el procesamiento. Los microdatos permanecen fuera de GitHub; se versionan el código, las pruebas y la documentación.

### 3.3. Modelo y gestión de versiones

Se planteó una regresión logística como punto de partida por su adecuación a la clasificación binaria y su interpretabilidad. La configuración externa mediante YAML permite comparar selecciones de variables (`ALL`, `NUMERIC` y `CUSTOM`) sin modificar la lógica principal. Según la guía, el alias `champion` solo debe cambiar si el candidato supera el AUC mínimo y mejora el resultado del modelo vigente.

**[PENDIENTE — Fredy]** Verificar el modelo definitivo, los parámetros, el umbral aplicado, el registro de `features` y `n_features` en MLflow y la comprobación efectiva de la regla de promoción.

## 4. Evidencias

| Evidencia | Comprobación |
|---|---|
| [Preparación y validaciones](fuentes_y_datos.md) | Fuente, transformaciones, particiones y controles de calidad documentados. |
| [Pull Request #8](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/8) | Integración a `main` de la ingesta, 21 pruebas y documentación. |
| **[PENDIENTE]** Captura de Pytest | Resultado `21 passed` en el entorno local. |
| **[PENDIENTE — Fredy]** Prefect y MLflow | Ejecuciones de entrenamiento e inferencia, AUC, tag `fase = entrenamiento`, selección de variables y versión `champion`. |
| **[PENDIENTE — Fredy]** Despliegues y FastAPI | Dos despliegues, horario diario de inferencia, archivo de predicciones fechado y respuesta de ejemplo de la API. |

Las capturas definitivas se incorporarán únicamente cuando estén disponibles y se haya comprobado su correspondencia con las ejecuciones descritas.

## 5. Resultados

La ingesta procesó 559.450 registros originales; 559.397 cumplieron las reglas de calidad y 53 fueron excluidos.

| Conjunto | Registros válidos | Destino USA | Porcentaje USA |
|---|---:|---:|---:|
| Entrenamiento | 404.214 | 51.751 | 12,80 % |
| Validación | 114.830 | 14.333 | 12,48 % |
| Septiembre | 40.353 | 4.558 | 11,30 % |
| **Total** | **559.397** | **70.642** | **12,63 %** |

Se verificaron siete columnas homogéneas, ausencia de valores nulos y separación correcta de los períodos. Las 21 pruebas automatizadas finalizaron satisfactoriamente.

**[PENDIENTE — Fredy]** Incorporar el AUC de validación, el resultado de la selección del `champion`, el estado de las ejecuciones en Prefect, el archivo de predicciones y un ejemplo real de respuesta de FastAPI. Presentar también los resultados que no hayan cumplido las expectativas.

## 6. Problemas encontrados

La ingesta requirió procesar ZIP anidados, interpretar números con formato decimal colombiano y preservar ceros iniciales de variables categóricas. Se resolvió mediante lectura por bloques, transformaciones explícitas y pruebas con datos sintéticos. También se excluyeron registros inconsistentes conforme a reglas documentadas. Para ejecutar las pruebas fue necesario instalar Pytest en el entorno virtual local.

**[PENDIENTE — Fredy]** Añadir únicamente las dificultades reales del entrenamiento, MLflow, Prefect o FastAPI y cómo fueron resueltas.

## 7. Conclusiones

La actividad permitió avanzar desde la configuración de herramientas hacia un caso reproducible con datos reales. La ingesta automatizada y las pruebas proporcionaron entradas consistentes para el modelado. La proporción de registros con destino a Estados Unidos (12,63 %) evidencia un desbalance de clases que debe considerarse al interpretar las métricas. El prototipo tiene fines académicos y no constituye un sistema oficial de predicción del DANE.

**[PENDIENTE — ambos integrantes]** Concluir, con base en las pruebas finales, si se logró la integración entre Prefect, MLflow y FastAPI, y señalar una mejora prioritaria.

## 8. Uso de herramientas de IA

Utilizamos ChatGPT como apoyo para estructurar la ingesta, formular pruebas automatizadas, interpretar errores y organizar la documentación. Liliana ejecutó y comprobó localmente el procesamiento, las validaciones y las pruebas; asimismo, revisó los textos y los ajustó a los resultados observados. Fredy asumió el desarrollo de los componentes de modelado, configuración, orquestación e inferencia.

No aceptamos automáticamente las propuestas de IA: revisamos las reglas de clasificación y exclusión, contrastamos las cifras con las salidas del programa y ajustamos el alcance del ejercicio para cumplir los requisitos de la actividad. Las métricas y capturas se incorporarán únicamente a partir de ejecuciones reales.

**[PENDIENTE — Fredy]** Precisar su uso de herramientas de IA, las tareas realizadas directamente y las recomendaciones aceptadas, modificadas o descartadas.
