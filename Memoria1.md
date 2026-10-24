# Memoria de actividad 1

## 1. Objetivo

Familiarizarse con los fundamentos y las principales herramientas utilizadas en MLOps, mediante un primer acercamiento práctico al control de versiones con Git y GitHub, la gestión de entornos y dependencias con Poetry, el uso de contenedores con Docker, la orquestación de flujos de trabajo con Prefect y el seguimiento de experimentos con MLflow, con el fin de comprender su funcionamiento e importancia en la organización, reproducibilidad y trazabilidad de proyectos de aprendizaje automático.

## 2. Descripción del trabajo

La actividad permitió conocer cómo se relacionan el control de versiones, la gestión de dependencias, la orquestación de tareas y el seguimiento de experimentos, así como enfrentar algunas dificultades técnicas habituales durante la preparación de un entorno de desarrollo.

### 2.1. Estructura del repositorio y preparación del entorno

Inicialmente, consultamos y clonamos el repositorio del docente, con el propósito de disponer de los materiales de referencia y conocer la organización propuesta para las actividades del curso.

Posteriormente, creamos nuestro propio repositorio en GitHub, denominado **`usta-mlops-exportaciones`**, (https://github.com/fesandovalm-sys/usta-mlops-exportaciones) para desarrollar las actividades de manera colaborativa. Configuramos una rama principal (`main`) y ramas individuales (`lili/actividad-1` y `fredy/actividad-1`), lo que nos permitió comprender cómo trabajar sobre un mismo proyecto conservando el historial y los aportes de cada integrante. Escogimos el nombre de `exportaciones` pensando en realizar, en los casos de uso, una aproximación a nuestro trabajo de tesis sobre observabilidad en el procesamiento de los datos de la operación estadística de exportaciones del DANE.

En cuanto a la estructura del proyecto, trabajamos con los siguientes componentes:

- **`README.md`:** documentación general del repositorio.
- **`.gitignore`:** identificación de archivos y carpetas que no deben incluirse en el control de versiones.
- **`pyproject.toml`:** definición de las dependencias y configuración del proyecto.
- **`poetry.lock`:** registro de las versiones específicas de las dependencias.
- **`flows/`:** carpeta destinada a los flujos de trabajo implementados con Prefect.
- **`flows/prueba_flow.py`:** archivo de prueba suministrado como referencia por el docente para comprobar el funcionamiento de Prefect y MLflow.

Para preparar el entorno utilizamos Python 3.12, Visual Studio Code y Poetry, incorporando las librerías necesarias para trabajar con Prefect, MLflow y Scikit-learn.

**Dificultades y aprendizajes**

Uno de los primeros inconvenientes se presentó durante la instalación de las dependencias. La ubicación inicial del proyecto en uno de nuestros equipos de Windows tenía una ruta extensa y la instalación de Prefect generó errores relacionados con la creación de archivos.

En un primer momento trasladamos el repositorio a una ubicación más corta, lo que nos llevó a revisar si se habían perdido los avances realizados. Mediante los comandos de Git comprobamos que las ramas, los commits y la conexión con GitHub permanecían intactos. También identificamos commits locales pendientes de publicación y los sincronizamos con el repositorio remoto.

Finalmente, resolvimos el inconveniente configurando Poetry para almacenar el entorno virtual en una ruta independiente y más corta (`C:\venvs`), sin necesidad de volver a trasladar los archivos del proyecto.

Esta experiencia nos permitió comprender mejor la diferencia entre el código del proyecto, su historial de versiones y el entorno donde se ejecutan las dependencias. También evidenció la importancia de verificar el estado de Git antes de realizar cambios en la organización de los archivos.

### 2.2. Puesta en funcionamiento de Prefect y MLflow

Una vez preparado el entorno, iniciamos los servicios de Prefect y MLflow para conocer sus interfaces y comprobar su funcionamiento.

Aunque inicialmente exploramos Docker y las herramientas de administración de contenedores de Visual Studio Code, optamos por ejecutar los servicios directamente mediante Poetry, siguiendo la alternativa sin Docker contemplada en las instrucciones del docente.

Para ello, iniciamos los servidores locales y verificamos el acceso a sus interfaces:

- **Prefect:** `http://127.0.0.1:4200`
- **MLflow:** `http://127.0.0.1:5000`

Este acercamiento nos permitió reconocer las diferencias entre ambas herramientas: Prefect facilita la organización, ejecución y supervisión de flujos de trabajo, mientras que MLflow permite registrar y consultar la información asociada con los experimentos.

**Dificultades y aprendizajes**

Durante el inicio de Prefect encontramos un error relacionado con el bloqueo de la base de datos SQLite (`database is locked`). Inicialmente parecía que el servicio no había iniciado correctamente; sin embargo, al consultar su interfaz web comprobamos que el servidor estaba respondiendo.

Esta situación nos enseñó que los mensajes de error deben analizarse en su contexto y que no necesariamente implican una falla completa del servicio. También comprendimos la importancia de revisar los registros de ejecución y comprobar el funcionamiento de las herramientas antes de tomar decisiones como reinstalarlas o modificar su configuración.

### 2.3. Ejecución del flujo de prueba con Prefect y MLflow

Con los servicios en funcionamiento, realizamos las pruebas del archivo **`flows/prueba_flow.py`**, tomando como referencia el código proporcionado por el docente.

El ejercicio consiste en ejecutar un flujo denominado `prueba`, que utiliza una tarea para calcular la suma de los números del 1 al 5. El resultado esperado es 15.

Posteriormente, el mismo flujo registra información en MLflow mediante el experimento `prueba-validacion`, incluyendo el parámetro `n` y la métrica `suma`.

Durante la práctica pudimos observar el flujo en la interfaz de Prefect y consultar su registro en MLflow, comprobando cómo una ejecución puede quedar documentada en ambas herramientas. Así mismo verificamos que eran accesibles en ambos equipos de los integrantes.

**Dificultades y aprendizajes**

Uno de los aspectos que inicialmente generó dudas fue encontrar las interfaces de Prefect y MLflow sin registros de ejecución. Posteriormente comprendimos que iniciar los servidores no equivale a ejecutar un flujo y que los resultados aparecen una vez se realizan las pruebas correspondientes.

Asimismo, identificamos la importancia de establecer correctamente las variables `PREFECT_API_URL` y `MLFLOW_TRACKING_URI`, ya que permiten indicar a qué servicios deben conectarse las ejecuciones.

### 2.4. Control de versiones y reproducibilidad

Finalmente, revisamos el estado del repositorio local y su sincronización con GitHub, comprobando que los avances realizados estuvieran asociados con las ramas de trabajo correspondientes.

También identificamos la necesidad de excluir del control de versiones los archivos propios del entorno de ejecución, como las bases de datos locales de MLflow y las configuraciones específicas de cada computador.

A través de este proceso comprendimos la función de `pyproject.toml` y `poetry.lock` en la reproducción del entorno, así como la importancia de mantener una estructura organizada y una documentación que permita a otros integrantes ejecutar el proyecto sin tener que reconstruir manualmente su configuración.

En conjunto, esta primera actividad fue más allá de instalar y ejecutar herramientas: nos permitió conocer su relación dentro de un flujo de trabajo de MLOps y resolver situaciones técnicas que difícilmente habríamos identificado mediante una aproximación exclusivamente teórica.

## 3. Decisiones técnicas

Durante el desarrollo de la actividad tomamos diferentes decisiones técnicas orientadas a facilitar el aprendizaje, organizar el trabajo colaborativo y garantizar que el entorno fuera funcional y reproducible. Algunas respondieron a las alternativas propuestas por el docente, mientras que otras surgieron de las dificultades encontradas durante la práctica.

### 3.1. Creación de un repositorio independiente para el trabajo colaborativo

Decidimos crear un repositorio propio en la cuenta de GitHub de uno de los integrantes, en lugar de realizar un *fork* del repositorio del docente. Aunque esta última alternativa habría permitido conservar directamente su estructura y facilitar la incorporación de actualizaciones, consideramos más apropiado mantener separados los materiales de referencia y nuestros desarrollos.

Esta decisión nos permitió organizar el proyecto de acuerdo con nuestras necesidades y familiarizarnos con la creación y administración de un repositorio desde cero.

Adicionalmente, optamos por utilizar ramas individuales de trabajo en lugar de realizar todas las modificaciones directamente sobre `main`. Con ello buscamos evitar interferencias entre nuestros avances y comprender cómo se gestionan los cambios en un entorno colaborativo.

### 3.2. Uso de Poetry para la gestión de dependencias

Siguiendo las alternativas presentadas en la guía, utilizamos Poetry para administrar el entorno y las dependencias del proyecto, en lugar de realizar instalaciones individuales mediante `pip` o utilizar gestores como Conda.

La principal razón fue conocer una herramienta que permite centralizar la definición de las dependencias en `pyproject.toml` y registrar sus versiones específicas en `poetry.lock`, facilitando que los integrantes trabajemos con configuraciones equivalentes.

Durante la práctica comprendimos que no basta con instalar las librerías: también es necesario asegurarse de que el código se ejecute desde el entorno correcto. Por esta razón, seleccionamos Python 3.12 y configuramos explícitamente su intérprete en Visual Studio Code.

### 3.3. Ubicación independiente del entorno virtual

Inicialmente trabajamos con un entorno virtual asociado a la ubicación del proyecto. Sin embargo, durante la instalación encontramos inconvenientes relacionados con la longitud de las rutas de Windows en uno de los equipos de los integrantes.

Consideramos trasladar nuevamente todo el repositorio a una ubicación más corta, pero finalmente preferimos conservar nuestra organización de carpetas y configurar Poetry para almacenar el entorno virtual en `C:\venvs` para el equipo de este integrante. Para el otro integrante sí se realizó de acuerdo con la guía, dentro del proyecto.

Esta alternativa permitió resolver el problema sin modificar nuevamente la ubicación del código ni afectar su relación con GitHub.

La experiencia nos ayudó a comprender que el entorno virtual no necesita estar dentro de la carpeta del proyecto y que es posible administrar sus dependencias de manera independiente.

### 3.4. Ejecución de Prefect y MLflow sin Docker

La guía del docente presenta dos alternativas para iniciar los servicios: utilizar contenedores mediante Docker Compose o ejecutarlos directamente desde un entorno Python.

Aunque exploramos las herramientas de contenedores disponibles en Visual Studio Code, decidimos utilizar inicialmente la segunda alternativa, dado que no contábamos con Docker Desktop instalado y nuestro propósito inmediato era comprender el funcionamiento de Prefect y MLflow.

Esta decisión nos permitió avanzar con las actividades prácticas sin incorporar una configuración adicional que todavía no conocíamos.

Posteriormente, repetimos el ejercicio utilizando Docker Desktop y Docker Compose, con el propósito de familiarizarnos con el uso de contenedores y facilitar la puesta en marcha conjunta de Prefect y MLflow, sin necesidad de iniciar cada servicio de manera individual. Esta experiencia también nos permitió explorar una alternativa que favorece la reproducibilidad y simplifica la administración del entorno para las siguientes actividades del curso.

### 3.5. Exclusión de archivos locales del control de versiones

Durante las pruebas identificamos que algunas herramientas generan archivos asociados con su funcionamiento, como las bases de datos locales de MLflow y los archivos de configuración específicos del entorno.

Frente a la posibilidad de incluir todos estos elementos en GitHub, optamos por excluir inicialmente los archivos locales que no forman parte del código fuente ni de las dependencias necesarias para reproducir el proyecto.

En particular, configuramos exclusiones locales para `mlflow.db` y `poetry.toml`, manteniendo los archivos `pyproject.toml` y `poetry.lock` como parte de la configuración reproducible del proyecto.

Esta decisión nos permitió reconocer la diferencia entre los archivos que deben compartirse para ejecutar un proyecto y aquellos que corresponden exclusivamente al funcionamiento de una instalación local. 

### 3.6. Priorización del flujo de prueba sobre el entrenamiento de un modelo

Para esta primera actividad decidimos concentrarnos en comprender la integración de Prefect y MLflow mediante el flujo de prueba proporcionado por el docente, en lugar de desarrollar inmediatamente una pipeline propia de entrenamiento.

Esta decisión responde tanto al carácter introductorio del módulo como a las instrucciones de la actividad, que establecen el entrenamiento del modelo como un ejercicio opcional.

Consideramos más conveniente comprender primero cómo se ejecuta y supervisa un flujo, cómo se registran sus parámetros y métricas, y cómo se consultan los resultados desde ambas herramientas.

De esta manera, priorizamos la comprensión del funcionamiento del entorno antes de incorporar la complejidad adicional del procesamiento de datos y el entrenamiento de modelos de aprendizaje automático.

## 4. Evidencias

- Enlaces o capturas de los runs de MLflow (parámetros, métricas, artefactos).
- Capturas de las UIs de Prefect y MLflow o salidas de terminal.
- Enlaces a commits o ejecuciones de CI cuando aplique.

## 5. Resultados

Métricas, tablas y gráficos. Presentar los resultados aunque no sean los esperados.

## 6. Problemas encontrados

Errores, bloqueos o resultados inesperados, y cómo se resolvieron.

## 7. Conclusiones

Qué se aprendió y qué mejoraría el sistema con más tiempo o recursos.

## 8. Uso de herramientas de IA

Requisito del curso: explicar cómo se usó la IA y detallar la contribución individual frente a la asistencia de IA.

- Herramientas utilizadas y para qué tareas.
- Qué partes del trabajo se hicieron sin IA.
- Qué salidas de la IA se revisaron, corrigieron o descartaron, y por qué.
