Es un excelente ejercicio de ingeniería inversa. El análisis del archivo Markdown es sumamente preciso y detecta problemas arquitectónicos que explican cualquier bajo rendimiento que estén teniendo. Seguramente tú y tus compañeros de equipo, como Luis Enrique, han notado que en proyectos de visión computacional, un error en el preprocesamiento o en la extracción de características define el "techo" de rendimiento de todo el modelo, sin importar qué tanto optimicen después.

Aquí tienes mis propias observaciones complementarias sobre su desarrollo en contraste con el artículo, seguidas del roadmap de corrección.

### Mis Observaciones sobre el Desarrollo y el Análisis

**1. El "Bug" de la Inversión y el Fitted Grid (El más destructivo)**
Concuerdo totalmente con el análisis: este es el problema más grave a nivel de procesamiento de imágenes. El artículo menciona que las características se calculan basándose en los píxeles de primer plano (foreground) para ajustar la ventana. En sus scripts de entrenamiento (`630LGH_*.py`), están operando con texto negro sobre fondo blanco (255).

* **La consecuencia:** La línea `np.any(win_img > 0, axis=1)` siempre va a detectar toda la altura de la ventana porque el fondo blanco es mayor a cero. Su "Fitted Grid", que según el artículo debería ser significativamente superior , se está comportando exactamente igual que un "Unfitted Grid".


* **La incompatibilidad:** Entrenar con gradientes dominados por transiciones negro→blanco y luego inferir en `LGH_utils.py` con transiciones blanco→negro (por la inversión `cv2.bitwise_not`) garantiza que el HMM evalúe datos que estadísticamente no se parecen a su entrenamiento.

**2. El Colapso Temporal del HMM**
El artículo es categórico: los HMM para palabras se entrenan usando 10 estados *por carácter*. Ustedes establecieron un límite duro de `MAX_STATES = 4` para *toda* la palabra. Entiendo perfectamente por qué lo hicieron: `Auditoria.txt` muestra que tienen muy pocas muestras limpias por clase, y un HMM con muchos estados y pocos datos lanza errores de singularidad o matrices mal condicionadas. Sin embargo, 4 estados no son suficientes para modelar la secuencia temporal de las transiciones de una palabra completa. Es mejor tener menos clases con suficientes datos, que muchas clases forzando al modelo a perder su capacidad secuencial.

**3. Clasificación Multiclase vs. Word Spotting**
Como señala el análisis, están resolviendo un problema distinto. El paper aborda el *word spotting*: evaluar si una imagen específica contiene la palabra de consulta, asignando un score. Ustedes armaron un clasificador: dada una imagen, encontrar cuál de las *N* clases es la ganadora (`argmax`). Esto explica por qué no están usando la distancia negativa en DTW ni las métricas correctas (curvas DET y Average Precision) . Si su rúbrica escolar les exige un clasificador multiclase, pueden mantenerlo, pero deben ser conscientes de que están desviándose del objetivo central del artículo.

**4. Validación Cruzada Contaminada**
El paper especifica que los 5 pliegues (folds) deben asegurar que un mismo escritor no se mezcle entre entrenamiento y prueba. Hacer un `KFold` aleatorio sobre todas las secuencias infla artificialmente sus resultados, ya que el modelo "memoriza" el estilo de un escritor en entrenamiento y luego lo reconoce fácilmente en la prueba.

---

### Roadmap de Corrección

Aquí tienen el plan de acción ordenado por prioridad. Lo crítico debe cambiarse antes de volver a correr cualquier entrenamiento.

#### Fase 1: Crítica (Hacerlo de inmediato)

*Estos errores invalidan los datos actuales. Sin esto, los modelos aprenden basura.*

1. **Estandarizar la Inversión de Imagen:** Modifiquen `630LGH_fitted.py`, `630LGH_unfitted.py` y `630LGH_irregular.py` para incluir `img = cv2.bitwise_not(img)` justo después de leer la imagen. Todo el pipeline debe trabajar con texto blanco (255) sobre fondo negro (0).
2. **Reparar el "Fitted Grid":** Al aplicar el paso 1, la lógica actual de `np.any(win_img > 0, axis=1)` ya funcionará correctamente, calculando el recorte basándose solo en los píxeles del texto.
3. **Regenerar las Características (Features):** Vuelvan a correr los tres scripts de LGH para reescribir todos los archivos CSV con los gradientes correctos.

#### Fase 2: Alta (Alineación de Modelado)

*Corregir la arquitectura matemática para que los HMM y DTW funcionen como en el artículo.*

1. **Dinámica de Estados del HMM:** Eliminen `MAX_STATES = 4`. Implementen una función que asigne estados proporcionalmente a la longitud promedio de la palabra en píxeles o frames (acercándose a la idea de 10 estados por carácter ).


2. **Proteger la Topología Left-to-Right:** En `hmm_train_test.py`, si inicializan manualmente `startprob_` y `transmat_`, cambien a `params="mc"` en el constructor del `GaussianHMM`. Si usan `params="stmc"`, el Baum-Welch reescribirá sus matrices y destruirá la direccionalidad de la escritura.
3. **Aumentar Iteraciones:** Suban `N_ITER` a un mínimo de 100 para asegurar la convergencia de las medias y covarianzas.

#### Fase 3: Media (Evaluación y Métricas)

*Ajustes para que la forma en que miden el éxito sea realista y comparable.*

1. **Separación por Escritor/Documento:** Modifiquen la lógica del K-Fold en `hmm_train_test.py` usando `GroupKFold` de scikit-learn. Usen el ID del documento (los primeros dígitos del nombre del archivo) como el grupo para garantizar que el modelo generalice ante nueva caligrafía.


2. 
**Ajuste del Score DTW:** Si deciden seguir el camino estricto del *spotting*, asegúrense de usar la distancia negativa de la consulta más cercana como su score de similitud final.


3. 
*(Opcional)* **Implementar mAP y Curvas DET:** En lugar de solo contar "Accuracy", almacenen los scores y usen las herramientas de `sklearn.metrics` para trazar la compensación entre Falsos Positivos y Falsos Negativos.



#### Fase 4: Baja (Limpieza y Tareas Secundarias)

*Mejoras incrementales una vez que el núcleo esté sano.*

1. 
**Limpieza del Diccionario:** Eliminen clases que son artefactos de segmentación como "LEXPRESSION" y "DAGRÉER" señaladas en `Auditoria.txt`. Solo meten ruido al entrenamiento.


2. 
**Testear las 3 Grillas:** Una vez que todo funcione, corran la evaluación para las características Unfitted e Irregular para comprobar empíricamente, como dice el paper, que el Fitted Grid da los mejores resultados para su caso.