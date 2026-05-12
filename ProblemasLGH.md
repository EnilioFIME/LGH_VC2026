Voy a revisar todos los archivos con cuidado antes de responderte.

Los scripts LGH ya los tengo en contexto. Déjame leer el PDF y los scripts de entrenamiento/inferencia con más detalle.El PDF ya está en contexto — tengo todo lo que necesito. Voy a hacer el análisis completo ahora.

Aquí va un análisis exhaustivo de todo lo que vi. Lo organizo de lo más general a lo más específico.

---

## Diferencias generales con el artículo

**El papel no es clasificación, es word spotting.** Esta es la diferencia conceptual más grande y afecta todo el proyecto. En el paper, el sistema no clasifica una imagen entre 10 clases — detecta si una imagen contiene una palabra específica, produciendo un *score* de similitud. La evaluación se hace con curvas DET (False Acceptance vs. False Rejection) y Average Precision por palabra. Tu implementación convierte esto en un clasificador multiclase (la imagen → clase ganadora), lo cual es diferente en propósito, evaluación y arquitectura. Para el proyecto de clase probablemente esté bien, pero hay que ser consciente de ello.

**El paper usa un paso de fast rejection previo a LGH.** La sección 2 del paper describe que antes de calcular LGH, un clasificador holístico descarta ~90% de las muestras (falseando solo el 5% de keywords). Sus resultados de mAP son sobre ese subconjunto filtrado, no sobre todos los recortes. Ustedes no tienen ese paso, así que sus números no son directamente comparables con los del paper aunque todo lo demás estuviera igual.

**El dataset del paper es distinto al de ustedes en escala de muestras.** El paper tiene entre 208 y 750 ejemplos positivos *por keyword*. El archivo `Auditoria.txt` muestra que algunas clases de ustedes ya tenían pocos ejemplos antes de limpiar — y después de eliminar decenas, el número disponible puede ser muy bajo para entrenar HMMs robustos.

---

## Errores de interpretación por archivo

### `hmm_train_test.py` — Múltiples problemas

**1. `MAX_STATES = 4` con lógica de reducción automática — contradice el paper completamente.**
El paper dice explícitamente: *"Word HMMs trained using 10 states per character"*. Con 4 estados máximos y la función `choose_n_states` que los reduce aún más según el mínimo de frames, muchos modelos van a entrenarse con 2 estados. Un HMM con 2 estados apenas puede capturar la secuencia temporal de una palabra escrita. Esto es probablemente la causa mayor de resultados pobres si los hay.

El problema real que llevó a reducir estados es que hay pocas muestras por clase, pero la solución correcta no es reducir estados sino regularizar más el modelo (más iteraciones, covarianza más robusta, o datos aumentados). Con pocas muestras y muchos estados sí hay riesgo de colapso numérico, pero `MAX_STATES = 4` es demasiado agresivo.

**2. `N_ITER = 30` es muy bajo.**
El paper no especifica el número de iteraciones de Baum-Welch, pero 30 es poco para convergencia. Con el volumen de datos que tienen, 100-200 es más razonable.

**3. `init_params="mc"` (solo means y covars) con `params="stmc"` (entrenar todo incluyendo startprob y transmat) es inconsistente.**
Si inicializas startprob y transmat manualmente (left-to-right) pero luego los marcas para entrenamiento con `params="stmc"`, el Baum-Welch los va a reestimar libremente y va a romper la topología left-to-right durante el entrenamiento. Para respetar la estructura left-to-right deberías usar `params="mc"` (solo entrenar means y covars) o bien usar `params="stmc"` pero sin fijar startprob/transmat manualmente. Lo más correcto para reproducir el paper es `params="stmc"` con inicialización aleatoria, o bien `params="mc"` con startprob/transmat fijos.

**4. La función `choose_n_states` usa `total_frames // max(len(seqs), 1)` como criterio.**
Esto hace que si hay 10 secuencias con 20 frames cada una (200 frames totales), el resultado sea `200 // 10 = 20`, lo que no limita nada. Pero si hay pocas secuencias largas, puede dar resultados raros. El criterio debería ser simplemente `min(target_states, min_frames_in_any_sequence)`.

**5. El 5-fold CV no separa por escritor como pide el paper.**
El paper dice: *"ensuring that the same writer is not mixed among them"*. Su `KFold` es aleatorio sobre secuencias, sin ningún control de escritor. Esto infla artificialmente los resultados porque el mismo escritor puede aparecer en train y test. El nombre del archivo contiene el ID del documento (`02121_L_...`) que en RIMES identifica la carta — y aunque no tienen el ID de escritor directamente, podrían agrupar por carta para evitar que imágenes de la misma carta aparezcan en train y test a la vez.

**6. `get_covars_diag` asume que `covars_` tiene ndim=3 para covarianza diagonal.**
`hmmlearn` con `covariance_type="diag"` devuelve `covars_` de forma `(n_components, n_features)` directamente, no `(n_components, n_features, n_features)`. La condición `if c.ndim == 3` nunca se cumple para covarianza diagonal, así que la función siempre devuelve `c` sin modificar — lo cual está bien accidentalmente, pero el código es confuso y puede fallar si alguien cambia el tipo de covarianza.

---

### `dtw_train_test.py` — Un problema crítico

**`dtw_ndim.distance` requiere arrays C-contiguous y del mismo número de dimensiones.**
El script pasa secuencias de forma `(n_frames, 128)` directamente a `dtw_ndim.distance`. Esto puede fallar si las secuencias no son C-contiguous (lo son después de `df.values` pero conviene asegurarlo con `.copy()`). Más importante: `dtw_ndim` espera que ambas secuencias tengan el mismo número de columnas, lo cual debería cumplirse, pero no hay validación explícita.

**El paper usa distancia negativa como score de similitud, no distancia directa.**
El paper dice: *"the negative distance to the closest query is taken as a similarity score"*. Tu implementación busca la distancia mínima (`best_dist`) para clasificar, lo cual es equivalente para clasificación, pero si quisieran calcular AP por clase como el paper, necesitarían usar el score negativo explícitamente.

**No hay evaluación de las 3 variantes de grilla.**
`FEATURES_DIR` está hardcodeado a `Fitted`. El paper compara Fitted, Unfitted e Irregular tanto con HMM como con DTW — necesitan correr `dtw_train_test.py` tres veces con distintos paths, igual que `hmm_train_test.py`.

---

### `630LGH_fitted.py`, `630LGH_unfitted.py`, `630LGH_irregular.py` — Problema común

**El gradiente se calcula sobre la imagen NORMALIZADA con fondo blanco, no sobre la imagen invertida.**
En `N630.py`, la imagen final se guarda con `cv2.bitwise_not(norm_img)` — texto negro sobre fondo blanco (0-255). Los scripts LGH leen esa imagen directamente y calculan gradientes. Esto está bien en teoría (los bordes del texto generan gradientes en ambas convenciones), pero la magnitud del gradiente va a ser dominada por las transiciones negro→blanco en lugar de blanco→negro. El `LGH_utils.py` sí invierte la imagen antes de calcular gradientes (`img_inv = cv2.bitwise_not(img)`), creando una **inconsistencia entre los scripts de extracción masiva y la función de inferencia**. Si los modelos se entrenaron con features de imagen no-invertida (scripts LGH) pero la inferencia usa features de imagen invertida (LGH_utils), los scores serán incorrectos.

**`win_img` se usa para detectar `y_indices` en Fitted, pero `win_img` viene de la imagen con fondo blanco.**
Esto significa que `np.any(win_img > 0, axis=1)` detecta casi todas las filas como activas (el fondo blanco = 255 > 0), no solo las del texto. La detección de `y_min` y `y_max` en la grilla Fitted va a ser casi siempre 0 y h-1, haciendo que la grilla fitted se comporte como una unfitted. Esto es un bug serio que elimina la ventaja principal del fitted grid. `LGH_utils.py` no tiene este problema porque trabaja sobre `img_inv`.

---

### `LGH_utils.py` vs scripts LGH — Inconsistencia de imagen

Como ya mencioné: `LGH_utils.py` invierte la imagen (`bitwise_not`) antes de calcular gradientes. Los scripts `630LGH_*.py` no lo hacen. Esto genera features distintas para el mismo recorte dependiendo de qué código se use. Los modelos HMM entrenados con features de los scripts LGH no serán compatibles con la inferencia que usa `LGH_utils.py`.

---

### `inferencia.py` — Problema menor

**`words[best_label]` puede fallar si `words` es una lista y `best_label` es un entero que no existe.**
Si el diccionario `final_models` tiene llaves que no son índices consecutivos (puede pasar si falla el entrenamiento de alguna clase), `best_label` podría ser un int que no corresponde a ningún índice válido de `words`. Es un edge case poco probable pero no imposible.

---

### `Auditoria.txt` — Observación

**"LEXPRESSION" y "DAGRÉER" son casi seguro artefactos de segmentación del XML, no palabras reales.**
"L'expression" en francés es una contracción con apóstrofo — el parser de `WC630.py` probablemente está concatenando la "L'" con "expression". Lo mismo con "D'agréer". Si estas dos clases son artefactos, están contaminando el vocabulario con clases que no existen como palabras independientes, lo que dificulta el entrenamiento.

---

## Resumen de prioridades

| Severidad | Problema |
|---|---|
| 🔴 Crítico | Inconsistencia de inversión de imagen entre scripts LGH y `LGH_utils.py` — hace que inferencia use features distintas a las de entrenamiento |
| 🔴 Crítico | Bug en Fitted: `y_indices` detectado sobre imagen no-invertida — la grilla fitted no funciona como tal |
| 🔴 Crítico | `MAX_STATES = 4` contradice el paper (10 estados) y limita drásticamente la capacidad del HMM |
| 🟠 Importante | `init_params="mc"` + `params="stmc"` rompe la topología left-to-right |
| 🟠 Importante | El 5-fold CV no separa por escritor como el paper |
| 🟠 Importante | No se calculan DET curves ni mAP — métrica diferente al paper |
| 🟡 Menor | `N_ITER = 30` es bajo para convergencia |
| 🟡 Menor | `dtw_train_test.py` solo evalúa Fitted, no las 3 variantes |
| 🟡 Menor | "LEXPRESSION" y "DAGRÉER" son probablemente artefactos de parseo |
| 🟡 Menor | El paper incluye fast rejection previo a LGH — sus mAP no son directamente comparables |