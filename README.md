# Departamento_de_Relaciones_P
Este código de Python utiliza el procesamiento de lenguajes naturales y Machine Learning para analizar el sentimiento de los clientes basándose en publicaciones y valoraciones en redes sociales. 😊

# Procesamiento del Lenguaje Natural

Los procesadores del lenguaje natural convierten las palabras en números y entrenan modelos de Machine Learning para hacer predicciones. De este modo, podemos saber si nuestros clientes están contentos o no sin tener que revisar manualmente un gran número de tweets o valoraciones.

El departamento de relaciones públicas ha recolectado una gran cantidad de datos sobre sus clientes, como valoraciones de sus productos en Internet. Basándose en esas valoraciones (en formato de texto), el equipo busca predecir si sus clientes están satisfechos o no con su producto.

## Dependencias

Para ejecutar este código, necesitará las siguientes librerías:

```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
```

