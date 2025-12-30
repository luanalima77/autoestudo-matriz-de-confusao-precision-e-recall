# 📊 Métricas de avaliação em modelos de classificação

Em problemas de classificação, as métricas são utilizadas para avaliar o desempenho de um modelo, comparando os valores reais com os valores preditos.

Essas métricas são geralmente calculadas a partir da matriz de confusão.


## 🧩 Matriz de confusão

A matriz de confusão é uma tabela que resume os acertos e erros de um modelo de classificação.

Para um problema binário, ela é formada por:
- TP (True Positive): previu positivo e era positivo;  
- FP (False Positive): previu positivo, mas era negativo;  
- FN (False Negative): previu negativo, mas era positivo;  
- TN (True Negative): previu negativo e era negativo.  

Exemplo de estrutura de matriz de confusão:
```
          Previsto
          1     0
Real  1   TP    FN
      0   FP    TN
```

## 🎯Métrica 1: acurácia (accuracy)
A acurácia indica a proporção de previsões corretas em relação ao total. Sua fórmula é dada por:

$$
Acurácia = \frac{TP + TN}{TP + TN + FP + FN}
$$
<br>

## 🔎 Métrica 2: precisão (precision)
A precisão mede quantos dos valores preditos como positivos realmente são positivos. Sua fórmula é dada por:

$$
Precisão = \frac{TP}{TP + FP}
$$

## 🔁 Métrica 3: recall (sensibilidade)
O recall indica quantos dos valores positivos reais foram corretamente identificados pelo modelo. Sua fórmula é dada por:

$$
Recall = \frac{TP}{TP + FN}
$$

### Relação entre precisão e recall
- Alta precisão = poucos falsos positivos
- Alto recall = poucos falsos negativos

Em muitos problemas, é necessário encontrar um equilíbrio entre essas duas métricas: o chamado f1-score.

## ⚖️ Métrica 4: F1-score
O F1-score é a média harmônica entre precisão e recall, sendo útil quando é necessário equilibrar falsos positivos e falsos negativos. Sua fórmula é dada por:

$$
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$


## 🧠 Qual métrica escolher?
Cada métrica oferece uma visão diferente do desempenho do modelo.
A escolha da métrica ideal depende do contexto do problema e do tipo de erro que se deseja minimizar.

