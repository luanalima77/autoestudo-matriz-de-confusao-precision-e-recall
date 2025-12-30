# 📊 Entendendo matriz de confusão e gerando uma no terminal

Este repositório apresenta um autoestudo prático sobre matriz de confusão, gerando uma no terminal
O estudo foi realizado com base neste artigo do Medium: <a href = "https://medium.com/data-hackers/entendendo-o-que-%C3%A9-matriz-de-confus%C3%A3o-com-python-114e683ec509">clique aqui para ver o artigo</a>

## 🎯 Objetivo do autoestudo

O objetivo deste projeto é:
- Compreender o conceito de matriz de confusão;
- Identificar TP, TN, FP e FN;
- Implementar manualmente uma matriz de confusão em Python;
- Praticar lógica de programação aplicada à avaliação de modelos preditivos.

## 🧠 Conceitos abordados

Neste autoestudo, são trabalhados os seguintes conceitos:
- Classificação binária;
- Valores reais vs valores preditos;
- True Positive (TP): valores da positiva previstos corretamente;
- True Negative (TN): valores da classe que não é a positiva previstos corretamente;
- False Positive (FP): valores da positiva previstos incorretamente;
- False Negative (FN): valores da classe que não é a positiva previstos incorretamente;
- Construção manual de uma matriz de confusão usando NumPy.

## 📦 Biblioteca utilizada

```python
import numpy as np
```

## 🔢 Dados utilizados

Foram utilizados dois arrays simples para simular um problema de classificação binária:

```python
# Valores reais
valores_reais = [1, 0, 1, 0, 0, 0, 1, 0, 1, 0]

# Valores preditos
valores_preditos = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
```

## 🧩 Implementação da matriz de confusão

A função abaixo cria manualmente uma matriz de confusão binária, considerando uma classe positiva definida pelo usuário.

```python
def gerar_matriz_de_confusao(reais, preditos, labels):
    if len(labels) > 2:
        return None
    
    if len(reais) != len(preditos):
        return None
    
    true_class = labels[0]

    # Valores verdadeiros
    tp = 0
    tn = 0

    # Valores falsos
    fp = 0
    fn = 0

    for (indice, v_real) in enumerate(reais):
        v_predito = preditos[indice]

        if v_real == true_class:
            tp += 1 if v_predito == v_real else 0
            fp += 1 if v_predito != v_real else 0
        else:
            tn += 1 if v_predito == v_real else 0
            fn += 1 if v_predito != v_real else 0

    return np.array([[tp, fp], [fn, tn]])
```

## ▶️ Execução do código

```python
print(
    gerar_matriz_de_confusao(
        reais=valores_reais,
        preditos=valores_preditos,
        labels=[1, 0]
    )
)
```

## 🚀 Como executar o projeto
1. Certifique-se de ter o Python, o Git e o VS Code instalados;

2. Clone este repositório com o seguinte comando no terminal do VS Code:
```bash
git clone https://github.com/luanalima77/Como-mostrar-matriz-de-confusao-no-terminal.git
```

3. Acesse a pasta do projeto por meio do seguinte comando no terminal do VS Code:
```bash
cd Como-mostrar-matriz-de-confusao-no-terminal
```

4. Instale, por meio do comando abaixo, o numpy no terminal do VS Code:
```bash
pip install numpy
```

5. Execute o arquivo principal colocando o seguinte comando no terminal do VS Code:
```python
python main.py
```

### ✍️ Projeto com fins educacionais