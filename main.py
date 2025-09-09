import numpy as np

#Arrays para realizar a matriz de confusão.
valores_reais = [1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
valores_preditos = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]

def gerar_matriz_de_confusao(reais, preditos, labels):
    if len(labels)>2:
        return None
    
    if len(reais) != len(preditos):
        return None
    
    true_class = labels[0]

    #Valores verdadeiros.
    tp = 0
    tn = 0

    #Valores falsos.
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

        return np.array[
            [tp, fp],
            [fn, tn]
        ]
    

        