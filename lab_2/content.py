# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np


def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.
    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    X = X.toarray()
    X_train = X_train.toarray()
    X_train_transposed = X_train.transpose()

    res = (1 - X) @ X_train_transposed
    helper = (1 - X_train).transpose()
    res2 = X @ helper
    return np.add(res, res2)


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2
    251 / 5000
Wyniki tłumaczenia
Zwraca indeksy, które posortowałyby tablicę.

 pośrednie sortowanie, używające algorytmu określonego
 przez słowo kluczowe kind. Zwraca tablicę indeksów o tym samym kształcie,
  co dane tego indeksu wzdłuż danej osi w posortowanej kolejności
    """

    sorted_tab_indices = Dist.argsort(kind='mergesort')
    return y[sorted_tab_indices]


def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """

    min_length = 4
    result_matrix = []
    for i in range(np.shape(y)[0]):
        helper = []
        for j in range(k):
            helper.append(y[i][j])
        line = np.bincount(helper, None, min_length)
        result_matrix.append([line[0] / k, line[1] / k, line[2] / k, line[3] / k])
    return result_matrix
    pass




def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """

    numOfRows = len(p_y_x)
    numOfCols = len(p_y_x[0])
    res = 0
    for i in range(0, numOfRows):
        if (numOfCols - np.argmax(p_y_x[i][::-1]) - 1) != y_true[i]:
            res += 1
    return res / numOfRows

def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """

    bestKIndex = 0

    errors = []

    sortedLabelsArr = sort_train_labels_knn(hamming_distance(X_val, X_train), y_train)

    for i in range(len(k_values)):

        error = classification_error(p_y_x_knn(sortedLabelsArr, k_values[i]), y_val)

        errors.append(error)

        if (errors[bestKIndex] > error):
            bestKIndex = i

    return (errors[bestKIndex], k_values[bestKIndex], errors)


def estimate_a_priori_nb(y_train):
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """

    res = np.array(np.bincount(y_train) / len(y_train))
    return res


def estimate_p_x_y_nb(X_train, y_train, a, b):
    """
    #ilość wystąpień kazdej wartości w tablicy pozytywnych liczb całk
    gdzie dana wartosc jest reprezentowana przez indeks)
    Wyznacz rozkład prawdopodobieństwa p(x|y) zakładając, że *x* przyjmuje
    wartości binarne i że elementy *x* są od siebie niezależne.

    :param X_train: dane treningowe NxD
    :param y_train: etykiety klas dla danych treningowych 1xN
    :param a: parametr "a" rozkładu Beta
    :param b: parametr "b" rozkładu Beta
    :return: macierz prawdopodobieństw p(x|y) dla obiektów z "X_train" MxD.
    """
    points_number = 4
    result_matrix = []
    vector = np.array(np.bincount(y_train))
    denominator = (vector + a + b - 2)

    def numerator(X_train, y_train, d): # wystapienie 1
        X_train = X_train.toarray()
        occurences = [0, 0, 0, 0]
        for i in range(np.shape(y_train)[0]):
            if X_train[i][d] == 1:
                occurences[y_train[i]] += 1
        occurences_add = [(x + (a - 1)) for x in occurences]
        return np.array(np.squeeze(occurences_add))

    for k in range(points_number):
        line = []
        for d in range(np.shape(X_train)[1]):
            line.append((numerator(X_train, y_train, d)[k]) / denominator[k])
        result_matrix.append(line)

    return np.array(result_matrix)


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) dla każdej z klas z wykorzystaniem
    klasyfikatora Naiwnego Bayesa.

    :param p_y: wektor prawdopodobieństw a priori 1xM
    :param p_x_1_y: rozkład prawdopodobieństw p(x=1|y) MxD
    :param X: dane dla których beda wyznaczone prawdopodobieństwa, macierz NxD
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" NxM
    """
    X = X.toarray()  # rozkład dwupunktowy
    p_x_1_y_rev = 1 - p_x_1_y  #1 ->0 , 0 -> 1
    X_rev = 1 - X
    res = []
    for i in range(X.shape[0]):  # X.shape[0] number of rows
        success = p_x_1_y ** X[i, ] # teta probab. for d-word occurences in the file
        fail = p_x_1_y_rev ** X_rev[i, ]
        a = np.prod(success * fail, axis=1) * p_y # numerator
        sum = np.sum(a) # dominator
        res.append(a / sum)
    return np.array(res)


def model_selection_nb(X_train, X_val, y_train, y_val, a_values, b_values):
    """
    Wylicz bład dla różnych wartości *a* i *b*. Dokonaj selekcji modelu Naiwnego
    Byesa, wyznaczając najlepszą parę wartości *a* i *b*, tj. taką, dla której
    wartość błędu jest najniższa.
    
    :param X_train: zbiór danych treningowych N2xD
    :param X_val: zbiór danych walidacyjnych N1xD
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrów "a" do sprawdzenia
    :param b_values: lista parametrów "b" do sprawdzenia
    :return: krotka (best_error, best_a, best_b, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_a" i "best_b" to para parametrów
        "a" i "b" dla której błąd był najniższy, a "errors" - lista wartości
        błędów dla wszystkich kombinacji wartości "a" i "b" (w kolejności
        iterowania najpierw po "a_values" [pętla zewnętrzna], a następnie
        "b_values" [pętla wewnętrzna]).
    """
    errors = np.ones((len(a_values), len(b_values)))
    estimated_p_y = estimate_a_priori_nb(y_train)
    best_a = 0
    best_b = 0
    best_error = np.inf
    for i in range(len(a_values)):
        for j in range(len(b_values)):
            error = classification_error(
                p_y_x_nb(estimated_p_y, estimate_p_x_y_nb(X_train, y_train, a_values[i], b_values[j]), X_val), y_val)
            errors[i][j] = error
            if error < best_error:
                best_a = a_values[i]
                best_b = b_values[j]
                best_error = error
    return best_error, best_a, best_b, errors
