# Projekt: Mini-AutoML dla Danych Tabelarycznych

## Cel Projektu

Celem projektu jest stworzenie **uproszczonego systemu AutoML**, który umożliwi automatyczne wykonanie zadania **klasyfikacji binarnej** na dowolnym dostarczonym zbiorze danych. System ma skupiać się na **skonstruowaniu i wykorzystaniu portfolio modeli**.

## Wymagania Systemu Mini-AutoML

System musi spełniać następujące kluczowe wymagania:

- **Portfolio Modeli:** Posiadać portfolio zawierające **maksymalnie 50 konfiguracji modeli**.
- **Selekcja:** Potrafić wybrać **najlepszy model (lub modele)** na podstawie dostarczonych danych treningowych (X_train, y_train).
- **Ensembling:** Umożliwiać wykonanie **ensemblingu** (łączenia predykcji) z wykorzystaniem **do 5 modeli**.
- **Powtarzalność:** Być w pełni **powtarzalny** i możliwy do wykonania na **dowolnym** zbiorze danych przeznaczonym do klasyfikacji binarnej.

---

## Etapy Realizacji Projektu

### Etap 1: Wstępna Selekcja Modeli (Obowiązkowa)

W celu zbudowania portfolio należy przygotować wstępną selekcję kandydatów. Dopuszczalne są dwa podejścia:

### Opcja A: Lokalny Screening Modeli

Przygotowanie wybranej liczby modeli i konfiguracji Hiperparametrów (HP) z wykorzystaniem:

- Podziału na zbiór treningowy/walidacyjny (`train/validation split`) lub Walidacji Krzyżowej (`CV`).
- Danych z publicznych benchmarków (np. OpenML).

Należy przetestować:

- **Różne typy modeli** (np. `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC`, `kNN`, `CatBoost`, `LightGBM` itp.).
- **Różne zestawy hiperparametrów**.
- **Na różnych zbiorach danych** np. z OpenML.

### Opcja B: Screening na Podstawie Wyników Zewnętrznych

Wykorzystanie wyników z ogólnodostępnych źródeł zamiast kosztownych eksperymentów:

- Benchmarki OpenML (np. OpenML-CC18).
- TabArena.
- MementoML https://www.kaggle.com/mi2datalab/mementoml
- Ogólnodostępne leaderboardy modeli.

### Etap 2: Plik JSON z Konfiguracją Modeli

Należy przygotować plik o nazwie np. `models.json`, zawierający listę modeli oraz komplet ich hiperparametrów (innych niż domyślne).

**Przykład struktury pliku:**

JSON

```jsx
[
  {
    "name": "logreg_l2",
    "class": "sklearn.linear_model.LogisticRegression",
    "params": {"C": 1.0, "penalty": "l2", "solver": "liblinear"}
  },
  {
    "name": "rf_100",
    "class": "sklearn.ensemble.RandomForestClassifier",
    "params": {"n_estimators": 100, "max_depth": 10}
  },
  {
    "name": "lightgbm_150",
    "class": "lightgbm.LGBMClassifier",
    "params": {"n_estimators": 150}
  }
]

```

### Etap 3: Implementacja Systemu Mini-AutoML

Należy stworzyć główną klasę **`MiniAutoML`** z wymaganymi metodami:

1. **`__init__(self, models_config)`**
    - Przyjmuje listę modeli wczytaną z pliku JSON.
2. **`fit(self, X_train, y_train)`**
    - Wykonuje selekcję najlepszego modelu lub modeli.
    - Trenuje wybrane modele.
    - Zwraca finalny wytrenowany model lub  skonstruowany ensemble.
3. **`predict(self, X_test)`**
    - Zwraca wektor predykcji klas dla danych testowych.
4. **`predict_proba(self, X_test)`**
    - Zwraca wektor prawdopodobieństwa przynależności do klasy pozytywnej dla danych testowych.

### Sposób Wyboru Modelu

**Sposób wyboru modelu dla danych treningowych musi odbywać się w wywołaniu metody `fit` . Jest to kluczowe zadanie projektowe.** Selekcja może być przeprowadzana jako.:

- Ranking modeli oparty na walidacji krzyżowej (np. 5-krotnej CV).
- Prosty meta-learner.
- Metryka stabilności.
- Heurystyka na podstawie charakterystyk danych (np. n_features, n_samples).
- Dowolna rozsądna i uzasadniona procedura selekcji.

### Ensembling (Opcjonalny)

Ponadto w fazie postprocessingu  można zaproponować ensemble składający się z **maksymalnie 5 modeli**. Możliwe metody:

- Voting (Głosowanie: twarde lub miękkie).
- Averaging (Uśrednianie prawdopodobieństw).
- Stacking.
- Inne.

---

## Etap 4: Ocena i Ewaluacja

### Założenia dotyczące zbiorów danych:

- dane tabelaryczne, zapisane w pliku .csv;
- poszczególne zmienne w reprezentowane w kolumnach;
- pierwszy wiersz zawiera nagłówki kolumn;
- dopuszczalne typy zmiennych: zmienne ciągłe i zmienne kategoryczne;
- w danych mogą wystąpić braki danych.

### Przygotowanie do Ewaluacji

Dwa tygodnie przed terminem oddania projektu zostanie dostarczony **przykładowy zbiór danych** w celu sprawdzenia struktury i formatu danych, na których ma działać rozwiązanie. Zbiór danych jest umieszczony w [folderze](https://github.com/woznicak/2025Z-AutoML/tree/main/projects/project2/example_data).

### Ostateczna Ewaluacja

Ostateczna ocena odbędzie się:

- Na zajęciach.
- Przy użyciu **nieznanego wcześniej zbioru testowego** (o tym samym formacie danych co zbiór przykładowy).
- Przy wykorzystaniu stworzonego kodu i listy modeli.

**Wymogi formalne:**

- Plik JSON musi być **poprawny**.
- Modele muszą dać się **utworzyć i wytrenować**.
- Procedura wyboru modeli musi być **przejrzysta i powtarzalna**.

## Szczegóły Rozwiązania (Elementy do oddania)

1. **Plik JSON z konfiguracjami modeli**
    - Nazwa: `models.json`
    - Maksymalnie 50 konfiguracji.
2. **Kod systemu Mini-AutoML**
    - Nazwa: `automl.py` lub `automl.ipynb` (w zależności od formatu).
3. **Raport (PDF)**
    - **Maksymalnie 4 strony A4**.
    - Zawiera:
        - Opis etapu wyboru modeli do portfolio (Etap 1) i wyniki eksperymentów przeprowadzonych w tym etapie.
        - Omówienie metody selekcji modeli z portfolio dla nowego zbioru danych (Etap 3) i wyniki eksperymentów przeprowadzonych w tym etapie.
        - Opis ewentualnego ensemblingu.
        - Wnioski.

## Oczekiwany Wynik (Punktacja)

- Raport (30 punktów)
    
    Opis wykorzystanych metod i wyniki eksperymentów dla obu technik (maksymalnie 4 strony A4).
    
- Jakość predykcji (5 punktów)
    
    Mierzona miarą **Balanced Accuracy** na zbiorze testowym. Wyniki zostaną uszeregowane (ranking). Zespół z **najlepszym wynikiem** (najbliższym 1) otrzymuje 5 punktów, zespół z **najgorszym wynikiem** (najbliższym 0) otrzymuje 2.5 punktu. Pozostałe wyniki zostaną przeskalowane i zaokrąglone do wartości 0.1 punktu.
    

### Termin Oddania Projektu

Termin oddania projektu to **21.01.2026 EOD** (End of Day).

### Sposób Oddania

Wszystkie elementy z sekcji *Szczegóły Rozwiązania* należy umieścić w katalogu o nazwie: `NUMERINDEKSU1_NUMERINDEKSU2_NUMERINDEKSU3`. Tak przygotowany katalog należy umieścić w repozytorium przedmiotu w folderze: **`projects/project2`**.
