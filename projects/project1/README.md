# Wstęp

Celem jest przeanalizowanie tunowalności hiperparametrów 3 wybranych algorytmów uczenia maszynowego (np. xgboost, random forest, elastic net) na co najmniej 4 zbiorach danych. Do tunowania modeli należy wykorzystać min. 2 różne techniki losowania punktów (opisane dokładniej poniżej).

### Metody samplingu

1.  Co najmniej jedna metoda powinna się opierać na wyborze punktów z rozkładu jednostajnego. Przykładami mogą być:

-   Uniform grid search
-   Random Search
s
  **Uwaga: dla wszystkich zbiorów danych w tym kroku powinniśmy korzystać z tej samej ustalonej siatki hiperparametrów dla każdego algorytmu.**

2.  Co najmniej jedna technika powinna opierać się na technice bayesowskiej

-   Bayes Optimization
    
    _warto wykorzystać pakiet SMAC3 do dostosowania metody, ale może być też scikit-optimize i funkcja BayesSearchCV_
    

Wyniki z poszczególnych metod tunowania (historia tuningu) powinny być wykorzystywane do wyznaczenia tunowalności algorytmów.

Tunowalność algorytmów i hiperparametrów powinna być określona zgodnie z definicjami w [Tunability: Importance of Hyperparameters of Machine Learning Algorithms](https://jmlr.org/papers/volume20/18-444/18-444.pdf). Zadaniem jest wyznaczenie nowej domyślnej konfiguracji hiperparametrów (dla każdego z rozważanych algorytmów), która osiąga średnio najlepsze wyniki dla wszystkich zbiorów danych. W następnym kroku należy przeanalizować rozkład różnic pomiędzy miarą otrzymaną dla wcześniej wyznaczonej domyślnej konfiguracji, a miarami uzyskanymi dla innych testowanych konfiguracji.



### Punkty, które należy rozważyć

Na podstawie wyników zgromadzonych w eksperymencie opisanym w sekcji [Wstęp] (#wstep) należy opisać i przeanalizować wyniki pod kątem: 

1.  ile iteracji każdej metody potrzebujemy żeby uzyskać stabilne wyniki optymalizacji
    
2.  określenie zakresów hiperparametrów dla poszczególnych modeli - motywacja wynikająca z literatury
    
3.  tunowalność poszczególnych algorytmów 

*lub* 

3. tunowalność poszczególnych hiperparametrów
        
4.  czy technika losowania punktów wpływa na różnice we wnioskach w punkcie 3 dotyczących tunowalności algorytmów i hiperparametrów - Odpowiedź na pytanie czy występuje bias sampling.

    

### Potencjalne punkty rozszerzające PD
-   Analiza wpływu wielkości zbioru danych – przeprowadzenie eksperymentu polegającego na powtarzaniu procedury tunowania dla podzbiorów danych o różnych rozmiarach (np. 25%, 50%, 75% i 100% oryginalnego zbioru). Celem jest zbadanie, jak zmienia się rozkład różnic miar (między konfiguracją referencyjną a pozostałymi) w zależności od liczby dostępnych obserwacji. 
-   Zastosowanie testów statystycznych _do porównania różnic wyników pomiędzy technikami losowania hiperparametrów_
-   Zastosowanie **[Critical Difference Diagrams](https://github.com/hfawaz/cd-diagram#critical-difference-diagrams) -** w przypadku zastosowania większej liczby technik losowania punktów
-   Zaproponowanie wizualizacji i analiz wyników innych niż użyte w cytowanym artykule

### Oczekiwany wynik

Na przygotowanie rozwiązania projektu będą składały się następujące elementy:

-   raport opisujący wykorzystane metody i wyniki eksperymentów dla obu technik (maksymalnie 4 strony A4) - max. 35 pkt.
-   rozmowa zespołu z prowadzącym na temat wyników i wniosków płynących z Projektu 1 - max. 5 pkt.


  
### Oddanie projektu

Wszystkie punkty z sekcji _Szczegóły rozwiązania_ należy umieścić w katalogu o nazwie `NUMERINDEKSU1_NUMERINDEKSU2` lub `NUMERINDEKSU1_NUMERINDEKSU2_NUMERINDEKSU3`. Tak przygotowany **katalog należy umieścić na repozytorium przedmiotu w folderze `projects/project1`.**



### Terminy  

Termin oddania pracy domowej to **18.11.2025 EOD**.
Oprócz oddania raportu, w dniu **20.11.2025** w czasie zajęć projektowych zostanie przeprowadzona rozmowa z każdy zespołem i prowadzący może zadać pytania odnośnie sposobu realizacji proejktu.
