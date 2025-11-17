## Zbiory danych jakich użyłyśmy
    1. https://www.kaggle.com/datasets/nezukokamaado/auto-loan-dataset
    2. https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset
    3. https://www.kaggle.com/datasets/anthonytherrien/depression-dataset
    4. https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package

## Foldery i pliki:
- Foldery z odpowiednimi nazwami ramek danych (depression, diabetes, loan, weather) zawierają orginalną ramkę danych plik, który został użyty do preprocessingu danych oraz wynikową ramkę danych używaną w dalszych etapach.
- Głównym plikiem jest 'tunning_all_models.ipynb' zawierający wszystkie kroki podjęte do wyznaczenia hiperparametrów ich domyślnych wersji oraz dalszą analizę. Funckje wykorzystywane w tym pliku są umieszczone w 'functions_tunning.py'.
- Folder results_tunning zawiera wyniki przeszukiwania siatek hiperparametrów dla każdego zbioru danych, modelu i rodzaju przeszukiwania (RandomSearch, BayesSearch). Zapis umożliwił pracę na danych bez konieczności wykonywania długiego przeszukiwania wielokrotnie.
- Folder raport zawiera końcowy raport oraz wykresy, do których odnosi się raport.