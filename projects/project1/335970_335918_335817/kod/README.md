# automl_project1

## 5 datasets:
- Adult(49k x 15): https://www.openml.org/search?type=data&status=active&sort=nr_of_likes&qualities.NumberOfInstances=between_10000_100000&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=between_10_100&id=1590
- Numarai(96k x 22): https://www.openml.org/search?type=data&status=active&sort=nr_of_likes&qualities.NumberOfInstances=between_10000_100000&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=between_10_100&id=23517
- MagicTelescope(19k x 12): https://www.openml.org/search?type=data&status=active&sort=nr_of_likes&qualities.NumberOfInstances=between_10000_100000&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=between_10_100&id=1120
- kr-vs-kp(3.2k x 37): https://www.openml.org/search?type=data&status=active&sort=nr_of_likes&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=between_10_100&qualities.NumberOfInstances=between_1000_10000&id=3
- mushroom(8.1k x 23): https://www.openml.org/search?type=data&status=active&sort=nr_of_likes&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=between_10_100&qualities.NumberOfInstances=between_1000_10000&id=43922

## 3 models: XGBoost, LightGBM, CatBoost

## Hyperaparameters
- CORE(for all 3 models): learning_rate, n_estimators, max_depth, subsample, colsample_bytree/feature_fraction
- XGBoost: gamma
- LightGBM: num_leaves
- CatBoost: l2_leaf_reg

## Documentations:
- xgboost: https://xgboost.readthedocs.io/en/stable/parameter.html#global-configuration
- lightgbm: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
- catboost: https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier

