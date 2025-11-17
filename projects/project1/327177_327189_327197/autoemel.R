regr <- read.csv("wyniki_regr.csv")
xgb <- read.csv("wyniki_xgb.csv")
knn <- read.csv("wyniki_knn.csv")

D_regr <- regr$tuna_optuna-regr$tuna_random
D_xgb <- xgb$tuna_optuna-xgb$tuna_random
D_knn <- knn$tuna_optuna-knn$tuna_random

w_regr <- wilcox.test(D_regr, alternative = "greater", mu=0)
w_xgb <- wilcox.test(D_xgb, alternative = "greater", mu=0)
w_knn <- wilcox.test(D_knn, alternative = "greater", mu=0)

#D_total = c(D_regr, D_xgb, D_knn)
#wilcox.test(D_total, alternative = "greater", mu=0)
w_regr$p.value
w_knn$p.value
w_xgb$p.value
