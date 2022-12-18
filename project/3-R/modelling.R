library(h2o)
library(tidyverse)
h2o.init(
  max_mem_size = "4g"
)

df <- h2o.importFile("./project/1-data/train_data.csv")

# Susipažiname su duomenimis ir paleidžiame automl
df
class(df)
class(data)
summary(df)

y <- "y"
x <- setdiff(names(df), c(y, "id"))

df$y <- as.factor(df$y)

summary(df)

splits <- h2o.splitFrame(df, c(0.6,0.2), seed=123)
train  <- h2o.assign(splits[[1]], "train") # 60%
valid  <- h2o.assign(splits[[2]], "valid") # 20%
test   <- h2o.assign(splits[[3]], "test")  # 20%


aml <- h2o.automl(x = x,
                  y = y,
                  training_frame = train,
                  validation_frame = valid,
                  exclude_algos = c("GLM", "DeepLearning", "GBM"),
                  max_runtime_secs = 600)

aml@leaderboard

model <- aml@leader

# Iš automl sąrašo pasirenkame geriausią modelį ir pritaikome prognozavimui
model <- h2o.getModel("StackedEnsemble_BestOfFamily_2_AutoML_1_20221211_134359")

h2o.varimp_plot(model)

h2o.performance(model, train = TRUE)

perf_valid <- h2o.performance(model, valid = TRUE)

h2o.auc(perf_valid)

h2o.performance(model, newdata = test)

h2o.varimp_plot(model)

h2o.auc(perf_valid)
plot(perf_valid, type = "roc")

test_data <- h2o.importFile("./project/1-data/test_data.csv")
h2o.performance(model, newdata = test_data)

predictions <- h2o.predict(model, test_data)

predictions %>%
  as_tibble() %>%
  mutate(id = row_number(), y = p0) %>%
  select(id, y) %>%
  write_csv("./project/5-predictions/predictions4.csv")

# Išsaugome turimą modelį, kaip "my_model84Æ

h2o.saveModel(model, "./project/4-model/", filename = "my_model84")

# Užsikrauname prieš tai naudotą modelį, pasirenkame XGBoost automl atrinktą modelį
# ir jį tobuliname.
model <- h2o.loadModel("./project/4-model/my_model84")

modeld@parameters

xgb <- h2o.xgboost(x = x,
                   y = y,
                   model_id = "XGBoost_1_AutoML_1_20221125_91712",
                   training_frame = train,
                   validation_frame = valid,
                   booster = "dart",
                   stopping_rounds = 3, 
                   stopping_metric = "AUC",
                   keep_cross_validation_models = TRUE,
                   distribution = 'bernoulli',
                   categorical_encoding = "OneHotInternal",
                   ntrees = 70,
                   max_depth = 17,
                   min_rows = 2,
                   nfolds = 5,
                   dmatrix_type = "dense",
                   backend = "cpu",
                   seed = "-0001")

perf <- h2o.performance(xgb, newdata = test)

predictions <- h2o.predict(xgb, test_data)

predictions %>%
  as_tibble() %>%
  mutate(id = row_number(), y = p0) %>%
  select(id, y) %>%
  write_csv("./project/5-predictions/predictions_last.csv")

# Išsaugome gautą XGBoost modelį.
h2o.saveModel(xgb, "./project/4-model/", filename = "my_model85_last")


model <- h2o.loadModel("./project/4-model/my_model85_last")

model@parameters


h2o.shutdown()
