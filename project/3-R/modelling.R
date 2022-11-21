library(h2o)
library(tidyverse)
h2o.init()


df <- h2o.importFile("project/1-data/train_data.csv")

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
                  max_runtime_secs = 200)

aml@leaderboard

model <- aml@leader

model <- h2o.getModel("GBM_1_AutoML_1_20221121_221738")
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
  write_csv("./project/5-predictions/predictions1.csv")

### ID, Y

h2o.saveModel(model, "./project/4-model/", filename = "my_model")

model <- h2o.loadModel("./project/4-model/my_model")


h2o.shutdown()
