library(tidyverse)
library(mlr3)
library(mlr3verse)
library(mlr3filters)

set.seed(15)
train = read.csv("data/train.csv")
summary(train)
#train = train %>% 
#  select(-c(id, hearing.left., Urine.protein, hearing.right.))
train = train[-1]
train1 = model.matrix(~.-1, train) %>% data.frame()
task = as_task_regr(train1, target = "smoking")
#task = as_task_classif(train, target = "smoking")
test = read.csv("data/test.csv")
test1 = model.matrix(~.-1, train) %>% data.frame()
learner = lrn("regr.lightgbm")
filter = flt("importance", learner)
filter$calculate(task)
as.data.table(filter)
search_space = ps(
  learning_rate = p_dbl(0.001, 0.1), 
  num_iterations = p_int(256, 1024, tags = "budget"), 
  max_depth = p_int(1, 10), 
  num_leaves = p_int(5, 53), 
  bagging_fraction = p_dbl(0.75, 1)
)
at = auto_tuner(tuner = tnr("hyperband"), 
                learner = learner, 
                resampling = rsmp("holdout"), 
                measure = msr("classif.acc"), 
                search_space = search_space, 
                terminator = trm("evals", n_evals = 60))
at$train(task)
learner$param_set$values = at$tuning_result$learner_param_vals[[1]]
rr = resample(task, learner, resampling = rsmp("cv", folds = 10))
rr$aggregate(msr("classif.acc"))
prediction = at$predict_newdata(test)
df1 = data.frame(prediction$response)
df2 = test[1]
df = cbind(df2, df1)
df = df %>% 
  rename(smoking = prediction.response)
write.csv(df, file = "result2.csv")
#交叉验证
learner_x = lrn("regr.xgboost")
learner_l = lrn("regr.lightgbm")
learner_c = lrn("regr.catboost")
#learner_x = lrn("classif.xgboost", predict_type = "prob")
#learner_l = lrn("classif.lightgbm")
#learner_c = lrn("classif.catboost", predict_type = "prob")
#xgboost超参数调整
learner_x$param_set$set_values(
  tree_method = "hist", 
  booster = "gbtree", 
  nrounds = to_tune(p_int(256, 1024, tags = "budget")),
  eta = to_tune(1e-4, 1, logscale = TRUE),
  max_depth = to_tune(1, 20),
  colsample_bytree = to_tune(1e-1, 1),
  colsample_bylevel = to_tune(1e-1, 1),
  lambda = to_tune(1e-3, 1e3, logscale = TRUE),
  alpha = to_tune(1e-3, 1e3, logscale = TRUE),
  subsample = to_tune(1e-1, 1)
)
instance = ti(
  task = task,
  learner = learner_x,
  resampling = rsmp("holdout"),
  measures = msr("regr.rmse"),
  terminator = trm("evals", n_evals = 50)
)
tuner = tnr("hyperband", eta = 2, repetitions = 1)
tuner$optimize(instance)
learner_x$param_set$values = instance$result_learner_param_vals
learner_x$train(task)
#lightgbm超参数调整
learner_l$param_set$set_values(
  learning_rate = to_tune(0.001, 0.1),
  num_iterations = to_tune(p_int(256, 1024, tags = "budget")),
  max_depth = to_tune(1, 10),
  num_leaves = to_tune(5, 53),
  bagging_fraction = to_tune(0.75, 1)
)
instance = ti(
  task = task,
  learner = learner_l,
  resampling = rsmp("holdout"),
  measures = msr("regr.rmse"),
  terminator = trm("evals", n_evals = 50)
)
tuner = tnr("hyperband", eta = 2, repetitions = 2)
tuner$optimize(instance)
learner_l$param_set$values = instance$result_learner_param_vals
learner_l$train(task)
#catboost超参数调参
learner_c$param_set$set_values(
  learning_rate = to_tune(0.001, 0.1), 
  iterations = to_tune(p_int(256, 2056, tags = "budget")), 
  depth = to_tune(1,15), 
  bagging_temperature = to_tune(0.1, 0.5), 
  metric_period = to_tune(50, 200), 
  task_type = "GPU"
)
instance = ti(
  task = task,
  learner = learner_c,
  resampling = rsmp("holdout"),
  measures = msr("regr.rmse"),
  terminator = trm("evals", n_evals = 50)
)
tuner = tnr("hyperband", eta = 2, repetitions = 2)
tuner$optimize(instance)
learner_c$param_set$values = instance$result_learner_param_vals
learner_c$train(task)
gstack_test = gunion(list(
  po("learner_cv", learner_x, id = "xgb"),
  po("learner_cv", learner_l, id = "lgb"), 
  po("learner_cv", learner_c, id = "ctb"))) %>>%
  po("featureunion")
resampled_1 = gstack_test$train(task)[[1]]
cv5_x = resample(task, learner_x, rsmp("cv", folds = 5))
cv5_l = resample(task, learner_l, rsmp("cv", folds = 5))
cv5_c = resample(task, learner_c, rsmp("cv", folds = 5))
resample_2 = data.table(
  truth = cv5_x$prediction()$truth, 
  x = cv5_x$prediction()$response, 
  l = cv5_l$prediction()$response
)
resample_2$truth = as.numeric(resample_2$truth)
library(Matrix)
library(glmnet)
x <- as.matrix(resample_2[, -1])
y <- as.matrix(resample_2[, 1])
lambda = cv.glmnet(x = x, y = y, alpha = 0)
bestlambda = lambda$lambda.min
learner_b = lrn("regr.glmnet", alpha = 0)
search_space_b = ps(s = p_dbl(lower = 0.001, upper = 2))
at_b = auto_tuner(
  tuner = tnr("grid_search"),
  learner = learner_b,
  resampling = rsmp("cv", folds = 5), 
  measure = msr("regr.rmse"), 
  search_space = search_space_b, 
  terminator = trm("evals", n_evals = 50)
)
at_b$train(resampled_1)
hp_b = at_b$tuning_result$learner_param_vals[[1]]
learner_b$param_set$values = hp_b
learner_b$param_set$values$lambda = bestlambda
glearner = gunion(list(
  po("learner_cv", learner_x, id = "xgb"),
  po("learner_cv", learner_l, id = "lgb"), 
  po("learner_cv", learner_c, id = "ctb"))) %>>%
  po("featureunion") %>>%
  learner_b %>% 
  as_learner()
glearner$train(task)
gprediction = glearner$predict_newdata(test)
df1 = data.frame(gprediction$response)
df2 = test[1]
max = cbind(df2,df1)
max = max %>%
  rename(smoking = gprediction.response)
write.csv(max, file = "result1.csv")

