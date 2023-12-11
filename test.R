set.seed(14)
task = as_task_classif(train1, target = "smoking")
learner = lrn("classif.xgboost")
learner$param_set$set_values(
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
  learner = learner,
  resampling = rsmp("holdout"),
  measures = msr("classif.acc"),
  terminator = trm("evals", n_evals = 50)
)
tuner = tnr("hyperband", eta = 2, repetitions = 1)
tuner$optimize(instance)
learner$param_set$values = instance$result_learner_param_vals
learner$train(task)
prediction = learner$predict_newdata(test)
df1 = data.frame(prediction$response)
df2 = test[1]
df = cbind(df2, df1)
df = df %>% 
  rename(smoking = prediction.response)
write.csv(df, file = "result2.csv")

set.seed(15)
task = as_task_classif(train1, target = "smoking")
learner = lrn("classif.catboost")
learner$param_set$set_values(
  learning_rate = to_tune(0.001, 0.1), 
  iterations = to_tune(p_int(256, 2056, tags = "budget")), 
  depth = to_tune(1,15), 
  bagging_temperature = to_tune(0.1, 0.5), 
  metric_period = to_tune(50, 200), 
  task_type = "GPU"
)
instance = ti(
  task = task,
  learner = learner,
  resampling = rsmp("holdout"),
  measures = msr("classif.acc"),
  terminator = trm("evals", n_evals = 50)
)
tuner = tnr("hyperband", eta = 2, repetitions = 2)
tuner$optimize(instance)
learner$param_set$values = instance$result_learner_param_vals
learner$train(task)
prediction = learner$predict_newdata(test)
df1 = data.frame(prediction$response)
df2 = test[1]
df = cbind(df2, df1)
df = df %>% 
  rename(smoking = prediction.response)
write.csv(df, file = "result2.csv")
