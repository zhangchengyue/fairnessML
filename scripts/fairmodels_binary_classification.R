setwd("D:/MScAC信息/Ji Lab/fairnessML/mental-health-in-tech-project-main")
#loading libraries
library(tidyverse)
library(caret)
library(ggplot2)
source("./code/script/functions/funs_do_feature_selection.R")
source("./code/script/functions/funs_do_caret_modeling.R")

#loading data
clean_df <- read.csv("./data/processed_data/mental-health-in-tech-2016-modeling.csv")

dim(clean_df)
head(clean_df)


###########################################################################
####### 为了试试fairmodels的包，我们先假设这是一个binary classification problem
# 找到所有"Maybe"的索引
maybe_indices <- which(clean_df$mh_discussion_negative == "Maybe")

# 随机打乱这些索引
set.seed(123)  # 为了保证结果可重复，我们可以设置随机种子
shuffled_indices <- sample(maybe_indices)

# 将一半的"Maybe"转换为"No"，另一半转换为"Yes"
half <- length(shuffled_indices) %/% 2
clean_df$mh_discussion_negative[shuffled_indices[1:half]] <- "No"
clean_df$mh_discussion_negative[shuffled_indices[(half + 1):length(shuffled_indices)]] <- "Yes"

###########################################################################


#train/test split-------------------------------------------------------------------------------------
trainIndex <- caret::createDataPartition(clean_df$mh_discussion_negative, 
                                         p = .75, 
                                         list = FALSE, 
                                         times = 1)
trainData <- clean_df[ trainIndex,]
testData <- clean_df[-trainIndex,]


#####################################
# Binary classification的话用这个：
trainSmote <- trainData
#####################################


###########################################
### Multiclass classification的话用这个：
# #class imbalance-------------------------------------------------------------------------------------
# PlotCatDist(trainData, trainData$mh_discussion_negative)
# 
# library(performanceEstimation)
# 
# trainSmote <- performanceEstimation::smote(mh_discussion_negative ~ ., data  = trainData)                         
# table(trainSmote$mh_discussion_negative)
###########################################



#linear svm classification--------------------------------------------------------------------------------
svmGrid <-  expand.grid(C = c(.001, .01, .1, 0.5, 1.0))


# fitting training data w/ sbf feature selection
# SelectFeatures和FitTrainingModel在funs_do_caret_modeling.R里,但是它们运行一直不成功
# trainSmote$mh_discussion_negative <- as.factor(trainSmote$mh_discussion_negative)
# ft_selected <- SelectFeatures(svmGrid, "svmLinear", trainSmote)
# svmMod <- FitTrainingModel(svmGrid, "svmLinear", trainSmote[,c(ft_selected, "mh_discussion_negative")])


# 所以用下面的：
# 1. 注意"mh_discussion_negative"是factor
# 2. 注意不能有na
################################################
# # Feature selection with SBF (Selection by Filtering)
trainSmote$mh_discussion_negative <- as.factor(trainSmote$mh_discussion_negative)

fitControl_mod <- trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 10,
                               classProbs = TRUE)

fitsbfControl <- sbfControl(method = "LGOCV",
                            number = 5,
                            p = .8,
                            functions = rfSBF,
                            saveDetails = TRUE)

ft_selected <- sbf(form = mh_discussion_negative ~ .,
                   data = trainSmote,
                   method = "svmLinear",
                   trControl = fitControl_mod,
                   tuneGrid = svmGrid,
                   sbfControl = fitsbfControl)
prop_included <- rowMeans(sapply(ft_selected$variables,function(i)ft_selected$coefnames %in% i))
selected <- ft_selected$coefnames[prop_included > 0.7]






###################################
#                                 #
#             SVM                 #
#                                 #
###################################

# ?caret::train()
svmMod <- train(mh_discussion_negative ~ .,
                data = trainSmote[, c(selected,
                                      "mh_discussion_negative")],
                method = "svmLinear",
                trControl = fitControl_mod,
                na.action=na.exclude,
                verbose = FALSE,
                tuneGrid = svmGrid)
################################################

#plotting training results
trellis.par.set(caretTheme())
ggplot(svmMod)  
densityplot(svmMod, pch = "|") # Looks like a bell shape

# FitTestModel在funs_do_caret_modeling.R里
# 注意：需要是factor
testData$mh_discussion_negative <- as.factor(testData$mh_discussion_negative)
#fitting test data
svm_test_results <- FitTestModel(svmMod, testData)
svm_test_results

#variable importance per model
plot(varImp(svmMod), top=20)


library(fairmodels)
library(DALEX)

train_copy <- trainSmote
train_copy$mh_discussion_negative <- as.character(train_copy$mh_discussion_negative)
# 把No变成0，Yes变成1
train_copy$mh_discussion_negative[train_copy$mh_discussion_negative == "No"] <- 0
train_copy$mh_discussion_negative[train_copy$mh_discussion_negative == "Yes"] <- 1
train_copy$mh_discussion_negative <- as.factor(train_copy$mh_discussion_negative)


y_numeric <- as.numeric(train_copy$mh_discussion_negative) - 1
y_numeric

########## Fairness Check for SVM########

# Build explainer
svm_explainer <- DALEX::explain(svmMod, data = train_copy, 
                                y = y_numeric, 
                                label = "svm",
                                colorize = FALSE)
# ?DALEX::explain
svm_explainer

# fairness check
# 查一下protected和privileged是什么东西，然后把对应的内容放进去
# https://modeloriented.github.io/fairmodels/articles/Basic_tutorial.html

# <protected>: Protected variable (also called sensitive attribute), containing privileged and unprivileged groups

# <privileged>: one value of <protected>, in regard to what subgroup parity loss is calculated

train_copy$genderFemale <- as.factor(train_copy$genderFemale)
train_copy$genderMale <- as.factor(train_copy$genderMale)

# Interpretation: If bars reach red field on the left it means that there is 
# bias towards certain unprivileged subgroup. If they reach one on the right 
# it means bias towards privileged
svm_fobject <- fairmodels::fairness_check(svm_explainer,            # explainer
                                          protected = train_copy$genderMale,   # protected variable as factor
                                          privileged = 1,                      # level in protected variable, potentially more privileged
                                          cutoff = 0.5,                        # cutoff - optional, default = 0.5
                                          colorize = FALSE)                         

print(svm_fobject, colorize = FALSE)
plot(svm_fobject)

# plot density
plot_density(svm_fobject)

# Metric scores plot
plot(metric_scores(svm_fobject))


# What consists of fairness object?
svm_fobject$parity_loss_metric_data



# TODO: 
# 1. Try more bias mitigation techniques
# 2. Think about what metrics should be used in our case (refer to the fairness tree)

### Pre-processing Bias Mitigation techniques #####
##### 1. Reweighting

train_copy$mh_discussion_negative <- as.numeric(train_copy$mh_discussion_negative) - 1

weights <- reweight(protected = train_copy$genderMale, y = train_copy$mh_discussion_negative)

svmMod_w <- train(mh_discussion_negative ~ .,
                  data = trainSmote[, c(selected, "mh_discussion_negative")],
                  method = "svmLinear",
                  trControl = fitControl_mod,
                  na.action=na.exclude,
                  verbose = FALSE,
                  weights = weights,
                  importance = "impurity",
                  tuneGrid = svmGrid)


# Build explainer after bias mitigation by reweighing
svm_explainer_w <- DALEX::explain(svmMod_w, 
                                  data = train_copy, 
                                  y = y_numeric, 
                                  label = "svm_weighted",
                                  colorize = FALSE)

svm_fobject <- fairmodels::fairness_check(svm_fobject, svm_explainer_w, verbose = FALSE)

plot(svm_fobject)



#### 2. Disparate Impact Remover (Not appropriate in our case, because all columns
# are categorical indices, not numeric values)

?disparate_impact_remover



#### 3. Resampling
# svm_explainer <- DALEX::explain(svmMod, data = train_copy, 
#                                 y = y_numeric, 
#                                 label = "svm",
#                                 colorize = FALSE)

# There are two ways of resampling: uniform and preferential.
uniform_indexes      <- resample(protected = train_copy$genderMale,
                                 y = y_numeric)


preferential_indexes <- resample(protected = train_copy$genderMale,
                                 y = y_numeric,
                                 type = "preferential",
                                 probs = svm_explainer$y_hat)
?resample

set.seed(1)
svmMod_u <- train(mh_discussion_negative ~ .,
                  data = trainSmote[uniform_indexes, 
                                    c(selected, "mh_discussion_negative")],
                  method = "svmLinear",
                  trControl = fitControl_mod,
                  na.action=na.exclude,
                  verbose = FALSE,
                  importance = "impurity",
                  tuneGrid = svmGrid)


# Build explainer after bias mitigation by reweighing
svm_explainer_u <- DALEX::explain(svmMod_u, 
                                  data = train_copy, 
                                  y = y_numeric, 
                                  label = "svm_uniform",
                                  colorize = FALSE)

set.seed(1)
svmMod_p <- train(mh_discussion_negative ~ .,
                  data = trainSmote[preferential_indexes, 
                                    c(selected, "mh_discussion_negative")],
                  method = "svmLinear",
                  trControl = fitControl_mod,
                  na.action=na.exclude,
                  verbose = FALSE,
                  importance = "impurity",
                  tuneGrid = svmGrid)


# Build explainer after bias mitigation by reweighing
svm_explainer_p <- DALEX::explain(svmMod_p, 
                                  data = train_copy, 
                                  y = y_numeric, 
                                  label = "svm_preferential",
                                  colorize = FALSE)

svm_fobject <- fairness_check(svm_fobject, svm_explainer_u, svm_explainer_p, 
                          verbose = FALSE)
plot(svm_fobject)




#### Post-processing Bias Mitigation Techniques #####
#### 1. ROC pivot
set.seed(1)

svm_explainer_r <- roc_pivot(svm_explainer,
                             protected = train_copy$genderMale,
                             privileged = "1")


svm_fobject <- fairness_check(svm_fobject, svm_explainer_r, 
                          label = "svm_roc",  # label as vector for explainers
                          verbose = FALSE) 

plot(svm_fobject)

### Check Bias-Accuracy Trade-off
?performance_and_fairness
svm_paf <- performance_and_fairness(svm_fobject, fairness_metric = "ACC",
                                performance_metric = "accuracy")
svm_paf
plot(svm_paf)
# The y-axis denotes the loss. The goal is to minimize the loss.

###################################
#                                 #
#          Random Forest          #
#                                 #
###################################

###########################################
#Random Forest classification--------------------------------------------------------------------------------
rfGrid <-  expand.grid(mtry = c(2, 3, 4, 5),
                       splitrule = c("gini", "extratrees"),
                       min.node.size = c(1, 3, 5))

#fitting training data w/ sbf feature selection
# rf_ft_selected <- SelectFeatures(svmGrid, "ranger", trainSmote)
# rfMod <- FitTrainingModel(rfGrid, "ranger", trainSmote[,c(rf_ft_selected,"mh_discussion_negative")])
################################################
# 上面代码还是搞不定。用这个吧
rf_selected <- sbf(form = mh_discussion_negative ~ .,
                   data = trainSmote,
                   method = "ranger",
                   trControl = fitControl_mod,
                   tuneGrid = rfGrid,
                   sbfControl = fitsbfControl)

rf_prop_included <- rowMeans(sapply(rf_selected$variables,function(i)rf_selected$coefnames %in% i))
rf_ft_selected <- rf_selected$coefnames[rf_prop_included > 0.7]

rfMod <- train(mh_discussion_negative ~ .,
               data = trainSmote[, c(rf_ft_selected, "mh_discussion_negative")],
               method = "ranger",
               trControl = fitControl_mod,
               na.action=na.exclude,
               verbose = FALSE,
               importance = "impurity",
               tuneGrid = rfGrid)



################################################

rfMod

#plotting training results
trellis.par.set(caretTheme())
ggplot(rfMod)  
densityplot(rfMod, pch = "|")

#fitting test data
rf_test_results <- FitTestModel(rfMod, testData)
rf_test_results

################################################
#variable importance per model
plot(varImp(svmMod), top=20)
plot(varImp(rfMod), top=20)

#direct model performance comparison
resamps <- resamples(list(LinearSVM = svmMod,
                          RF = rfMod))
resamps
summary(resamps)



difValues <- diff(resamps)
difValues
summary(difValues)

trellis.par.set(caretTheme())
bwplot(difValues, layout = c(3, 1))


########## Fairness Check for Random Forest########

# explainer
rf_explainer <- DALEX::explain(rfMod, data = train_copy, 
                               y = y_numeric, 
                               label = "rf",
                               colorize = FALSE)
# ?DALEX::explain
rf_explainer

# fairness check
# 查一下protected和privileged是什么东西，然后把对应的内容放进去
# https://modeloriented.github.io/fairmodels/articles/Basic_tutorial.html

train_copy$genderFemale <- as.factor(train_copy$genderFemale)
train_copy$genderMale <- as.factor(train_copy$genderMale)

# Interpretation: If bars reach red field on the left it means that there is 
# bias towards certain unprivileged subgroup. If they reach one on the right 
# it means bias towards privileged
rf_fobject <- fairmodels::fairness_check(rf_explainer,            # explainer
                          protected = train_copy$genderMale,   # protected variable as factor
                          privileged = 1,                      # level in protected variable, potentially more privileged
                          cutoff = 0.5,                        # cutoff - optional, default = 0.5
                          colorize = FALSE)                         

print(rf_fobject, colorize = FALSE)
plot(rf_fobject)

# plot density
plot_density(rf_fobject)

# Metric scores plot
plot(metric_scores(rf_fobject))


# What consists of fairness object?
rf_fobject$parity_loss_metric_data


###### Pre-processing Bias Mitigation
### 1. Reweighting

## Already reweighted above.
# train_copy$mh_discussion_negative <- as.numeric(train_copy$mh_discussion_negative) - 1
# 
# weights <- reweight(protected = train_copy$genderMale, y = train_copy$mh_discussion_negative)

rfMod_w <- train(mh_discussion_negative ~ .,
               data = trainSmote[, c(rf_ft_selected, "mh_discussion_negative")],
               method = "ranger",
               trControl = fitControl_mod,
               na.action=na.exclude,
               verbose = FALSE,
               weights = weights,
               importance = "impurity",
               tuneGrid = rfGrid)


# explainer after bias mitigation by reweighing
rf_explainer_w <- DALEX::explain(rfMod_w, 
                                 data = train_copy, 
                                 y = y_numeric, 
                                 label = "rf_weighted",
                                 colorize = FALSE)

rf_fobject <- fairmodels::fairness_check(rf_fobject, rf_explainer_w, verbose = FALSE)

plot(rf_fobject)

#### 2. Disparate Impact Remover (Not appropriate in our case, because all columns
# are categorical indices, not numeric values)



#### 3. Resampling

# There are two ways of resampling: uniform and preferential.
uniform_indexes      <- resample(protected = train_copy$genderMale,
                                 y = y_numeric)


preferential_indexes <- resample(protected = train_copy$genderMale,
                                 y = y_numeric,
                                 type = "preferential",
                                 probs = rf_explainer$y_hat)

set.seed(1)
rfMod_u <- train(mh_discussion_negative ~ .,
                  data = trainSmote[uniform_indexes, 
                                    c(selected, "mh_discussion_negative")],
                  method = "ranger",
                  trControl = fitControl_mod,
                  na.action=na.exclude,
                  verbose = FALSE,
                  importance = "impurity",
                  tuneGrid = rfGrid)


# Build explainer after bias mitigation by reweighing
rf_explainer_u <- DALEX::explain(rfMod_u, 
                                  data = train_copy, 
                                  y = y_numeric, 
                                  label = "rf_uniform",
                                  colorize = FALSE)

set.seed(1)
rfMod_p <- train(mh_discussion_negative ~ .,
                  data = trainSmote[preferential_indexes, 
                                    c(selected, "mh_discussion_negative")],
                  method = "ranger",
                  trControl = fitControl_mod,
                  na.action=na.exclude,
                  verbose = FALSE,
                  importance = "impurity",
                  tuneGrid = rfGrid)


# Build explainer after bias mitigation by reweighing
rf_explainer_p <- DALEX::explain(rfMod_p, 
                                  data = train_copy, 
                                  y = y_numeric, 
                                  label = "rf_preferential",
                                  colorize = FALSE)

rf_fobject <- fairness_check(rf_fobject, rf_explainer_u, rf_explainer_p, 
                              verbose = FALSE)
plot(rf_fobject)




#### Post-processing Bias Mitigation Techniques #####
#### 1. ROC pivot
set.seed(1)

rf_explainer_r <- roc_pivot(rf_explainer,
                             protected = train_copy$genderMale,
                             privileged = "1")


rf_fobject <- fairness_check(rf_fobject, rf_explainer_r, 
                              label = "rf_roc",  # label as vector for explainers
                              verbose = FALSE) 

plot(rf_fobject)

### Check Bias-Accuracy Trade-off
?performance_and_fairness
rf_paf <- performance_and_fairness(rf_fobject, fairness_metric = "ACC",
                                    performance_metric = "accuracy")
rf_paf
plot(rf_paf)
# The y-axis denotes the loss. The goal is to minimize the loss.

fc <- fairness_check(svm_fobject, rf_fobject, verbose = FALSE)
plot(fc)
paf <- performance_and_fairness(fc, fairness_metric = "ACC", performance_metric = "accuracy")
plot(paf)
















