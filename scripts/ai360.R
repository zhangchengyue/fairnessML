
setwd("D:/MScAC信息/Ji Lab/fairnessML")

# devtools::install_github("Trusted-AI/AIF360/aif360/aif360-r")
library(aif360)

# Setting up AI360 in R:
# https://github.com/Trusted-AI/AIF360/blob/main/aif360/aif360-r/README.md

# AIF360 is distributed as a Python package and so needs to be installed 
# within a Python environment on your system.

### Check Python
# reticulate::install_miniconda() # Install conda if not yet installed
# reticulate::conda_create("r-reticulate", python = "3.7") # Create conda environment
### Note that AIF360 use old versions of TensorFlow, which are only compatible with
### Python version 3.7 or older.

reticulate::conda_list()
reticulate::conda_python()
reticulate::py_config()

# reticulate::conda_remove("r-reticulate")

# Install all AIF360 dependencies
# aif360::install_aif360(envname = "r-reticulate", conda_python_version = "3.7")

#Activate Conda Environment
reticulate::use_miniconda(condaenv = "r-reticulate", required = TRUE)


### Get started
library(aif360)
load_aif360_lib()

# load a toy dataset
data <- data.frame("feature1" = c(0,0,1,1,1,1,0,1,1,0),
                   "feature2" = c(0,1,0,1,1,0,0,0,0,1),
                   "label" = c(1,0,0,1,0,0,1,0,1,1))

# format the dataset
formatted_dataset <- aif360::binary_label_dataset(data_path = data,
                                                  favor_label = 0,
                                                  unfavor_label = 1,
                                                  unprivileged_protected_attribute = 0,
                                                  privileged_protected_attribute = 1,
                                                  target_column = "label",
                                                  protected_attribute = "feature1")
load_aif360_lib()
ad <- adult_dataset()

### Bias Mitigation Techniques:
# 1. Pre-processing bias mitigation
aif360::disparate_impact_remover()
?disparate_impact_remover

di <- disparate_impact_remover(repair_level = 1.0, sensitive_attribute = "feature1")
rp <- di$fit_transform(formatted_dataset)

di_2 <- disparate_impact_remover(repair_level = 0.8, sensitive_attribute = "feature1")
rp_2 <- di_2$fit_transform(formatted_dataset)

aif360::reweighing()
?reweighing

# 2. In-processing bias mitigation
aif360::adversarial_debiasing()
?adversarial_debiasing

aif360::prejudice_remover()

# 3. Post-processing bias mitigation
aif360::reject_option_classification()

# There are no bias detection metrics in their R package version. Only in Python version












#loading libraries
library(tidyverse)
library(caret)
library(ggplot2)
source("D:/MScAC信息/Ji Lab/fairnessML/mental-health-in-tech-project-main/code/script/functions/funs_do_feature_selection.R")
source("D:/MScAC信息/Ji Lab/fairnessML/mental-health-in-tech-project-main/code/script/functions/funs_do_caret_modeling.R")

#loading data
clean_df <- read.csv("D:/MScAC信息/Ji Lab/fairnessML/mental-health-in-tech-project-main/data/processed_data/mental-health-in-tech-2016-modeling.csv")

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
# svmGrid <-  expand.grid(C = c(.001, .01, .1, 0.5, 1.0))


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

# format the dataset to create AIF compatible dataset
# Our target column is: 
## mh_discussion_negative = Do.you.think.that.discussing.a.mental.health.
## disorder.with.your.employer.would.have.negative.consequences.

# Thus, the unfavored label is "Yes", because we don't want negative consequences.
train_copy <- trainSmote[, c(ft_selected$coefnames, "mh_discussion_negative")]
train_copy$mh_discussion_negative <- as.character(train_copy$mh_discussion_negative)
# 把No变成0，Yes变成1
train_copy$mh_discussion_negative[train_copy$mh_discussion_negative == "No"] <- 0
train_copy$mh_discussion_negative[train_copy$mh_discussion_negative == "Yes"] <- 1
train_copy$mh_discussion_negative <- as.factor(train_copy$mh_discussion_negative)

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
#### Dispartate Impact Remover
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
# But I still implemented the coding syntax for future reference.

formatted_dataset <- aif360::binary_label_dataset(data_path = train_copy,
                                                  favor_label = 0,
                                                  unfavor_label = 1,
                                                  unprivileged_protected_attribute = 0,
                                                  privileged_protected_attribute = 1,
                                                  target_column = "mh_discussion_negative", # y labels
                                                  protected_attribute = "genderMale")


di_remover <- disparate_impact_remover(repair_level = 1.0, sensitive_attribute = "genderMale")

dir_data <- di_remover$fit_transform(formatted_dataset)
dir_data # This is the de-biased dataset which can directly be used for Random Forest classification.

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
