setwd("D:/MScAC信息/Ji Lab/fairnessML/fairnessML/mental-health-in-tech-project-main")
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
# TODO：SelectFeatures和FitTrainingModel在funs_do_caret_modeling.R里,但是它们运行一直不成功
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
densityplot(svmMod, pch = "|")

# FitTestModel在funs_do_caret_modeling.R里
# 注意：需要是factor
testData$mh_discussion_negative <- as.factor(testData$mh_discussion_negative)
#fitting test data
svm_test_results <- FitTestModel(svmMod, testData)
svm_test_results


#linear svm classification--------------------------------------------------------------------------------
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


########## Fairness Check ########
library(fairmodels)
library(DALEX)

train_copy <- trainSmote
train_copy$mh_discussion_negative <- as.character(train_copy$mh_discussion_negative)
# 把No变成0，Yes变成1，这样就是一个二分类问题
train_copy$mh_discussion_negative[train_copy$mh_discussion_negative == "No"] <- 0
train_copy$mh_discussion_negative[train_copy$mh_discussion_negative == "Yes"] <- 1
train_copy$mh_discussion_negative <- as.factor(train_copy$mh_discussion_negative)

y_numeric <- as.numeric(train_copy$mh_discussion_negative) - 1
y_numeric

# explainer
rf_explainer <- DALEX::explain(rfMod, data = train_copy, 
                               y = y_numeric, 
                               colorize = FALSE)
# ?DALEX::explain
rf_explainer

# fairness check
# TODO: 查一下protected和privileged是什么东西，然后把对应的内容放进去
# https://modeloriented.github.io/fairmodels/articles/Basic_tutorial.html

train_copy$genderFemale <- as.factor(train_copy$genderFemale)
train_copy$genderMale <- as.factor(train_copy$genderMale)

# Interpretation: If bars reach red field on the left it means that there is 
# bias towards certain unprivileged subgroup. If they reach one on the right 
# it means bias towards privileged
fobject <- fairmodels::fairness_check(rf_explainer,            # explainer
                          protected = train_copy$genderMale,   # protected variable as factor
                          privileged = 1,                      # level in protected variable, potentially more privileged
                          cutoff = 0.5,                        # cutoff - optional, default = 0.5
                          colorize = FALSE)                         

print(fobject, colorize = FALSE)
plot(fobject)


