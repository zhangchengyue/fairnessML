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

#train/test split-------------------------------------------------------------------------------------
trainIndex <- caret::createDataPartition(clean_df$mh_discussion_negative, 
                                         p = .75, 
                                         list = FALSE, 
                                         times = 1)
trainData <- clean_df[ trainIndex,]
testData <- clean_df[-trainIndex,]


#class imbalance-------------------------------------------------------------------------------------
PlotCatDist(trainData, trainData$mh_discussion_negative)

library(performanceEstimation)

trainSmote <- performanceEstimation::smote(mh_discussion_negative ~ ., data  = trainData)                         
table(trainSmote$mh_discussion_negative)



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
