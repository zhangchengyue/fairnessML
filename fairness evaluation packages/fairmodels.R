
# Link to tutorial with compas dataset: 
# https://modeloriented.github.io/fairmodels/articles/Basic_tutorial.html


# install.packages("fairmodels")
library(fairmodels)

# We use the compas dataset
data("compas")
head(compas)


# For fairmodels package to work properly we want to flip factor levels in 
# target variable, so positive outcome (not being a recidivist) is being 
# predicted by models. It is only needed for one specific function but more 
# on it later.
compas$Two_yr_Recidivism <- as.factor(ifelse(compas$Two_yr_Recidivism == '1', 
                                             '0', '1'))
# Basic features
# We train a ranger (Wright and Ziegler (2017)) model 
# and create an explainer with DALEX
library(DALEX)
library(ranger) # The ranger package implements a random forest model

# train model
rf_compas <- ranger::ranger(Two_yr_Recidivism ~., data = compas, probability = TRUE)
?ranger::ranger
# numeric target values
y_numeric <- as.numeric(compas$Two_yr_Recidivism)-1

# explainer
rf_explainer <- DALEX::explain(rf_compas, data = compas[,-1], y = y_numeric, colorize = FALSE)

# fairness check
fobject <- fairness_check(rf_explainer,                         # explainer
                          protected = compas$Ethnicity,         # protected variable as factor
                          privileged = "Caucasian",             # level in protected variable, potentially more privileged
                          cutoff = 0.5,                         # cutoff - optional, default = 0.5
                          colorize = FALSE)                         

print(fobject, colorize = FALSE)
plot(fobject)


# plot density
plot_density(fobject)


# Metric scores plot
plot(metric_scores(fobject))


# fairness object - idea
# To really see what fairness_object is about, we need to make some more models and explainers.
library(gbm)

rf_compas_1 <- ranger(Two_yr_Recidivism ~Number_of_Priors+Age_Below_TwentyFive,
                      data = compas,
                      probability = TRUE)

lr_compas_1 <- glm(Two_yr_Recidivism~.,
                   data=compas,
                   family=binomial(link="logit"))

rf_compas_2 <- ranger(Two_yr_Recidivism ~., data = compas, probability = TRUE) 
rf_compas_3 <- ranger(Two_yr_Recidivism ~ Age_Above_FourtyFive+Misdemeanor,
                      data = compas,
                      probability = TRUE)

df <- compas
df$Two_yr_Recidivism <- as.numeric(compas$Two_yr_Recidivism)-1
gbm_compas_1<- gbm(Two_yr_Recidivism~., data = df) 

explainer_1 <- explain(rf_compas_1,  data = compas[,-1], y = y_numeric)
explainer_2 <- explain(lr_compas_1,  data = compas[,-1], y = y_numeric)
explainer_3 <- explain(rf_compas_2,  data = compas[,-1], y = y_numeric, label = "ranger_2")
explainer_4 <- explain(rf_compas_3,  data = compas[,-1], y = y_numeric, label = "ranger_3")
explainer_5 <- explain(gbm_compas_1, data = compas[,-1], y = y_numeric)


# we create one object with all explainers
fobject <- fairness_check(explainer_1, explainer_2,
                          explainer_3, explainer_4,
                          explainer_5,
                          protected = compas$Ethnicity,
                          privileged = "Caucasian",
                          verbose = FALSE) 



# What consists of fairness object?
fobject$parity_loss_metric_data

# for the first model
fobject$groups_data$ranger$TPR

# for first model
fobject$cutoff$ranger


# We now have a few models in our fairness_object(fobject)
# Let’s see how they perform in different metrics.

# Stacked Barplot
sm <- stack_metrics(fobject)
plot(sm)


# Plot metric
cm <- choose_metric(fobject, "TPR")
plot(cm)


#Plot fairness PCA
fair_pca <- fairness_pca(fobject)
print(fair_pca)
plot(fair_pca)

# Plot Heatmap
fheatmap <- fairness_heatmap(fobject)
plot(fheatmap, text_size = 3)

# Metric and Performance Plot
fap <- performance_and_fairness(fobject, fairness_metric = "STP")
plot(fap)


# Group Metric
fobject2 <- fairness_check(explainer_1,explainer_2, 
                           protected = compas$Ethnicity,
                           privileged = "Caucasian", 
                           verbose = FALSE)


gm <- group_metric(fobject2, fairness_metric = "FPR")
plot(gm)


# Radar plot
fradar <- fairness_radar(fobject2)
plot(fradar)


# We may see how cutoff affects parity loss of metrics:
# All cutoffs
ac <- all_cutoffs(fobject2)

plot(ac)

# Ceteris paribus cutoff
cpc <- ceteris_paribus_cutoff(fobject2, subgroup = "African_American")

plot(cpc)


