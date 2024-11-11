# install.packages('fairness')
library(fairness)
vignette('fairness')

# Basic Tutorial
data('compas')
# The data already contains all variables necessary to run all parity metrics. 
# In case you set up your own predictive model, you will need to concatenate 
# predicted probabilities or predictions (0/1) to your original dataset or 
# supply them as a vector to the corresponding metric function.
head(compas)

# There are a total of 9 columns in this dataset
ncol(compas)

# create a binary numeric version of the outcome variable that we will 
# supply as outcome in fairness metrics functions
compas$Two_yr_Recidivism_01 <- ifelse(compas$Two_yr_Recidivism == 'yes', 1, 0)


# Below are all the fairness metrics implemented in this package:

# Demographic parity
# This is one of the most popular fairness indicators in the literature.
# Formula: (TP + FP)
output <- equal_odds(data    = compas,
           outcome = 'Two_yr_Recidivism_01',
           probs   = 'probability',
           group   = 'ethnicity',
           cutoff  = 0.5,
           base    = 'Caucasian')

# Proportional parity (Statistical parity ratio)
# Formula: (TP + FP) / (TP + FP + TN + FN)
output <- prop_parity(data    = compas, 
            outcome = 'Two_yr_Recidivism_01',
            group   = 'ethnicity',
            probs   = 'probability', 
            cutoff  = 0.5, 
            base    = 'Caucasian')

# Equalized odds (Equal opportunity ratio)
# Formula: TP / (TP + FN)
output <- equal_odds(data    = compas, 
           outcome = 'Two_yr_Recidivism_01', 
           group   = 'ethnicity',
           probs   = 'probability', 
           cutoff  = 0.5, 
           base    = 'African_American')

# Predictive rate parity
# Formula: TP / (TP + FP)
output <- pred_rate_parity(data    = compas, 
                 outcome = 'Two_yr_Recidivism_01', 
                 group   = 'ethnicity',
                 probs   = 'probability', 
                 cutoff  = 0.5, 
                 base    = 'African_American')


# Accuracy parity(Accuracy equality ratio)
# Formula: (TP + TN) / (TP + FP + TN + FN)
output <- acc_parity(data    = compas, 
           outcome = 'Two_yr_Recidivism_01', 
           group   = 'ethnicity',
           probs   = 'probability', 
           preds   = NULL,
           cutoff  = 0.5, 
           base    = 'African_American')


# False negative rate parity
# Formula: FN / (TP + FN)
output <- fnr_parity(data    = compas, 
           outcome = 'Two_yr_Recidivism_01', 
           group   = 'ethnicity',
           probs   = 'probability', 
           cutoff  = 0.5, 
           base    = 'African_American')


# False positive rate parity
# Formula: FP / (TN + FP)
output <- fpr_parity(data    = compas, 
           outcome = 'Two_yr_Recidivism_01', 
           group   = 'ethnicity',
           probs   = 'probability', 
           cutoff  = 0.5, 
           base    = 'African_American')

# Negative predictive value parity
# Formula: TN / (TN + FN)
output <- npv_parity(data    = compas, 
           outcome = 'Two_yr_Recidivism_01', 
           group   = 'ethnicity',
           probs   = 'probability', 
           cutoff  = 0.5, 
           base    = 'African_American')

# Specificity parity
# Formula: TN / (TN + FP)
output <- spec_parity(data    = compas, 
            outcome = 'Two_yr_Recidivism_01', 
            group   = 'ethnicity',
            probs   = 'probability', 
            cutoff  = 0.5, 
            base    = 'African_American')


# ROC AUC comparison
output <- roc_parity(data    = compas, 
           outcome = 'Two_yr_Recidivism_01', 
           group   = 'ethnicity',
           probs   = 'probability', 
           base    = 'African_American')

# Matthews correlation coefficient comparison
# Formula: (TP×TN-FP×FN)/√((TP+FP)×(TP+FN)×(TN+FP)×(TN+FN))
output <- mcc_parity(data    = compas, 
           outcome = 'Two_yr_Recidivism_01', 
           group   = 'ethnicity',
           probs   = 'probability', 
           cutoff  = 0.5, 
           base    = 'African_American')


##### Output and visualizations
output$Metric

output$Metric_plot

output$Probability_plot

output$ROCAUC_plot

