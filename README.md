# Investigating the Impact of Fairness Machine Learning in Mental Health Survey
Codes for fairness ML in mental health project.

## Roadmap
### 2024.Nov.10
#### Progress:
* Following [this github repo](https://github.com/emiburns/mental-health-in-tech-project) and performed random forest on OSMI2016 dataset with the predictor variable `Do you think that discussing a mental health disorder with your employer would have negative consequences`, renamed as the `mh_discussion_negative` column in the dataset.
  * There are three categories for this variable value: Yes, No, Maybe
  - [x] Data Cleaning
  - [x] Feature Selection
  - [x] Caret Modeling
* Tried the fairness evaluation using [fairmodels](https://cran.r-project.org/web/packages/fairmodels/index.html) R package. 
  - [x] Since it is designed only for binary classification problems, I randomly split half of the `Maybe` samples to `Yes`, and the other half to `No` for now.
  - [x] The `protected` attribute chosen is `gender`. The `privileged` group I chose is `1`, which indicates the `Male` subgroup. 
  - [x] Applied `fairmodels::fairness_check`.
* Generated several plots regarding fairness

#### TODOs:
- [ ] Interpret the plots generated by the `fairmodels` package. What do they mean?
- [ ] Apply bias mitigation techniques to the current model and then perform fairness evaluation again
- [ ] Search for multiclass classification R packages
  - [ ] Predict on the `mh_discussion_negative` variable as a multiclass classification problem.
  - [ ] Evaluate fairness
    - The fairness metrics are just calculating the ratio using TPR, FPR, TNR, FNR, so I think I can still use the `fairmodels` package for evaluation after applying multiclass classification models.
  - [ ] Bias mitigation
    - [ ] Are there bias mitigation techniques for multiclass classification models in R?


### 2024.Nov.5
#### Progress:
* Found R packages that provide thorough Fairness metrics for ML models.
* Read [a paper](https://www.nature.com/articles/s41598-024-58427-7) about how the fairness of ML models are being evaluated and mitigated on given datasets. (Get a general idea of the workflow)

#### TODOs:
- [x] Train several ML models on the [OSMI Mental Health in Tech Survey](https://can01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fosmihelp.org%2Fresearch.html&data=05%7C02%7Cshengyue.zhang%40mail.utoronto.ca%7C63ee11d3f89c4e59dcc408dcf2f586b5%7C78aac2262f034b4d9037b46d56c55210%7C0%7C0%7C638652383542602749%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=BwIVIM%2BeTmuQFw50H0Rh6WQ2f4GnEE9WYYMm5MdBJ3g%3D&reserved=0), and evaluate the fairness before and after applying bias mitigation techniques.
- [ ] Read more papers regarding how other people research about the fairness ML
  - [ ] Provide new bias mitigation technique
  - [ ] …
- [x] Follow [this GitHub tutorial](https://github.com/emiburns/mental-health-in-tech-project?tab=readme-ov-file) on [OSMI Mental Health in Tech Survey 2016 dataset](https://osmihelp.org/research.html) and seek corresponding R packages to perform ML models ([caret](https://cran.r-project.org/web/packages/caret/index.html)).

#### Ideas:
* I looked over the  OSMI Mental Health in Tech Survey dataset, and thought about the fairness regarding the COVID factor. Would the covid factor have any effect with respect to bias of ML prediction?


### 2024.Oct.25
#### TODOs:
- [x] Look at the open source dataset: [OSMI Mental Health in Tech Survey](https://can01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fosmihelp.org%2Fresearch.html&data=05%7C02%7Cshengyue.zhang%40mail.utoronto.ca%7C63ee11d3f89c4e59dcc408dcf2f586b5%7C78aac2262f034b4d9037b46d56c55210%7C0%7C0%7C638652383542602749%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=BwIVIM%2BeTmuQFw50H0Rh6WQ2f4GnEE9WYYMm5MdBJ3g%3D&reserved=0)
- [x] Find available bias mitigation techniques on ML (R package)
- [x] Find tools that evaluates fairness of a ML model




