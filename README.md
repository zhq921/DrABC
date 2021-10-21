# DrABC

This repository provides the code for the paper "DrABC: Deep Learning Accurately Predicting Germline Pathogenic Mutation Status in Breast Cancer Patients Based on Phenotype Data".

### BACKGROUND

- Identifying breast cancer patients with DNA repair pathway-related germline pathogenic variants (GPVs) is important for effectively employing systemic treatment strategies and risk-reducing interventions. However, current criteria and risk prediction models for prioritizing genetic testing among breast cancer patients do not meet the demands of clinical practice due to insufficient accuracy.

### What is DrABC

- A phenotype-based GPV risk prediction model named DNA-repair Associated Breast Cancer (DrABC) was developed based on hierarchical neural network architecture and achieved superior performance in identifying GPV carriers among Chinese breast cancer patients.

### Comparison with other machine learning algorithms

- To evaluate the performance between the DrABC model and other machine learning models, we explored six kinds of common machine learning algorithms, including a fixed grid of Generalized Linear Models (GLMs), a naive Bayes (NB) classifier, five pre-specified Gradient Boosting Machine (GBM) models, three pre-specified and a random grid of eXtreme Gradient Boosting (XGBoost) models, a default Random Forest (RF), a near-default Deep Neural Net (DNN) and a random grid of DNNs.

### Online tool for the DrABC model

- We implemented a [website](http://gifts.bio-data.cn/) interface (http://gifts.bio-data.cn/) to accommodate extensions to the DrABC model and make it easily accessible to healthcare providers and researchers.

### Who do I talk to?

- Jiaqi Liu (J.Liu@cicams.ac.cn)
