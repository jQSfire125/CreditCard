# 1. Overview and Introduction
# 1.1 Introduction
# 1.1.1 Acknowledgments
# 1.2 Overview
# 1.3 Library imports
# Make sure the user has the required packages

#####################################################################
#### Note: if you have to install glmnet, choose the binary over ####
#### the compiled version. The compiled version did not work for ####
#### me. Type "no" when asked if you want the complied version  ####
#####################################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot)) install.packages("ggcorrplot", repos = "http://cran.us.r-project.org")
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(bestNormalize)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(Matrix)) install.packages("Matrix", repos = "http://cran.us.r-project.org")
if(!require(MLmetrics)) install.packages("MLmetrics", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
if(!require(fastAdaboost)) install.packages("fastAdaboost", repos = "http://cran.us.r-project.org")

# Library imports
library(tidyverse)
library(scales)
library(ggcorrplot)
library(plyr)
library(dplyr)
library(reshape2)
library(bestNormalize)
library(caret)
library(glmnet)
library(Matrix)
library(MLmetrics)
library(e1071)
library(ranger)
library(kernlab)
library(fastAdaboost)

# 1.4 Read in the compressed file 
# Note: this process takes 1-2 minutes
df <- read_csv("creditcard.csv.zip")

# Review the column classes as parsed by readr
lapply(df, class)

# readr assigned the numeric class to all our columns
# Let's change the target variable "Class" to integer
df$Class <- as.integer(df$Class)

# 2. Analysis 
# 2.1 Initial Exploratory Analysis
# Look at the file
head(df)
glimpse(df)

# Let's look at the last rows of our data
tail(df)

# There are over 280,000 observations, 30 features and 1 target variable
dim(df)

# Summary statistics
summary(df)

# Check for missing values
# It seems there are no missing values
colSums(is.na(df))

# Frauds are 0.173% of transactions (492 out of 284,807)
table(df$Class)
sum(df$Class) / length(df$Class)

# Let's look at the correlations between variables
# First calculate the correlations
corr <- round(cor(df), 2)

# Then we can plot the correlation matrix
# All PCA components have 0 correlation, as expected
# None of the correlations are big in absolute value
ggcorrplot(corr, title = "Correlation Matrix",
           type = "lower",
           lab = TRUE,
           lab_size = 2,
           digits = 1,
           tl.cex = 8,
           outline.color = "white", 
           ggtheme = ggplot2::theme_gray,
           show.legend = FALSE)

# Now that the correlations have been calculated, we can
# change Class into a factor (It will help in caret and with some graphs)
df$Class = as.factor(ifelse(df$Class == 1, "Yes", "No"))

# 2.2 Visual Analysis
# Histograms of PCA components
# Let's first look at the distribution of our PCA variables (V1-V28)
d <- melt(df[, 2:29])
ggplot(d, aes(x = value)) +
  geom_histogram(bins = 20) +
  facet_wrap(~variable, scales = "free_x") +
  labs(title = "Distributions of features V1 to V28",
       x = "Value",
       y = "Count")

# Histogram of Time Variable
# Data is over two days. One can see the night areas with less transactions
df %>% 
  ggplot(aes(x = Time)) +
  geom_histogram(bins = 100, fill = "gray", color = "black") +
  scale_x_continuous(label = comma, breaks = seq(0, 175000, 25000)) +
  scale_y_continuous(label = comma) +
  labs(title = "Transactions by Time",
         x = "Time (seconds)",
         y = "Number of Transactions")

# Histogram of Time between 4000 and 27000
# I will zoom into those two night ranges, Night 1
df %>% 
  filter(Time %in% (4000:27000)) %>%
  ggplot(aes(x = Time)) +
  geom_histogram(bins = 30, fill = "gray", color = "black") +
  scale_x_continuous(label = comma, breaks = seq(5000, 25000, 5000)) +
  labs(title = "Transactions, Night 1",
       x = "Time (seconds)",
       y = "Number of Transactions")

# Histogram of Time between 89000 and 112000
# Night 2
df %>% 
  filter(Time %in% (89000:112000)) %>%
  ggplot(aes(x = Time)) +
  geom_histogram(bins = 30, fill = "gray", color = "black") +
  scale_x_continuous(label = comma, breaks = seq(90000, 110000, 5000)) +
  labs(title = "Transactions, Night 2",
       x = "Time (seconds)",
       y = "Number of Transactions")

# Fraud by time of day
# Let's look at transaction by hour of day
# More fraud happens at night and early morning
df %>% 
  mutate(hour = (Time/3600) %% 24) %>%
  ggplot(aes(x = hour, fill = Class)) +
  geom_density(alpha = 0.4) +
  scale_x_continuous(limits = c(0, 24), breaks = seq(0, 24, 2)) +
  labs(title = "Fraud by Hour of Day",
       x = "Hour of Day",
       y = "Density",
       col = "Class") +
  scale_fill_discrete(labels = c("Not Fraud", "Fraud"))

# Histogram of Amount
# There seems to be a really long right tail in the distribution
df %>% 
  ggplot(aes(x = Amount)) +
  geom_histogram(bins = 100, fill = "gray", color = "black") +
  scale_x_continuous(label = comma) +
  scale_y_continuous(label = comma) +
  labs(title = "Transaction Amount",
       x = "Amount (Euros)",
       y = "Number of Transactions")

# Histogram of Amount with log scale on the x axis
# Let's make the x axis logarithmic to see the distribution better
df %>% 
  ggplot(aes(x = Amount)) +
  geom_histogram(bins = 100, fill = "gray", color = "black") +
  scale_x_log10() +
  scale_y_continuous(label = comma) +
  labs(title = "Transaction Amount",
       x = "Amount (Euros), Log Scale",
       y = "Number of Transactions")

# Warning in last step indicates a number of zeroes in the data
# We already know the minimum of Amount is zero
# How many transactions have value 0
sum(df$Amount == 0)

# 2.3 Data Cleaning and Feature Engineering
# 2.3.1 Data Cleaning
# Look for duplicates
sum(duplicated(df))

# Remove duplicated rows
df <- df[!duplicated(df), ]

# Transform Amount, (create Amount_Log)
# First we fit the data with the yeojohnson function
# We need to use Yeo Johnson since there are some zeroes in the data
yj_obj <- yeojohnson(df$Amount, standardize = TRUE)

# Then we transform the values using our Yeo Johnson object and save them
df$Amount_Log <- predict(yj_obj)

# We can now visualize our transformation
# It looks just like the graph were we used a log scale on the x axis
df %>% 
  ggplot(aes(x = Amount_Log)) +
  geom_histogram(bins = 100, fill = "gray", color = "black") +
  scale_x_continuous(label = comma) +
  scale_y_continuous(label = comma) +
  labs(title = "Transaction Amount (Yeo Johnson Transformation)",
       x = "Amount_Log (Euros)",
       y = "Number of Transactions")

# 2.3.2 Feature Engineering
# Engineer Night Feature
# The Time feature shows the seconds since the first transaction
# By itself it gives little predictive information, in my opinion
# Let's create a feature that has value 1 for "nights" and 0 otherwise
Night <- (df$Time %in% 4000:27000) | (df$Time %in% 89000:112000) 
df$Night <- as.integer(Night)

# How many transactions happened at "night"?
sum(df$Night)

# Drop Time and Amount_Log
# We don't need the Time feature any more
# I will also drop Amount_Log because I will make the transformation after
# the train/test split (to prevent data leakage)
df <- subset(df, select = -c(Time, Amount_Log))

# Verify everything looks right:
head(df)

# 2.4 Modeling Approach
# Create a validation set (final hold-out test set)
# Validation set will be 20% of the dataset
# We have a good amount of data, but because of the huge imbalance, I believe
# we need to keep a larger validation set to include enough fraud transactions
set.seed(775, sample.kind="Rounding")
test_index <- createDataPartition(y = df$Class, 
                                  times = 1, p = 0.20, 
                                  list = FALSE)
cc_train <- df[-test_index,]
cc_test <- df[test_index,]

# Let's make sure the number of fraudulent transactions is similar
sum(cc_train$Class == "Yes") / length(cc_train$Class)
sum(cc_test$Class == "Yes") / length(cc_test$Class)

# Transform Amount
# We will transform train and test data using the fit on the train data
# First we the fit Yeo Johnson model using the train data
# We use standardize = FALSE because we will scale at the model level
yj_obj <- yeojohnson(cc_train$Amount, standardize = FALSE)

# Then we transform our Amount variable on the train and test sets
cc_train$Amount <- predict(yj_obj)
cc_test$Amount <- predict(yj_obj, newdata = cc_test$Amount)

# As a sanity check, let's plot the transformed data 
cc_train %>% 
  ggplot(aes(x = Amount)) +
  geom_histogram(bins = 100, fill = "gray", color = "black") +
  scale_x_continuous(label = comma) +
  scale_y_continuous(label = comma) +
  labs(title = "Transaction Amount (Yeo Johnson), Train Set",
       x = "Amount_Log (Euros)",
       y = "Number of Transactions")

cc_test %>% 
  ggplot(aes(x = Amount)) +
  geom_histogram(bins = 100, fill = "gray", color = "black") +
  scale_x_continuous(label = comma) +
  scale_y_continuous(label = comma) +
  labs(title = "Transaction Amount (Yeo Johnson), Test Set",
       x = "Amount_Log (Euros)",
       y = "Number of Transactions")

# We are going to use a shared trainContol object to better compare 
# between models. First, let's create the K folds:
set.seed(111, sample.kind="Rounding")
myFolds <- createFolds(cc_train$Class, k = 5)

# Then we use those indices in our trainControl object
# using index will make sure all models use the exact same folds for 
# cross-validation
# We are using prSummary to use Area Under the Precision Recall Curve 
# to select the best hyper parameters
myControl <- trainControl(
  summaryFunction = prSummary,
  classProbs = TRUE,
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)

# 2.4.1 Regularized Logistic Regression with glmnet
# This model has two tuning parameters: alpha and lambda
# alpha = 0 means Ridge regularization and alpha = 1 means lasso
# lambda is the regularization strength
# Let's create a Tuning grid for glmnet
myGrid_glmnet <- expand.grid(
  alpha = c(0, 0.5, 1),
  lambda = seq(0.0001, 0.1, length = 10)
)

# We can now train the model using the caret package
# We are adding a pre-processing step that eliminates features with zero
# variance, then centers and scales our features
# This takes a while
model_glmnet <- train(
  Class ~ .,
  cc_train,
  method = "glmnet",
  metric = "AUC",
  preProcess = c("zv", "center", "scale"),
  tuneGrid = myGrid_glmnet,
  trControl = myControl
)

# Visualize the main hyper parameters
# We can see that Ridge regularization works better in this model
plot(model_glmnet, main="Regularized Logistic Regression")

# Results
# We can see the tuning parameters of the best model, the AUC of the PR curve,
# the Precision, Recall and F1 scores
model_glmnet$bestTune
model_glmnet$results$AUC
model_glmnet$results$Precision
model_glmnet$results$Recall
model_glmnet$results$F

# 2.4.2 Support Vector Machines with svmLinear
# Now let's use a SVM model
# svmLinear has only one tuning parameter C or cost.
# Tuning grid for svmLinear
myGrid_svm <- data.frame(C = c(0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1))

# Train the model
# We are adding a pre-processing step that eliminates features with zero
# variance, then centers and scales our features
# This takes a while
model_svm <- train(
  Class ~ .,
  cc_train,
  method = "svmLinear",
  metric = "AUC",
  preProcess = c("zv", "center", "scale"),
  tuneGrid = myGrid_svm,
  trControl = myControl
)

# Visualize the main hyper parameters
# We can see that a small value of C is best for this model
plot(model_svm, main="Support Vector Machines")

# Results
# We can see the tuning parameters of the best model, the AUC of the PR curve,
# the Precision, Recall and F1 scores
model_svm$bestTune
model_svm$results$AUC
model_svm$results$Precision
model_svm$results$Recall
model_svm$results$F

# 2.4.3 Random Forest with ranger
# We are going to use the ranger package for our random forest model
# ranger has 3 tunable parameters: mtry, splitrule, and min.node.size
# mtry is the number of randomly selected predictors (features)
# splitrule is the method to choose splits in our decision tree
# and min.node.size is a rule that stops splitting nodes when the minimum
# number of observations in a node is reached
# Tuning grid for ranger
# We ran several trials and extratrees always won for this dataset
# Also performance decreased for larger numbers of mtry
myGrid_rf <- expand.grid(
  mtry = c(1, 2, 3, 4, 5),
  splitrule = "extratrees",
  min.node.size = c(1, 5, 10)
)

# Train the model
# We don't need to scale and center for this algorithm
# This takes a while
model_rf <- train(
  Class ~ .,
  cc_train,
  method = "ranger",
  metric = "AUC",
  preProcess = c("zv"),
  tuneGrid = myGrid_rf,
  trControl = myControl
)

# Visualize the main hyper parameters
# We can see how larger values of mtry have lower performance
plot(model_rf, main="Random Forest")

# Results
# We can see the tuning parameters of the best model, the AUC of the PR curve,
# the Precision, Recall and F1 scores
model_rf$bestTune
model_rf$results$AUC
model_rf$results$Precision
model_rf$results$Recall
model_rf$results$F

# 2.4.4 Adaptive Boosting with adaboost
# We can now train our adaboost algorithm
# adaboost has two tunable parameters
# nIter is the number of trees in the ensemble
# method is the method the package uses. Real adaboost worked
# best in my trials.
myGrid_ada <- expand.grid(
  nIter = c(50, 100, 150),
  method = "Real adaboost"
)

# Train the model
# We do not need to scale for this algorithm either
# This takes a while
model_ada <- train(
  Class ~ .,
  cc_train,
  method = "adaboost",
  metric = "AUC",
  preProcess = c("zv"),
  tuneGrid = myGrid_ada,
  trControl = myControl
)

# Visualize the main hyper parameters
plot(model_ada, main="Adaptive Boosting")

# Results
# We can see the tuning parameters of the best model, the AUC of the PR curve,
# the Precision, Recall and F1 scores
model_ada$bestTune
model_ada$results$AUC
model_ada$results$Precision
model_ada$results$Recall
model_ada$results$F

# 2.4.5 Algorithm Selection
# We can now compare our models to choose the best performing 
# algorithm. First, we create a list of our models:
model_list <- list(
  glmnet = model_glmnet,
  svm = model_svm,
  rf = model_rf,
  ada = model_ada
)

# Then we call the function resamples on our models
resamps <- resamples(model_list)

# And we can now compare the results
summary(resamps)

# Plot the Area Under the Precision Recall Curve (AUPRC)
dotplot(resamps, metric = "AUC", main = "AUPRC by model")

# Plot the Precision score
dotplot(resamps, metric = "Precision", main = "Precision by model")

# Plot the Recall (Sensitivity) score
dotplot(resamps, metric = "Recall", main = "Recall by model")

# Visualize the F1 score
dotplot(resamps, metric = "F", main = "F1 score by model")

# 3. Results 
# Final model
# Now that we have chosen our model, we will use it on our test data
final_preds <- predict(model_ada, cc_test)

# With these predictions, we can generate the Confusion Matrix:
final_cm <- confusionMatrix(final_preds, 
                            as.factor(cc_test$Class), 
                            positive = "Yes")

# And finally we calculate the Kappa coefficient, Precision, Recall and F1 score of our final model
# For this model where 99.82% of observations are not fraud, accuracy is not a good metric
final_cm$overall["Kappa"]
final_cm$byClass["Precision"]
final_cm$byClass["Recall"]
final_cm$byClass["F1"]

## 4. Conclusion