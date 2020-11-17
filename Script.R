#################################################
##          STAT 8051 Kaggle Project           ##
##           Linh Nguyen - Group 4             ##
##           Created: 10-Nov-2020              ##
##         Last updated: 16-Nov-2020           ##
#################################################

# META ====
# > Libraries ----
library(tidyverse)
library(codebook)
library(apaTables)
library(caret)
library(faraway)
library(ggplot2)
library(glmnet)
library(EnvStats) #for boxcox
library(cplm) #for tweedie 
set.seed(8051)

# > Data ----
data <- read.csv("InsNova_train.csv")
attach(data)

test <- read.csv("InsNova_test.csv")

# BASIC DESCRIPTIVES ====
summary(data)

# > Bivariate correlations ----
data %>% select(veh_value, exposure, veh_age, dr_age, claim_ind, claim_count, claim_cost) %>% 
  apa.cor.table(file = "bivariate corr.doc")

# > Individual regression for cost ----
## gender
modGen <- lm(claim_cost ~ gender)
summary(modGen)

## vehicle body
modBod <- lm(claim_cost ~ veh_body)
summary(modBod)

## vehical area 
modArea <- lm(claim_cost ~ area)
summary(modArea)

# MODEL FITTING ====
# > Split data for cross-validation ----
control <- trainControl(method = "cv", number = 10)
model <- train(claim_cost ~., data = data, method = "lm",
               trControl = control)

training <- data$claim_cost %>% 
  createDataPartition(p = 0.8, list = FALSE)
train <- data[training,]
test <- data[-training,]
rm(training)


# > Predict claim_ind ----
ind <- data %>% select(-claim_cost, -claim_count, -id)

ggplot(data, 
       aes(x = claim_ind)) + 
  geom_histogram(position = "dodge", binwidth = 1)

mInd <- glm(claim_ind ~., family = binomial, data = ind)
summary(mInd)

predInd <- predict(mInd, type = "response")
data <- cbind(data, predInd)

mInd <- train(as.factor(claim_ind) ~., data = ind, method = "glm",
                trControl = control)
print(mInd)
summary(mInd)

# > Predict claim_count ----
count <- data %>% select(-claim_cost, -claim_ind, -id)

countTrain <- train %>% select(-claim_cost, -claim_ind, -id)

## Poisson regression
mCount <- train(claim_count ~., method = "glm", family = "poisson", data = count, 
                trControl = control)
print(mCount)

test$claim_count <- predict(mCount, test)
summary(test$claim_count)

## Tweedie model -> good performance
mCount <- cpglm(claim_count ~., link = "log", data = countTrain)
summary(mCount)
predictions <- mCount %>% predict(test)
data.frame( R2 = R2(predictions, test$claim_count),
            RMSE = RMSE(predictions, test$claim_count),
            MAE = MAE(predictions, test$claim_count))

# > Predict claim_cost ----
cost <- data %>% select(-claim_ind, -id)

mCost <- glm(claim_cost ~., data = cost, offset = log(claim_count), family = "Gamma"(link = "log"))
