#################################################
##          STAT 8051 Kaggle Project           ##
##           Linh Nguyen - Group 4             ##
##           Created: 10-Nov-2020              ##
##         Last updated: 29-Nov-2020           ##
#################################################

# META ====
# > Libraries ----
library(tidyverse)
library(codebook)
library(apaTables)
library(caret)
library(faraway)
library(ggplot2)
library(glmnet) #for lasso
library(EnvStats) #for boxcox
library(cplm) #for tweedie 
library(logistf) #for firth
library(flexmix) #for poisson mixture model
library(pROC) #for AUC
library(gbm) #for gradient boosting
library(TDboost) #for TD boost
set.seed(8051)
options(scipen = 999)

# > Data ----
data <- read.csv("InsNova_train.csv")
data$veh_body <- as.factor(data$veh_body)
data$gender <- as.factor(data$gender)
data$area <- as.factor(data$area)

submit <- read.csv("InsNova_test.csv")
submit <- submit %>% select(-id)

# > Split data for cross-validation ----
training <- data$claim_cost %>% 
  createDataPartition(p = 0.8, list = FALSE)
train <- data[training,]
test <- data[-training,]
rm(training)

attach(train)

# > Function for Gini index (Pitcher) ----
normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}

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

## area 
modArea <- lm(claim_cost ~ area)
summary(modArea)

## value
modVal <- lm(claim_cost ~ veh_value)
summary(modVal)

## exposure
modExp <- lm(claim_cost ~ exposure)
summary(modExp)

## vehicle age
modVage <- lm(claim_cost ~ veh_age)
summary(modVage)

## driver age
modDrage <- lm(claim_cost ~ dr_age)
summary(modDrage)

## interactions
##modInt <- lm(claim_cost ~ (veh_value+exposure+veh_body+veh_age+gender+area+dr_age)^2)
##drop1(modInt, test = "Chisq")
##
##modInt2 <- lm(claim_count ~ (veh_value+exposure+veh_body+veh_age+gender+area+dr_age)^2)
##drop1(modInt2, test = "Chisq")
##
##modInt3 <- glm(claim_ind ~ (veh_value+exposure+veh_body+veh_age+gender+area+dr_age)^2, 
##               family = "binomial")
##drop1(modInt3, test = "Chisq")

## count and indicator
modCount <- lm(claim_cost ~ claim_count + claim_ind)
summary(modCount)

# MODEL FITTING ====
# > Predict claim_ind ----
indTrain <- train %>% select(-claim_cost, -claim_count, -id)
indTest <- test %>% select(-claim_cost, -claim_count, -id)

# >> logistic regression ----
##mInd <- glm(claim_ind ~., family = binomial, data = indTrain)
##summary(mInd)
##
##predictions <- mInd %>% predict(indTest, type = "response")
##data.frame( R2 = R2(predictions, test$claim_ind),
##            RMSE = RMSE(predictions, test$claim_ind),
##            MAE = MAE(predictions, test$claim_ind))

# >> outliers ----
##cooksd <- cooks.distance(mInd)
##plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
##abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
##text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="red")  # add ##labels
##
##influential <- as.numeric(names(cooksd)[(cooksd > 4*mean(cooksd, na.rm=T))])
##head(train[influential, ])  # influential observations

# >> Firth logistic regression ----
mfInd <- logistf(claim_ind ~., data = indTrain)
summary(mfInd)

mIndCoef <- mfInd$coefficients[-1]

indpred <- function(newdata){
  newdata <- newdata %>% mutate(
  veh_body_coef = ifelse(veh_body == "BUS", 0,
                     ifelse(veh_body == "CONVT" , mIndCoef[[3]],
                        ifelse(veh_body == "COUPE" , mIndCoef[[4]],
                           ifelse(veh_body == "HBACK" , mIndCoef[[5]],
                              ifelse(veh_body == "HDTOP" , mIndCoef[[6]],
                                 ifelse(veh_body == "MCARA" , mIndCoef[[7]],
                                    ifelse(veh_body == "MIBUS" , mIndCoef[[8]],
                                       ifelse(veh_body == "PANVN" , mIndCoef[[9]],
                                          ifelse(veh_body == "RDSTR" , mIndCoef[[10]],
                                             ifelse(veh_body == "SEDAN" , mIndCoef[[11]],
                                                ifelse(veh_body == "STNWG" , mIndCoef[[12]],
                                                   ifelse(veh_body == "TRUCK" , mIndCoef[[13]],
                                                      ifelse(veh_body == "UTE" , mIndCoef[[14]], 
                                                         NA))))))))))))))
newdata <- newdata %>% mutate(
  area_coef = ifelse(area == "A", 0,
                     ifelse(area == "B", mIndCoef[[17]],
                            ifelse(area == "C", mIndCoef[[18]],
                                   ifelse(area == "D", mIndCoef[[19]],
                                          ifelse(area == "E", mIndCoef[[20]],
                                                 ifelse(area == "F", mIndCoef[[21]],
                                                        NA)))))))
newdata <- newdata %>% mutate(
  gender_coef = ifelse(gender == "F", 0, mIndCoef[[16]]))

newdata <- newdata %>% mutate(
  indPred = faraway::ilogit(mIndCoef[[1]] * veh_value + mIndCoef[[2]] * exposure + 
    veh_body_coef + veh_age * mIndCoef[[15]] + gender_coef + area_coef + dr_age * mIndCoef[[22]]))

newdata %>% select(-veh_body_coef, -area_coef, -gender_coef)
} #function to compute predicted values

indTrain <- indpred(indTrain)
data.frame( R2 = R2(indTrain$indPred, indTrain$claim_ind),
            RMSE = RMSE(indTrain$indPred, indTrain$claim_ind),
            NRMSE = RMSE(indTrain$indPred, indTrain$claim_ind)/(max(indTrain$claim_ind) - min(indTrain$claim_ind)),
            MAE = MAE(indTrain$indPred, indTrain$claim_ind))

indTest <- indpred(indTest)
data.frame( R2 = R2(indTest$indPred, indTest$claim_ind),
            RMSE = RMSE(indTest$indPred, indTest$claim_ind),
            NRMSE = RMSE(indTest$indPred, indTest$claim_ind)/(max(indTest$claim_ind) - min(indTest$claim_ind)),
            MAE = MAE(indTest$indPred, indTest$claim_ind))

# >> Firth with factor age ----
##indTrainF <- indTrain %>% 
##  mutate(dr_age = as.factor(dr_age),
##         veh_age = as.factor(veh_age))
##
##mfvaInd <- logistf(claim_ind ~., data = indTrainF)
##summary(mfvaInd)
##
##mfIndCoef <- mfvaInd$coefficients[-1]
##
##indpredF <- function(newdata){
##  newdata <- newdata %>% mutate(
##  veh_body_coef = ifelse(veh_body == "BUS", 0,
##                     ifelse(veh_body == "CONVT" , mfIndCoef[[3]],
##                        ifelse(veh_body == "COUPE" , mfIndCoef[[4]],
##                           ifelse(veh_body == "HBACK" , mfIndCoef[[5]],
##                              ifelse(veh_body == "HDTOP" , mfIndCoef[[6]],
##                                 ifelse(veh_body == "MCARA" , mfIndCoef[[7]],
##                                    ifelse(veh_body == "MIBUS" , mfIndCoef[[8]],
##                                       ifelse(veh_body == "PANVN" , mfIndCoef[[9]],
##                                          ifelse(veh_body == "RDSTR" , mfIndCoef[[10]],
##                                             ifelse(veh_body == "SEDAN" , mfIndCoef[[11]],
##                                                ifelse(veh_body == "STNWG" , mfIndCoef[[12]],
##                                                   ifelse(veh_body == "TRUCK" , mfIndCoef[[13]],
##                                                      ifelse(veh_body == "UTE" , mfIndCoef[[14]], 
##                                                         NA))))))))))))))
##newdata <- newdata %>% mutate(
##  area_coef = ifelse(area == "A", 0,
##                     ifelse(area == "B", mfIndCoef[[19]],
##                            ifelse(area == "C", mfIndCoef[[20]],
##                                   ifelse(area == "D", mfIndCoef[[21]],
##                                          ifelse(area == "E", mfIndCoef[[22]],
##                                                 ifelse(area == "F", mfIndCoef[[23]],
##                                                        NA)))))))
##newdata <- newdata %>% mutate(
##  gender_coef = ifelse(gender == "F", 0, mfIndCoef[[18]]))
##
##newdata <- newdata %>% mutate(
##  veh_age_coef = ifelse(veh_age == "1", 0, 
##                        ifelse(veh_age == "2", mfIndCoef[[15]],
##                               ifelse(veh_age == "3", mfIndCoef[[16]],
##                                      ifelse(veh_age == "4", mfIndCoef[[17]],
##                                             NA)))))
##
##newdata <- newdata %>% mutate(
##  dr_age_coef = ifelse(dr_age == "1", 0, 
##                        ifelse(dr_age == "2", mfIndCoef[[24]],
##                               ifelse(dr_age == "3", mfIndCoef[[25]],
##                                      ifelse(dr_age == "4", mfIndCoef[[26]],
##                                             ifelse(dr_age == "5", mfIndCoef[[27]],
##                                                    ifelse(dr_age == "6", mfIndCoef[[28]],
##                                                           NA)))))))
##
##newdata <- newdata %>% mutate(
##  indPred = faraway::ilogit(mfIndCoef[[1]] * veh_value + mfIndCoef[[2]] * exposure + 
##    veh_body_coef + veh_age_coef + gender_coef + area_coef + dr_age_coef))
##
##newdata %>% select(-veh_body_coef, -area_coef, -gender_coef, -veh_age_coef, -dr_age_coef)
##} #function to compute predicted value
##
##indTrainF <- indpredF(indTrainF)
##data.frame( R2 = R2(indTrainF$indPred, indTrainF$claim_ind),
##            RMSE = RMSE(indTrainF$indPred, indTrainF$claim_ind),
##            NRMSE = RMSE(indTrainF$indPred, indTrainF$claim_ind)/(max(indTrainF$claim_ind) - min(indTrainF$claim_ind)),
##            MAE = MAE(indTrainF$indPred, indTrainF$claim_ind))
##
##indTest <- indpredF(indTest)
##data.frame( R2 = R2(indTest$indPred, indTest$claim_ind),
##            RMSE = RMSE(indTest$indPred, indTest$claim_ind),
##            NRMSE = RMSE(indTest$indPred, indTest$claim_ind)/(max(indTest$claim_ind) - min(indTest$claim_ind)),
##            MAE = MAE(indTest$indPred, indTest$claim_ind))

# > Predict claim_count ----
countTrain <- train %>% select(-claim_cost, -id)
countTest <- test %>% select(-claim_cost, -id)

# >> Poisson regression ----
##mCount <- glm(claim_count ~., family = "poisson", data = countTrain)
##summary(mCount)
##
##predictions <- mCount %>% predict(test)
##data.frame( R2 = R2(predictions, test$claim_count),
##            RMSE = RMSE(predictions, test$claim_count),
##            MAE = MAE(predictions, test$claim_count))


# >> Tweedie model -> better performance ----
mCount <- cpglm(claim_count ~., link = "log", data = countTrain)
summary(mCount)

predictions <- mCount %>% predict(countTrain)
data.frame( R2 = R2(predictions, countTrain$claim_count),
            RMSE = RMSE(predictions, countTrain$claim_count),
            NRMSE = RMSE(predictions, countTrain$claim_count)/(max(countTrain$claim_count)-min(countTrain$claim_count)),
            MAE = MAE(predictions, countTrain$claim_count))

predictions <- mCount %>% predict(countTest)
data.frame( R2 = R2(predictions, countTest$claim_count),
            RMSE = RMSE(predictions, countTest$claim_count),
            NRMSE = RMSE(predictions, countTest$claim_count)/(max(countTest$claim_count)-min(countTest$claim_count)),
            MAE = MAE(predictions, countTest$claim_count))

# >> Tweedie without claim_ind ----
countTrain <- train %>% select(-claim_cost, -claim_ind, -id)
countTest <- test %>% select(-claim_cost, -claim_ind, -id)

miCount <- cpglm(claim_count ~., link = "log", data = countTrain)
summary(miCount)

predictions <- miCount %>% predict(countTest)
data.frame( R2 = R2(predictions, countTest$claim_count),
            RMSE = RMSE(predictions, countTest$claim_count),
            NRMSE = RMSE(predictions, countTest$claim_count)/(max(countTest$claim_count)-min(countTest$claim_count)),
            MAE = MAE(predictions, countTest$claim_count))

# >> Tweedie with TDBoost without ind ----
mtd1Count <- TDboost(claim_count ~ veh_value + exposure + veh_body + veh_age + gender
                    + area + dr_age, data = countTrain, cv.folds = 5,
                    distribution = list(name="EDM", alpha=1.7), n.trees = 1000,
                    shrinkage = 0.01, 
                    interaction.depth = 2)

bestCount1 <- TDboost.perf(mtd1Count , method="cv")
summary(mtd1Count, n.trees = bestCount1)

predictions <- predict(mtd1Count, countTrain, bestCount1)
data.frame( R2 = R2(predictions, countTrain$claim_count),
            RMSE = RMSE(predictions, countTrain$claim_count),
            NRMSE = RMSE(predictions, countTrain$claim_count)/(max(countTrain$claim_count)-min(countTrain$claim_count)),
            MAE = MAE(predictions, countTrain$claim_count))

normalizedGini(countTrain$claim_count, predictions)

predictions <- predict(mtd1Count, countTest, bestCount1)
data.frame( R2 = R2(predictions, countTest$claim_count),
            RMSE = RMSE(predictions, countTest$claim_count),
            NRMSE = RMSE(predictions, countTest$claim_count)/(max(countTest$claim_count)-min(countTest$claim_count)),
            MAE = MAE(predictions, countTest$claim_count))

normalizedGini(costTest$claim_count, predictions)

# >> Tweedie with TDBoost with ind ----
mtd2Count <- TDboost(claim_count ~ veh_value + exposure + veh_body + veh_age + gender
                    + area + dr_age + claim_ind, data = countTrain, cv.folds = 5,
                    distribution = list(name="EDM", alpha=1.7), n.trees = 1000,
                    shrinkage = 0.01, 
                    interaction.depth = 2)

bestCount2 <- TDboost.perf(mtd2Count , method="cv")
summary(mtd1Count, n.trees = bestCount2)

predictions <- predict(mtd2Count, countTrain, bestCount2)
data.frame( R2 = R2(predictions, countTrain$claim_count),
            RMSE = RMSE(predictions, countTrain$claim_count),
            NRMSE = RMSE(predictions, countTrain$claim_count)/(max(countTrain$claim_count)-min(countTrain$claim_count)),
            MAE = MAE(predictions, countTrain$claim_count))

normalizedGini(countTrain$claim_count, predictions)

predictions <- predict(mtd2Count, test, bestCount2)
data.frame( R2 = R2(predictions, countTest$claim_count),
            RMSE = RMSE(predictions, countTest$claim_count),
            NRMSE = RMSE(predictions, countTest$claim_count)/(max(countTest$claim_count)-min(countTest$claim_count)),
            MAE = MAE(predictions, countTest$claim_count))

normalizedGini(countTest$claim_count, predictions)

# >> Tweedie with factors ----
##countTrain <- train %>% select(-claim_cost, -claim_ind, -id)
##countTest <- test %>% select(-claim_cost, -claim_ind, -id)
##
##countTrainF <- countTrain %>% 
##  mutate(dr_age = as.factor(dr_age),
##         veh_age = as.factor(veh_age))
##countTestF <- countTest %>% 
##  mutate(dr_age = as.factor(dr_age),
##         veh_age = as.factor(veh_age))
##
##mifCount <- cpglm(claim_count ~., link = "log", data = countTrainF)
##summary(mifCount)
##
##predictions <- mifCount %>% predict(countTestF)
##data.frame( R2 = R2(predictions, countTestF$claim_count),
##            RMSE = RMSE(predictions, countTestF$claim_count),
##            NRMSE = RMSE(predictions, countTestF$claim_count)/(max(countTestF$claim_count)-min(countTestF$claim_count)),
##            MAE = MAE(predictions, countTestF$claim_count))

# >> Tweedie model with interactions -> did not improve fit ----
##mintCount <- cpglm(claim_count ~. 
##                   + exposure:veh_body + veh_body:gender,
##                   data = countTrain)
##summary(mintCount)
##
##predictions <- mintCount %>% predict(countTrain)
##data.frame( R2 = R2(predictions, countTrain$claim_count),
##            RMSE = RMSE(predictions, countTrain$claim_count),
##            NRMSE = RMSE(predictions, countTrain$claim_count)/(max(test$claim_count)-min(test$claim_count)),
##            MAE = MAE(predictions, countTrain$claim_count))
##
##
##predictions <- mintCount %>% predict(test)
##data.frame( R2 = R2(predictions, test$claim_count),
##            RMSE = RMSE(predictions, test$claim_count),
##            NRMSE = RMSE(predictions, test$claim_count)/(max(test$claim_count)-min(test$claim_count)),
##            MAE = MAE(predictions, test$claim_count))

# >> Poisson mixture model ----
##mzCount <- flexmix(claim_count ~ ., model = FLXMRglm(family = "poisson"), k = 2, data = countTrain)
##summary(mzCount)
##parameters(mzCount, component = 1)
##
##predictions <- mzCount %>% predict(countTrain)
##data.frame( R2 = R2(predictions[1]$Comp.1, countTrain$claim_count),
##            RMSE = RMSE(predictions[1]$Comp.1, countTrain$claim_count),
##            NRMSE = RMSE(predictions[1]$Comp.1, countTrain$claim_count)/(max(countTrain$claim_count)-min##(countTest$claim_count)),
##            MAE = MAE(predictions[1]$Comp.1, countTrain$claim_count))
##
##predictions <- mzCount %>% predict(countTest)
##data.frame( R2 = R2(predictions[1]$Comp.1, countTest$claim_count),
##            RMSE = RMSE(predictions[1]$Comp.1, countTest$claim_count),
##            NRMSE = RMSE(predictions[1]$Comp.1, countTest$claim_count)/(max(countTest$claim_count)-min##(countTest$claim_count)),
##            MAE = MAE(predictions[1]$Comp.1, countTest$claim_count))

# > Predict claim_cost ----
costTrain <- train %>% select(-id)
costTest <- test %>% select(-id)

# >> Tweedie model ----
mCost <- cpglm(claim_cost ~., link = "log", data = costTrain)
summary(mCost)

predictions <- mCost %>% predict(costTrain)
data.frame( R2 = R2(predictions, costTrain$claim_cost),
            RMSE = RMSE(predictions, costTrain$claim_cost),
            NRMSE = RMSE(predictions, costTrain$claim_cost)/(max(costTrain$claim_cost)-min(costTrain$claim_cost)),
            MAE = MAE(predictions, costTrain$claim_cost))

predictions <- mCost %>% predict(costTest)
data.frame( R2 = R2(predictions, costTest$claim_cost),
            RMSE = RMSE(predictions, costTest$claim_cost),
            NRMSE = RMSE(predictions, costTest$claim_cost)/(max(costTest$claim_cost)-min(costTest$claim_cost)),
            MAE = MAE(predictions, costTest$claim_cost))

# >> Tweedie with TDBoost without ind ----
mtd1Cost <- TDboost(claim_cost ~ veh_value + exposure + veh_body + veh_age + gender
                    + area + dr_age + claim_count, data = costTrain, cv.folds = 5,
                    distribution = list(name="EDM", alpha=1.7), n.trees = 1000,
                    shrinkage = 0.01, 
                    interaction.depth = 2)

best <- TDboost.perf(mtd1Cost , method="cv")
summary(mtd1Cost, n.trees = best)

predictions <- predict(mtd1Cost, costTrain, best)
data.frame( R2 = R2(predictions, costTrain$claim_cost),
            RMSE = RMSE(predictions, costTrain$claim_cost),
            NRMSE = RMSE(predictions, costTrain$claim_cost)/(max(costTrain$claim_cost)-min(costTrain$claim_cost)),
            MAE = MAE(predictions, costTrain$claim_cost))

normalizedGini(costTrain$claim_cost, predictions)

predictions <- predict(mtd1Cost, costTest, best)
data.frame( R2 = R2(predictions, costTest$claim_cost),
            RMSE = RMSE(predictions, costTest$claim_cost),
            NRMSE = RMSE(predictions, costTest$claim_cost)/(max(costTest$claim_cost)-min(costTest$claim_cost)),
            MAE = MAE(predictions, costTest$claim_cost))

normalizedGini(costTest$claim_cost, predictions)

# >> Tweedie with TDBoost with ind ----
mtd2Cost <- TDboost(claim_cost ~ veh_value + exposure + veh_body + veh_age + gender
                    + area + dr_age + claim_count + claim_ind, data = costTrain, cv.folds = 5,
                    distribution = list(name="EDM", alpha=1.7), n.trees = 1000,
                    shrinkage = 0.01, 
                    interaction.depth = 2)

best2 <- TDboost.perf(mtd2Cost , method="cv")
summary(mtd2Cost, n.trees = best2)

predictions <- predict(mtd2Cost, costTrain, best2)
data.frame( R2 = R2(predictions, costTrain$claim_cost),
            RMSE = RMSE(predictions, costTrain$claim_cost),
            NRMSE = RMSE(predictions, costTrain$claim_cost)/(max(costTrain$claim_cost)-min(costTrain$claim_cost)),
            MAE = MAE(predictions, costTrain$claim_cost))

normalizedGini(costTrain$claim_cost, predictions)

predictions <- predict(mtd2Cost, costTest, best2)
data.frame( R2 = R2(predictions, costTest$claim_cost),
            RMSE = RMSE(predictions, costTest$claim_cost),
            NRMSE = RMSE(predictions, costTest$claim_cost)/(max(costTest$claim_cost)-min(costTest$claim_cost)),
            MAE = MAE(predictions, costTest$claim_cost))

normalizedGini(costTest$claim_cost, predictions)

# >> Tweedie with TDBoost without ind or count ----
mtd3Cost <- TDboost(claim_cost ~ veh_value + exposure + veh_body + veh_age + gender
                    + area + dr_age, data = costTrain, cv.folds = 5,
                    distribution = list(name="EDM", alpha=1.7), n.trees = 1000,
                    shrinkage = 0.01, 
                    interaction.depth = 2)

best3 <- TDboost.perf(mtd3Cost , method="cv")
summary(mtd3Cost, n.trees = best3)

predictions <- predict(mtd3Cost, costTrain, best3)
data.frame( R2 = R2(predictions, costTrain$claim_cost),
            RMSE = RMSE(predictions, costTrain$claim_cost),
            NRMSE = RMSE(predictions, costTrain$claim_cost)/(max(costTrain$claim_cost)-min(costTrain$claim_cost)),
            MAE = MAE(predictions, costTrain$claim_cost))

normalizedGini(costTrain$claim_cost, predictions)

# >> Tweedie with factors ----
##costTrainF <- costTrain %>% 
##  mutate(dr_age = as.factor(dr_age),
##         veh_age = as.factor(veh_age))
##costTestF <- costTest %>% 
##  mutate(dr_age = as.factor(dr_age),
##         veh_age = as.factor(veh_age))
##
##mfCost <- cpglm(claim_cost ~., link = "log", data = costTrainF)
##summary(mfCost)
##
##predictions <- mfCost %>% predict(costTrainF)
##data.frame( R2 = R2(predictions, costTrainF$claim_cost),
##            RMSE = RMSE(predictions, costTrainF$claim_cost),
##            NRMSE = RMSE(predictions, costTrainF$claim_cost)/(max(costTrainF$claim_cost)-min(costTrainF$claim_cost)),
##            MAE = MAE(predictions, costTrainF$claim_cost))
##
##predictions <- mfCost %>% predict(costTestF)
##data.frame( R2 = R2(predictions, costTestF$claim_cost),
##            RMSE = RMSE(predictions, costTestF$claim_cost),
##            NRMSE = RMSE(predictions, costTestF$claim_cost)/(max(costTestF$claim_cost)-min(costTestF$claim_cost)),
##            MAE = MAE(predictions, costTestF$claim_cost))

# >> Test cost model with ind and count ----

# >>> firth for ind ----
costTest2 <- test %>% select(-claim_count, -claim_ind, -claim_cost, -id)

costTest2 <- indpred(costTest2)
names(costTest2)[names(costTest2) == "indPred"] <- "claim_ind"
costTest2$claim_count <- mCount %>% predict(costTest2)
costTest2$claim_cost <- mCost %>% predict(costTest2)

data.frame( R2 = R2(costTest2$claim_ind, costTest$claim_ind),
            RMSE = RMSE(costTest2$claim_ind, costTest$claim_ind),
            NRMSE = RMSE(costTest2$claim_ind, costTest$claim_ind)/(max(costTest$claim_ind)-min(costTest$claim_ind)),
            MAE = MAE(costTest2$claim_ind, costTest$claim_ind))

data.frame( R2 = R2(costTest2$claim_count, costTest$claim_count),
            RMSE = RMSE(costTest2$claim_count, costTest$claim_count),
            NRMSE = RMSE(costTest2$claim_count, costTest$claim_count)/(max(costTest$claim_count)-min(costTest$claim_count)),
            MAE = MAE(costTest2$claim_count, costTest$claim_count))

data.frame( R2 = R2(costTest2$claim_cost, costTest$claim_cost),
            RMSE = RMSE(costTest2$claim_cost, costTest$claim_cost),
            NRMSE = RMSE(costTest2$claim_cost, costTest$claim_cost)/(max(costTest$claim_cost)-min(costTest$claim_cost)),
            MAE = MAE(costTest2$claim_cost, costTest$claim_cost))

normalizedGini(costTest2$claim_cost, costTest$claim_cost)

# >>> logistics for ind ----
##costTest2 <- test %>% select(-claim_count, -claim_ind, -claim_cost, -id)
##
##costTest2$claim_ind <- mInd %>% predict(costTest2, type = "response")
##costTest2$claim_count <- mCount %>% predict(costTest2)
##costTest2$claim_cost <- mCost %>% predict(costTest2)
##
##data.frame( R2 = R2(costTest2$claim_ind, costTest$claim_ind),
##            RMSE = RMSE(costTest2$claim_ind, costTest$claim_ind),
##            NRMSE = RMSE(costTest2$claim_ind, costTest$claim_ind)/(max(costTest$claim_ind)-min(costTest$claim_ind)),
##            MAE = MAE(costTest2$claim_ind, costTest$claim_ind))
##
##data.frame( R2 = R2(costTest2$claim_count, costTest$claim_count),
##            RMSE = RMSE(costTest2$claim_count, costTest$claim_count),
##            NRMSE = RMSE(costTest2$claim_count, costTest$claim_count)/(max(costTest$claim_count)-min(costTest$claim_count)),
##            MAE = MAE(costTest2$claim_count, costTest$claim_count))
##
##data.frame( R2 = R2(costTest2$claim_cost, costTest$claim_cost),
##            RMSE = RMSE(costTest2$claim_cost, costTest$claim_cost),
##            NRMSE = RMSE(costTest2$claim_cost, costTest$claim_cost)/(max(costTest$claim_cost)-min(costTest$claim_cost)),
##            MAE = MAE(costTest2$claim_cost, costTest$claim_cost))

# >>> without ind for count ----
costTest2 <- test %>% select(-claim_count, -claim_ind, -claim_cost, -id)

costTest2 <- indpred(costTest2)
names(costTest2)[names(costTest2) == "indPred"] <- "claim_ind"

costTest2$claim_count <- mtd1Count %>% predict(costTest2, bestCount1)
costTest2$claim_cost <- mtd2Cost %>% predict(costTest2, best2)

data.frame( R2 = R2(costTest2$claim_count, costTest$claim_count),
            RMSE = RMSE(costTest2$claim_count, costTest$claim_count),
            NRMSE = RMSE(costTest2$claim_count, costTest$claim_count)/(max(costTest$claim_count)-min(costTest$claim_count)),
            MAE = MAE(costTest2$claim_count, costTest$claim_count))

data.frame( R2 = R2(costTest2$claim_cost, costTest$claim_cost),
            RMSE = RMSE(costTest2$claim_cost, costTest$claim_cost),
            NRMSE = RMSE(costTest2$claim_cost, costTest$claim_cost)/(max(costTest$claim_cost)-min(costTest$claim_cost)),
            MAE = MAE(costTest2$claim_cost, costTest$claim_cost))

normalizedGini(costTest2$claim_cost, costTest$claim_cost)

# >>> without ind for cost ----
costTest2 <- test %>% select(-claim_count, -claim_ind, -claim_cost, -id)

costTest2 <- indpred(costTest2)
names(costTest2)[names(costTest2) == "indPred"] <- "claim_ind"

costTest2$claim_count <- mtd2Count %>% predict(costTest2, bestCount2)
costTest2$claim_cost <- mtd1Cost %>% predict(costTest2, best)

data.frame( R2 = R2(costTest2$claim_count, costTest$claim_count),
            RMSE = RMSE(costTest2$claim_count, costTest$claim_count),
            NRMSE = RMSE(costTest2$claim_count, costTest$claim_count)/(max(costTest$claim_count)-min(costTest$claim_count)),
            MAE = MAE(costTest2$claim_count, costTest$claim_count))

data.frame( R2 = R2(costTest2$claim_cost, costTest$claim_cost),
            RMSE = RMSE(costTest2$claim_cost, costTest$claim_cost),
            NRMSE = RMSE(costTest2$claim_cost, costTest$claim_cost)/(max(costTest$claim_cost)-min(costTest$claim_cost)),
            MAE = MAE(costTest2$claim_cost, costTest$claim_cost))

normalizedGini(costTest2$claim_cost, costTest$claim_cost)

# >>> without ind at all ----
costTest2 <- test %>% select(-claim_count, -claim_ind, -claim_cost, -id)
costTest2$claim_count <- mtd1Count %>% predict(costTest2, bestCount1)
costTest2$claim_cost <- mtd1Cost %>% predict(costTest2, best)

normalizedGini(costTest2$claim_cost, costTest$claim_cost)

# >>> with ind for cost and count ----
costTest2 <- test %>% select(-claim_count, -claim_ind, -claim_cost, -id)

costTest2 <- indpred(costTest2)
names(costTest2)[names(costTest2) == "indPred"] <- "claim_ind"

costTest2$claim_count <- mtd2Count %>% predict(costTest2, bestCount2)
costTest2$claim_cost <- mtd2Cost %>% predict(costTest2, best2)

data.frame( R2 = R2(costTest2$claim_count, costTest$claim_count),
            RMSE = RMSE(costTest2$claim_count, costTest$claim_count),
            NRMSE = RMSE(costTest2$claim_count, costTest$claim_count)/(max(costTest$claim_count)-min(costTest$claim_count)),
            MAE = MAE(costTest2$claim_count, costTest$claim_count))

data.frame( R2 = R2(costTest2$claim_cost, costTest$claim_cost),
            RMSE = RMSE(costTest2$claim_cost, costTest$claim_cost),
            NRMSE = RMSE(costTest2$claim_cost, costTest$claim_cost)/(max(costTest$claim_cost)-min(costTest$claim_cost)),
            MAE = MAE(costTest2$claim_cost, costTest$claim_cost))

normalizedGini(costTest2$claim_cost, costTest$claim_cost)


# >>> with factors ----
##costTest2F <- test %>% select(-claim_count, -claim_ind, -claim_cost, -id) %>% 
##  mutate(dr_age = as.factor(dr_age),
##         veh_age = as.factor(veh_age))
##
##costTest2F <- indpredF(costTest2F)
##names(costTest2F)[names(costTest2F) == "indPred"] <- "claim_ind"
##costTest2F$claim_count <- mifCount %>% predict(costTest2F)
##costTest2F$claim_cost <- mfCost %>% predict(costTest2F)
##
##data.frame( R2 = R2(costTest2F$claim_ind, costTest$claim_ind),
##            RMSE = RMSE(costTest2F$claim_ind, costTest$claim_ind),
##            NRMSE = RMSE(costTest2F$claim_ind, costTest$claim_ind)/(max(costTest$claim_ind)-min(costTest$claim_ind)),
##            MAE = MAE(costTest2F$claim_ind, costTest$claim_ind))
##
##data.frame( R2 = R2(costTest2F$claim_count, costTest$claim_count),
##            RMSE = RMSE(costTest2F$claim_count, costTest$claim_count),
##            NRMSE = RMSE(costTest2F$claim_count, costTest$claim_count)/(max(costTest$claim_count)-min(costTest$claim_count)),
##            MAE = MAE(costTest2F$claim_count, costTest$claim_count))
##
##data.frame( R2 = R2(costTest2F$claim_cost, costTest$claim_cost),
##            RMSE = RMSE(costTest2F$claim_cost, costTest$claim_cost),
##            NRMSE = RMSE(costTest2F$claim_cost, costTest$claim_cost)/(max(costTest$claim_cost)-min(costTest$claim_cost)),
##            MAE = MAE(costTest2F$claim_cost, costTest$claim_cost))

# SUBMISSION ----
# > Predict count and ind ----

## pred ind with firth no cut off 
submit <- indpred(submit)
names(submit)[names(submit) == "indPred"] <- "claim_ind"

##submit$claim_count <- mCount %>% predict(submit)

## pred count without _ind 
##submit$claim_count <- miCount %>% predict(submit)

## pred everything with factors
##submitF <- submit %>% mutate(veh_age = as.factor(veh_age), dr_age = as.factor(dr_age))
##submitF <- indpredF(submitF)
##names(submitF)[names(submitF) == "indPred"] <- "claim_ind"
##submitF$claim_count <- mifCount %>% predict(submitF)

## pred count with tdboost without ind
submit$claim_count <- mtd1Count %>% predict(submit, bestCount1)

# > Predict cost ----
##submit$claim_cost <- mCost %>% predict(submit)
##submit <- submit %>% select(claim_cost)

## pred everything with factors
##submitF$claim_cost <- mfCost %>% predict(submitF)
##submit <- submitF %>% select(claim_cost)

## pred cost without count or ind
##submit$claim_cost <- mtd3Cost %>% predict(submit, best3)
##submit <- submit %>% select(claim_cost)

## pred cost with tdboost with count and ind
submit$claim_cost <- mtd2Cost %>% predict(submit, best2)
submit <- submit %>% select(claim_cost)

# export
write.csv(submit, "submit.csv")