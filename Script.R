#################################################
##          STAT 8051 Kaggle Project           ##
##           Linh Nguyen - Group 4             ##
##           Created: 10-Nov-2020              ##
##         Last updated: 22-Nov-2020           ##
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
set.seed(8051)
options(scipen = 999)

# > Data ----
data <- read.csv("InsNova_train.csv")
data$veh_body <- as.factor(data$veh_body)
data$gender <- as.factor(data$gender)
data$area <- as.factor(data$area)
attach(data)

submit <- read.csv("InsNova_test.csv")

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

## count and indicator
modCount <- lm(claim_cost ~ claim_count + claim_ind)
summary(modCount)

# MODEL FITTING ====
# > Split data for cross-validation ----
control <- trainControl(method = "cv", number = 10)

training <- data$claim_cost %>% 
  createDataPartition(p = 0.8, list = FALSE)
train <- data[training,]
test <- data[-training,]
rm(training)


# > Predict claim_ind ----
ind <- data %>% select(-claim_cost, -claim_count, -id)

indTrain <- train %>% select(-claim_cost, -claim_count, -id)
indTest <- test %>% select(-claim_cost, -claim_count, -id)

# >> logistic regression ----
##mInd <- glm(claim_ind ~., family = binomial, data = indTrain)
##summary(mInd)
##
##predictions <- mInd %>% predict(test)
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

# >> Firth logistic regression -> better performance, but not great ----
mInd <- logistf(claim_ind ~., data = indTrain)
summary(mInd)

predict(mInd, newdata=test, type="response")


mIndCoef <- mInd$coefficients[-1]

indTest <- indTest %>% mutate(
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
indTest <- indTest %>% mutate(
  area_coef = ifelse(area == "A", 0,
                     ifelse(area == "B", mIndCoef[[17]],
                            ifelse(area == "C", mIndCoef[[18]],
                                   ifelse(area == "D", mIndCoef[[19]],
                                          ifelse(area == "E", mIndCoef[[20]],
                                                 ifelse(area == "F", mIndCoef[[21]],
                                                        NA)))))))

indTest <- indTest %>% mutate(
  gender_coef = ifelse(gender == "F", 0, mIndCoef[[16]]))

indTest <- indTest %>% mutate(
  indPred = faraway::ilogit(mIndCoef[[1]] * veh_value + mIndCoef[[2]] * exposure + 
    veh_body_coef + veh_age * mIndCoef[[15]] + gender_coef + area_coef + dr_age * mIndCoef[[22]]))

data.frame( R2 = R2(indTest$indPred, indTest$claim_ind),
            RMSE = RMSE(indTest$indPred, indTest$claim_ind),
            NRMSE = RMSE(indTest$indPred, indTest$claim_ind)/(max(indTest$claim_ind) - min(indTest$claim_ind)),
            MAE = MAE(indTest$indPred, indTest$claim_ind))

# > Predict claim_count ----
count <- data %>% select(-claim_cost, -id)

countTrain <- train %>% select(-claim_cost, -id)

# >> Poisson regression ----
##mCount <- train(claim_count ~., method = "glm", family = "poisson", data = count, 
##                trControl = control)
##print(mCount)
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
            NRMSE = RMSE(predictions, countTrain$claim_count)/(max(test$claim_count)-min(test$claim_count)),
            MAE = MAE(predictions, countTrain$claim_count))


predictions <- mCount %>% predict(test)
data.frame( R2 = R2(predictions, test$claim_count),
            RMSE = RMSE(predictions, test$claim_count),
            NRMSE = RMSE(predictions, test$claim_count)/(max(test$claim_count)-min(test$claim_count)),
            MAE = MAE(predictions, test$claim_count))

# > Predict claim_cost ----

# >> Tweedie model ----
mCost <- cpglm(claim_cost ~., link = "log", data = train)
summary(mCost)

predictions <- mCost %>% predict(train)
data.frame( R2 = R2(predictions, train$claim_cost),
            RMSE = RMSE(predictions, train$claim_cost),
            NRMSE = RMSE(predictions, train$claim_cost)/(max(test$claim_cost)-min(test$claim_cost)),
            MAE = MAE(predictions, train$claim_cost))

predictions <- mCost %>% predict(test)
data.frame( R2 = R2(predictions, test$claim_cost),
            RMSE = RMSE(predictions, test$claim_cost),
            NRMSE = RMSE(predictions, test$claim_cost)/(max(test$claim_cost)-min(test$claim_cost)),
            MAE = MAE(predictions, test$claim_cost))

# SUBMISSION ----
# > Predict count and ind ----
submit <- submit %>% mutate(
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
submit <- submit %>% mutate(
  area_coef = ifelse(area == "A", 0,
                     ifelse(area == "B", mIndCoef[[17]],
                            ifelse(area == "C", mIndCoef[[18]],
                                   ifelse(area == "D", mIndCoef[[19]],
                                          ifelse(area == "E", mIndCoef[[20]],
                                                 ifelse(area == "F", mIndCoef[[21]],
                                                        NA)))))))

submit <- submit %>% mutate(
  gender_coef = ifelse(gender == "F", 0, mIndCoef[[16]]))

submit <- submit %>% mutate(
  claim_ind = faraway::ilogit(mIndCoef[[1]] * veh_value + mIndCoef[[2]] * exposure + 
    veh_body_coef + veh_age * mIndCoef[[15]] + gender_coef + area_coef + dr_age * mIndCoef[[22]]))

submit$claim_count <- mCount %>% predict(submit)

# > Predict cost ----
submit$claim_cost <- mCost %>% predict(submit)
submit <- submit %>% select(claim_cost)

# export
write.csv(submit, "submit.csv")
