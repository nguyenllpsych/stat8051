#################################################
##          STAT 8051 Kaggle Project           ##
##           Linh Nguyen - Group 4             ##
##           Created: 10-Nov-2020              ##
##         Last updated: 10-Nov-2020           ##
#################################################

# META ====
# Libraries
library(tidyverse)
library(codebook)
library(apaTables)

# Data
data <- read.csv("InsNova_train.csv")
attach(data)

# CLEANING ====
# > Make sure variable types are correct ----
names <- c("id", "veh_body", "gender", "area")
data[,names] <- 
  lapply(data[,names], as.factor)

rm(names)

# > Variable lables ----
var_label(data$id) <- "Policy key"
var_label(data$veh_value) <- "Market value of the vehicle in $10,000's"
var_label(data$veh_body) <- "Type of vehicles"
var_label(data$veh_age) <- "Age of vehicles"
var_label(data$gender) <- "Gender of driver"
var_label(data$area) <- "Driving area of residence"
var_label(data$dr_age) <- "Driver's age category"
var_label(data$exposure) <- "The basic unit of risk underlying an insurance premium"
var_label(data$claim_ind) <- "Indicator of claim"
var_label(data$claim_count) <- "The number of claims"
var_label(data$claim_cost) <- "Claim amount"

# > Value labels ----
val_labels(data$veh_age) <- c("Youngest" = 1,
                              "Oldest" = 4)

val_labels(data$dr_age) <- c("Young" = 1,
                             "Old" = 6)

val_labels(data$claim_ind) <- c("No" = 0,
                                "Yes" = 1)

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