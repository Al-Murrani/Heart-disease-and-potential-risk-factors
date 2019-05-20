library(readr)
library(dplyr)
library(ggplot2)
library(broom) # messy output of built-in functions into tidy df
library(Metrics)

# Read dataset
cleveland_colnames <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                        "thalach", "exang", "oldpeak", "slope", "ca",
                        "thal", "class")
cleveland <- read_csv("processed.cleveland.data", col_names = cleveland_colnames)

# Exploring raw data
glimpse(cleveland)
summary(cleveland)

# To visualize chol values to check for outliers
hist(cleveland$chol)
cleveland %>% filter(chol > 300)

# Tidying data and prepare data for analysis
# Converting sex into factor data type
cleveland %>% mutate(sex = factor(sex, levels = 0:1, labels = c("Female", "Male")))-> cleveland

# Outcome variable has more than two levels, convert to binary as hd
cleveland %>% mutate(hd = ifelse(class > 0, 1, 0))-> cleveland

# Explore the association graphically
# for continuous variables use box plot
cleveland %>% mutate(hd_labelled = ifelse(hd==1, "Disease", "Non Disease")) -> cleveland
ggplot(data = cleveland, aes(x = hd_labelled,y = age)) + geom_boxplot()
ggplot(data = cleveland, aes(x=hd_labelled, y=thalach))  + geom_boxplot() 

# for binary variable use bar graph
ggplot(data = cleveland,aes(x=hd_labelled, fill=sex)) + geom_bar(position="fill") + ylab("sex %") 

# calculate pairwise correlations between age and thalach, quantitative variables
cor.test(cleveland$age, cleveland$thalach)

# split data into training and testing
train = sample_frac(cleveland, 0.8, replace = FALSE)
test = sample_frac(cleveland, 0.2, replace = FALSE)

# multiple logistic regression model built using train data
model <- glm(data = train, hd ~ age + sex + thalach, family = 'binomial')

# extract the model summary
# coefficient table represents the log(Odds Ratio) of the outcome
# Residual deviance is a measure of the lack of fit of your model taken as a 
# whole, whereas the Null deviance is such a measure for a reduced model that 
# only includes the intercept. The bigger the difference between the null 
# deviance and residual deviance is, the more helpful our input variables were 
# for predicting the output variable.
# AIC is an estimate of how well your model is describing the patterns in your data. 
# Used for comparing models trained on the same dataset. Model with the lower 
# AIC is doing a better job describing the variance in the data.
# Due to the small value of p-value for sex and thalach, there are an association between sex, thalach and HD.
summary(model)

# Odds Ratio (OR) to quantify how strongly the presence or absence of property 
# A is associated with the presence or absence of the outcome. When the OR is 
# greater than 1, we say A is positively associated with outcome B (increases 
# the Odds of having B). Otherwise, we say A is negatively associated with B 
# (decreases the Odds of having B).
# male patient is apprx four times more likely to have HD than female
# slight negative association between thalach and HD 
model_tidy <- tidy(model)
model_tidy$OR <- exp(model_tidy$estimate) 
model_tidy$lower_CI <- exp(model_tidy$estimate - 1.96 * model_tidy$std.error)
model_tidy$upper_CI <- exp(model_tidy$estimate + 1.96 * model_tidy$std.error)

# test model, output probabilies P(Y=1|X) instead of logit
pred_prob <- predict.glm(model, test, type = "response")

# create a decision rule using probability 0.5 as cutoff and save the predicted decision into the main data frame
test$pred_hd <- ifelse(pred_prob >= 0.5, 1, 0)

#calculate auc, accuracy, clasification error
auc <- auc(test$hd, test$pred_hd)
accuracy <- accuracy(test$hd, test$pred_hd)
classification_error <- ce(test$hd, test$pred_hd)

# print out the metrics on to screen
print(paste("AUC=", auc))
print(paste("Accuracy=", accuracy))
print(paste("Classification Error=", classification_error))

# confusion matrix
table(test$hd,test$pred_hd, dnn=c('Predicted Status','True Status'))

