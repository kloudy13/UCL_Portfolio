#STATISTICS G003 In- Course Assessment 2
# Question 1
# by Klaudia Ludwisiak 20.02.2017

##################################################################################
#  SETUP
data=read.table("vitd.dat .txt",header=TRUE)
head(data)

# set the names of the new variables
vitd=data[,1]
bmi=data[,2]
outside=data[,3]
supply= data[,4]
exercise= data[,5]

###################### DATA EXPLORATION #########################################
##  Exploratory data analysis

# summarise data and explore standard deviations
summary(data) 

sd(data$vitd)
sd(data$bmi)
sd(data$outside)
sd(data$exercise)

# plot histograms to visualise the above results
par(mfrow=c(2,2))
hist(vitd)
hist(bmi)
hist(outside)
hist(exercise)

# Observations:
# Vitamin D levels range between 0 and ~ 400ng/ml, they are concentrated below the mean and speared out above the mean.
# BMI ranges between ~19 (underweight) and ~33 (obese), most values concerted in the centre of the distribution but not at the mean/median.
# The 'Outside' variable has a large variability, with many extremely high and low values indicating some people spend virtually no time outdoors whereas others spend a lot (may be due to occupation?)
# The exercise variable is concentrated around the median, with relatively small quantile range, indicating most people exercise similar amounts of time.

# Make pairs plot to visualise data dependencies 
pairs(data )
# Observations:
# - Strong Linear relationship between vit.D levels and being outside (hardly surprising as vitamin D gets produced when sunlight meets the skin)
# - Based on bmi histogram and the pairs scatterplot evidence exits to try a log transformation in the BMI variable
# - Exercise seams to be uncorrelated and random, thus is would be this variable has less impact on vit.D levels
# - It should be noted that treatment may affect vit.D levels and a the potential difference in results should be should be examined more closely.

# Split data by categorical variable (treated vs not treated) to examine impact of treatment
noTreat <- data[which(data$suppl==0),] # person does not take vit.D
Treat <- data[which(data$suppl==1),] # person takes vit.D

################ DATA EXPLORATION #### split by categorical variable ###############
### Make Scatterplots with data split:
# vitd vs bmi:
par(mfrow=c(1,1))
plot(vitd,bmi,type="n", main="vitd vs bmi")
text(noTreat$vitd,noTreat$bmi,label=noTreat$suppl)
text(Treat$vitd,Treat$bmi,label=Treat$suppl,col=5)
legend(290,32.5,col=c(1,5),lty=c(1,1),legend=c("1: Treatment", "0: noTreatmnet"),bty="n")

# vitd vs outside:
par(mfrow=c(1,1))
plot(vitd,outside,type="n", main="vitd vs outside")
text(noTreat$vitd,noTreat$outside,label=noTreat$suppl)
text(Treat$vitd,Treat$outside,label=Treat$suppl,col=4)
legend(0,12,col=c(1,4),lty=c(1,1),legend=c("1: Treatment", "0: noTreatmnet"),bty="n")
# If one drew a line of best for both teh treated and untreated groups what 
# jumpes out is that these are roughly paralel and that untreated peaople tend to spend marginaly more time outdoors.

# vitd vs exercise:
par(mfrow=c(1,1))
plot(vitd,exercise,type="n", main="vitd vs exercise")
text(noTreat$vitd,noTreat$exercise,label=noTreat$suppl)
text(Treat$vitd,Treat$exercise,label=Treat$suppl,col=3)
legend(290,5,col=c(1,3),lty=c(1,1),legend=c("1: Treatment", "0: noTreatmnet"),bty="n")

### Make Boxplots with data split:
par(mfrow=c(2,2))
boxplot(vitd~supply,xlab="supply",ylab="Vit.D",main="Vit.D vs. Supply",ylim=c(0,400)) 
boxplot(bmi~supply,xlab="supply",ylab="BMI",main="BMI vs. Supply",ylim=c(18,35)) 
boxplot(outside~supply,xlab="supply",ylab=" outside",main=" Outside vs. Supply",ylim=c(0,14)) 
boxplot(exercise~supply,xlab="supply",ylab="exercise",main="Exercise vs. Supply",ylim=c(0,7)) 

### Observations:
# Unsurprisingly, those supplied with vit.D have in general higher levels of vit.D (higher mean and interquartile range)
# Possible anomaly in the group with no treatment, where one patient has exceptionally high levels of vit.D.
# BMI of treated group generally better than that of untreated (less variability and lower BMI values).
# From boxplot, on average, treated patients spend more time outdoors.
# Treated group exercises more than untreated group. 

#?? Rise questions: how was data collected? and target group for treatment choses? would they be the people who are health conscious? (ie. exercise more and are aware of health benefits of vit.D) or would it be people who were diagnosed with lack of vit.D and hence administered a supplement?. Looking at exercise, time spent outside and BMI it is reasonable to claim those treated are health conscious individuals, but this would have to be further investigated. 

##???# worth examining interaction with supply +boxplots for interaction
#also speak about number of patients with n without treatment 
# discuss extreme values both in case of no treatment (illness?)
# keep in mind there may be significant interaction (impact) between supply and bmi so can include in model. 

################ FITTING MODEL & Analysis ##################################################
# Problem definition & setup:
# The best model is that which achieves good vitamin.D predictions making use of the available data 
# and is as simple as possible.
# Initial exploration is undertaken in models 1-3 then backward elimination is followed as the 
# systematic process to select the best model. Starting from fitting an overreaching model including 
# all variables, summary will be used to decide which variables should be kept in the model and 
# which should be discarded (inspecting variable significance given by the P value from the F test). 
# Changes will be made to one element of the model at a time so as to keep track of their effect on results.

### Start by considering a simple additive model: 
model1 <- lm( vitd ~ bmi + outside + supply + exercise)
summary(model1)
# Observations: 1. variables exercise and bmi not statistically significant since p_value>5%
#               2. high R^2 (about 86.15%)
#               3. the constant (intercept) is not statistically significant

# Consider transformation of bmi to log(bmi)
model2 <- lm( vitd ~ log(bmi) + outside + supply + exercise)
summary(model2)
# Observations: 1. Still even after the transformation the variable bmi is not significant

# Consider including interaction term between log(bmi) and supply 
model3 <- lm( vitd ~ I(log(bmi))+supply+I(log(bmi)):supply + outside + exercise )
summary(model3)
# Observations: 1. based on EDA (explanatory data analysis) include to the additive model the 
#                  interaction between log(bmi) and supply
#               2. exercise, supply and the interaction between log(bmi) and supply not statistically significant
#               3. slightly improves R^2

# PERFORM BACKWARD ELIMINATION 
# 1. START FROM MODEL: VITD~b0+b1*log(bmi)+b2*exercise+b3*supply+b4*log(bmi)*supply+b5*outside+e (1)


# 2. Exclude the variable with the highest p-value => exercise
model4 <- lm( vitd ~ I(log(bmi))+supply+I(log(bmi)):supply + outside) #excl supply

# perform F-test to check if the variable exercise is statistically significant given the nested model (model 4)
anova(model4, model3) 
# Observations: F test indicates there is enough evidence to exclude
summary(model4)
# Observations: 1. supply and the interaction between log(bmi) not statistically significant
#               2. good R^2
#               3. intercept not significant
#               4. log(bmi) only marginally significant

#3. Remove again the variable with the highest p-value, i.e: consider removing supply variable but keep interaction
model5 <- lm( vitd ~ I(log(bmi)):supply+I(log(bmi)) + outside) 
anova(model5, model4) 
# Observations: F test indicates enough evidence to exclude supply variable.
summary(model5)
# Observations: 1. log(bmi) and the interaction between log(bmi) not statistically significant
#               2. good R^2
#               3. intercept not significant
#               4. remember in the previous model the variable log(bmi) was marginally significant

#4. Removing log(bmi) variable but keep interaction
model6<- lm( vitd ~ I(log(bmi)):supply + outside ) 
anova(model6, model5) 
# Observations: F test indicates enough evidence to exclude log(bmi).
summary(model6)
# Observations: 1. final model: VITD~b0+b1*outside+b2*log(bmi)*supply+e (2)
#               2. good R^2
#               3. intercept not significant- but we still keep it in the model, for interpertation reasons



#11% INDICATING THAT LOG(BMI) SO YOIU CANNOT REJECT Null HYP that beta2=0 (slope) is the slope of log(bmi) so we conclude that this is also not sig and the only sig vars is the interaction terms and the outside var
# the process followed is the backward elimination where we start from incl. all vars and eliminate the least sig ones looking at P value from an F test

# now we need to look at residuals -goodness of fit analysis

################## MODEL SELECTION & Refinement #####################

### Examine Residuals -- Goodness of Fit
# diagnostic plot - inspect QQ plot and cook distance:
par(mfrow=c(2,2))
plot(model6,which=1:4,add.smooth=FALSE,ask=FALSE)
# Observations: 
# QQ plot not good, visible outliers (points 5, 15, 22), which could be excluded.
# Consider point 22 to be excl. above the others because of the magnitude of the residuals difference.

# remove point 22:
newdata=data[-22,]

# re-fit selected model:
new_model6=lm( vitd ~ I(log(bmi)):suppl + outside, data=newdata )
summary(new_model6)
plot(new_model6,which=1:4,add.smooth=FALSE,ask=FALSE)
