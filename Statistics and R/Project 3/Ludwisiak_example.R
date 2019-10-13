# STATG003 ICA 3 GLM 
# Klaudia Ludwisiak 

# test data script demonstrating the working of my function 
# just press source and see results for two fitted models (details below)

### FUNCTION ###
# below the erm function is included so it can be called within this script with ease
erm<- function(y,X,startval = 0){
  
  ######################################################################  
  ## Dimension CHECKs
  # y should be a column vector with one value per data instance
  
  if (is.vector(y) == FALSE ){ stop("wrong y imput, should be a vector")}
  
  # if test passed specify size of y
  n = length(y)
  
  # X should be a matrix where number of rows (data instances) is the same as length(y)=n
  # and the number of columns is the number of explanatory variables (m) 
  if (is.vector(X)==TRUE){sizeX = c(n,1)}
  else {sizeX = dim(X) }
  
  if (sizeX[1] != n) { stop("wrong X input, matrix should have observations in its rows, these should be same as number or responses")}
  
  ## Data CHECK
  # is dataset suitable for modelling using the exponential distribution?
  # data has to be positive and non-zero  
  # 
  if (any(y <= 0)) {stop("response data not suitable to model with exponential distribution")}
  
  ######################################################################  
  ## SETUP -- initialise variables 
  
  m = sizeX[2]
  
  # assign uniform start values to vector of betas, this shoudl be a 1 by p vector
  betahat = rep(startval,dim(X)[2])
  
  # starter parameters for the IWLS below:
  U = 1  # Define a start value for the score, U
  iter = 0 # Define a start value for iteration count
  
  ######################################################################  
  ## Implement IWLS - adapted from IWLS code in lab 8
  
  while(any(abs(U) > 1e-7)) { # iterate until IWLS converges to desired accuracy (here 1e-7)
    
    # Calculate linear predictors, means and variances:
    
    eta <- X %*% betahat #as.vector(X %*% betahat) 
    # find estimate of distribution parameter lambda:
    lambda <- exp(eta)
    # using definition of the mean and variance function for a gamma distrib. with a=1:
    mu <- 1/lambda # mean i.e. expected value of y
    gdash <- -1/mu # 1st derivative of the link function wrt the mean 
    V <- 1/(lambda^2) # variance  (also = mu^2)
    
    # Calculate diagonal elements of W, where for the ith element: Wi = [g′ (μi)]^−2 /Vi.
    # it is observed that W is one, this makes the exponential distribution a special case,
    W <-(gdash^-2)/V # ((mu^-1)^-2)/V =1 if V= mu^2      
    
    # Calculate adjusted dependent variate
    z <- eta + ((y-mu)*gdash)   
    
    # Calcultion of the X'W elemnet t(W*X) can be replaced by t(x) making use of teh fact tah W=1. 
    XW <- t(X)     
    
    # Calculate U:
    U <- XW %*% (z-eta)
    
    # Estimate new beta:
    betahat <- solve(XW %*% X, XW %*% z)
    
    # Output current values to screen - disable in the submission version of function as not required
    #cat("Iteration",iter, " Estimate", round(betahat,6), " Score", round(U,8),"\n")   
    
    # count iterations
    iter <- iter + 1      
  }
  ###################################################################### 
  ## Other calculations 
  
  # Calculate variance - covariance matrix:
  # this is the inverse of the information matrix and is given by [X'WX]^-1.
  XWX = solve(XW %*% X) 
  covbeta = solve(XWX)  
  
  # Calculate standard errors
  SEbeta <- sqrt(diag(XWX)) 
  
  # Assuming the value of phi is known and equal to 1 we can test using the Z statistic as opposed to the T statistic:
  Zval = betahat/SEbeta
  Pval=2*pnorm(-abs(Zval)) #assuming two tailed test 
  
  # Assemble results into a table (data frame) and print
  (mytable <- data.frame(Beta_Estimate = round(betahat,4), S.E. = round(SEbeta,4), Z_stat = round(Zval,5), p_val= round(Pval,8) ))   
  cat("\nFinal results table:\n")
  row.names = seq(1,m,by=1) #need to change to p if doing intercept
  print(mytable, quote = TRUE, row.names = row.names)
  
  # Calculate fitted values:
  lambdahat <- as.vector(exp(X %*% betahat))#exp(as.vector(X %*% betahat)) #???
  fit = 1/lambdahat
  muhat <- fit
  
  # Calculate degrees of freedom:
  # The total variance has n-1 degrees of freedom.
  # reference: ref: http://stats.idre.ucla.edu/stata/output/regression-analysis-2/
  df = m-1 # model degrees of freedom correspond to the number of coefficients estimated minus 1.
  dfRes = n-1-df # residual degrees of freedom (total df - model df)
  
  # Calculate hat matrix
  hat_mat <- X%*%XWX%*%t(X)
  
  # Calculate deviance:
  dev_per_point <- 2*(((y-muhat)/muhat)-log(y/muhat))
  mydev = sum(dev_per_point)  # deviance 
  
  ## FURTHER CALC's for plotting:
  difference = y - fit # simple residual 
  mydev_resid = sign(difference)*sqrt(abs(dev_per_point))   # deviance residuals
  mydev_resid_std = mydev_resid/sqrt(1-diag(hat_mat))  # standardised deviance residuals
  
  # assemble results into a list object   
  list(y = y, fitted = fit, betahat = betahat, sebeta = SEbeta, cov.beta = covbeta, df.model = df, df.residual = dfRes, deviance= mydev)
  
  ######################################################################   
  ## MAKE PLOTS
  # use your knowledge of model checking for GLMs to produce an appropriate selection of diagnostics
  
  ## check distributional assumptions - is data exponentially distributed? 
  # for this incl. both simple plot and histogram, which allow visual inspection
  par(mfrow=c(1,2))
  plot(sort(y,decreasing = TRUE), xlab = 'Index', ylab = 'Observed response', main = 'Data Distribution check 1') 
  hist(y, xlab = 'Observations', ylab = 'Response variable', main = 'Data Distribution check 2', breaks = (n/5))
  
  ## check model fit -residuals:
  par(mfrow=c(2,2))
  
  # 1) plot residuals versus index of data point; aim is to see how many residuals close to zero?
  plot(difference, xlab = 'Index', ylab = 'Residual', main = 'Residuals Scatter')
  abline(a = 0, b = 0, col = 3) # make a supporting horizontal line at difference = 0
  
  # 2)  plot Residuals vs. Fitted
  plot(fit, difference, xlab = 'Fitted response', ylab = 'Residual', main = 'Residuals vs. Fitted')
  abline(a = 0, b = 0, col = 3) # make a supporting horizontal line at difference = 0
  
  # 3) plot Deviance Residuals vs. Fitted 
  plot(fit, mydev_resid, xlab = 'Fitted response', ylab = 'Deviance Residuals', main ='Deviance Residuals vs. Fitted')
  abline(a = 0, b = 0, col = 3) # make a supporting horizontal line at difference = 0
  
  # 4) Quantile-Quantile plot
  qqnorm(mydev_resid_std, xlab = 'Theoretical Quantiles',ylab ='Standardised Deviance Residuals')
  qqline(mydev_resid_std, col = 3)
  
}

### TESTING SCRIPT ###

# TWO TESTING INSTANCES PRESENTED BELOW #

# 1) generate my own exponential data
## here the approach is to first set a value for beta's, then generate 
# corresponding amount of X variables (with intercept term included as a column of 1's). 
# using the GLM formation, this than can be used to estimate the value of lambda per point
# this can then be used to sample at random from the exponential distribution and generate the
# response y, which will have a different lambda for each datapoint. 

# MY data:
set.seed(26)
size = 100
# beta:
B = as.vector(c(0.1, 0.4, 0.3, 0.2))
# generate X
x1 = rep(1, size)
x4 = rnorm(size, mean = 0, sd = 1)^2
x2 = rnorm(size, mean = 0, sd = 2)^2
x3 =rnorm(size, mean = 0, sd = 2)^2
X = cbind(x1,x2, x3,x4)
# lambda:
mylamb= exp(X %*% B)
# response:
y = rexp(size, rate = mylamb)

# MY test
cat('My Test 1- own data, results:')
mylist1 <- erm(y,X, startval = 0)

# 2) use available dataset BUT modify to fit task
# due to time constraints searching for a perfect exponentially distributed datasets was abandoned in favour of using the known 
# esoph (Smoking, Alcohol and (O)esophageal Cancer) dataset. The responses here; no. of cancer cases, can be modelled by
# the exponential distribution with a slight alteration, that all integers are added 1 to them to prevent the occurrence of zeros.
# this is not adequate if true inference about the data were to be made but for the purpose of testing model it is sufficient. 
library(datasets)
#plot(sort(esoph$ncases,decreasing = TRUE)) # plot proves exponential shape distribution
y<-esoph$ncases+1 # y contains 0 values, thus not suitable to model withe exponential distribution, thus add +1 to data points
plot(sort(y,decreasing = TRUE)) # plot proves exponential shape distribution
# 
X2 <- esoph$ncontrols # take another explanatory var from dataset, this is arguably a bad variable choice as this variable is the number of controls
X1 <- rep(1,length(y)) # inrouduce column of 1s equivalent to the intercept 
X3 <-rnorm(length(y), mean = 0, sd = 2)^2 # also generate another random var
X <-cbind(X1,X2,X3)

# MY test
cat('My Test 2- data loosley based on esoph dataset, results:')
mylist1 <- erm(y,X, startval = 0)

# commentary/ interpretation:
# observations: unsurprisingly intercept is not significant
# the X2 variable, which comes from the data set is somewhat significant, but because it only captures the amount of controls it is not significant 
# Thus the randomly generated normal distribution is the most significant in predicting the y response \
# plots are generaly unsatisfactory as the data itself is bad 
