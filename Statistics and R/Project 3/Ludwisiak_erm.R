# STATG003 ICA 3 GLM 
# Klaudia Ludwisiak 

### TASK: develop a function to fit a glm for an exponentially distributed response variable using IWLS. 
# The exponential function belongs to the exponential family, thus a GLM approach is viable. 
# It should be noted that the exponential distribution is a special case of the gamma distribution where
# the alpha parameter is equal to one and the beta parameter is equal to lambda.
#
### Function Input:
# y a vector of responses
# X, a design matrix of covariates. 
# X Should include intercept column of 1's, if intercept term is to be fitted.
# startval is an initial estimate of the model coefficients.
# startval is one value and this is assigned equaly as a first guess to all beta's 
# (number of betas same as number of explanatory varibles/ columns in X)
#
### Function Output:
# A list object with the following components:
# y: The observed responses.
# fitted: The fitted values. 
# betahat: The estimated regression coefficients.
# sebeta: The standard errors of the estimated regression coefficients. 
# cov.beta: The covariance matrix of the estimated regression coefficients.
# df.model: The degrees of freedom for the model.
# df.residual:The residual degrees of freedom.
# deviance: The deviance for the model.
# AND 2 figures:
# Figure 1) plots to check distributional assumptions
# Figure 2) plots to check model fit 

######################################################################  
## FUNCTION

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

