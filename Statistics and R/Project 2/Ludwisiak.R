# STATISTICS G003 In- Course Assesment 2
# Question 2
# by Klaudia Ludwisiak
##################################################################################
# In this question cells.data is load an visualised (Parts A and B). 
# A function is then implemented to calculate the negative log likelihood (Part C). 
# This in turn is used to calculate the maximum likelihood estimates (MLE) of the 
# parameters mu and tau (Part E), with the corresponding standard errors (Part F).

# ============================== PART A -- SETUP ==============================
# Download and read the data cells.dat 
data=read.table("cells.dat.txt",header=TRUE)

cell = data[,1]
comp = data[,2]
numCells = sum(cell *comp) #total num of cells

# =========================== PART B -- EXPLORATION ===========================
# Obtain and print out the total number of compartments, 
# along with the mean number of cells per compartment:

numComp = sum(comp)
meanCells = numCells/numComp 

cat("\nPart A and B output:\n")
cat("There are ",numCells," cells, containing a total of",numComp,"compartments.\n")
cat("Thus the mean number of cells per compartment is approx. ", round(meanCells,2),"\n")

# plot required barchart:
pdf("MyBarplot.pdf")
bp <- barplot(comp, width = 3, names.arg = cell, xlab = 'Cells', ylab = 'Compartments', main = 'Bar Chart', col = 3)
#text(x<-c(1,6,1,8),y=NULL,meanCells,cex=1,pos=3)
legend(5,250,legend="mean cells = 0.73",bty = "n")
#text(bp, 0, round(meanCells,2) ,cex=1,pos=4) ??? HOW to improve
dev.off()
# ============================  PART C -- FUNCTION ===========================

negbinll <- function(params,dat){
  # This function returns as a single value the negative log likelihood
  # for Negative Binomial distribution. Arguments:
  #
  # param: The vector of parameters of the distribution (tau and mu)
  # dat: A matrix of the data pairs, (x is cells and fx is compartements)
  
  #sanity check (maake sure params are positive needed for part E and F)
  if (params[1] < 0 |params[2] < 0) {
    params[1] <-0.1
    params[2] <-0.1
  } 
  
  #assign values
  r <- params[1]
  mu <-  params[2]
  x <- dat[,1]
  fx <- dat[,2]
  p <- r/(r+mu) # probabily of success
  
  # calculate NB probabilty 
  Px2 <- dnbinom(x, size = r, prob = p, log = FALSE) # NB probabilty mass function
  # alternative parametrisation : 
  # Px3 <- dnbinom(x, size = r, mu = mu, log = FALSE)
  
  # calculate negative log likelihood as per formula provided:
  Lik2 <- fx*log(Px2)
  LogLik <-  -sum(Lik2)
  
  # return  negative log-likelihood
  return(LogLik) 
}

# ============================  PART D -- EVALUATION ===========================

# Use function negbinll to evaluate and print out the negative log-likelihood

# some sample sensible parameter values 
params1 <- c(0.1,0.1)
LogLik1 <- negbinll(params1,data)
params2 <- c(0.1,0.5)
LogLik2 <- negbinll(params2,data)
params3<- c(0.5,0.5)
LogLik3 <- negbinll(params3,data)
params4<- c(1,0.7)
LogLik4 <- negbinll(params4,data)
params5<- c(1.3,1)
LogLik5 <- negbinll(params5,data)

# save output to file:
sink("Ludwisiak_out.txt") 
source("Ludwisiak.r")

##Make a data frame summarising the above results 
# concatenate parameters and Log.Lik values and store in data frame:

paramsMat <- matrix(c(params1, params2, params3, params4, params5), ncol=2, byrow=TRUE)
l <- matrix(round(c(LogLik1, LogLik2, LogLik3, LogLik4, LogLik5),1), ncol=1, byrow=TRUE)
results <- data.frame(cbind(paramsMat,l)) 
colnames(results) <- c("tau","mu","Log.Lik")
rownames(results) <- c("Case 1","Case 2","Case 3","Case 4","Case 5")
attr(results,"title") <- "Summary of log-likelihood for different parameters"
cat("\nPart C and D output:\n")
cat("Summary of log-likelihood for different parameters\n")
print(results)

sink()
# ============================ PART E -- FIND MLE ==================================????

# Use the R function nlm to find and print out the maximum likelihood estimates
# of thau and μ for the data in cells.dat by minimising the negative log likelihood.
# note: minimising the negative log-likelihood == finding the maximum likelihood estimate 

mymin <- nlm(negbinll, params1, dat = data, hessian = TRUE) # hessian needed for part F
# note various starting points for parameter values were inspected and it was discovered
# that they successfully converge at the true mean of the data found in previous parts of 
# the exercise. However, initialising at artificially high values of mu (ie. mu=7) the 
# solution converged to an alternative minimum.

maxTau <- mymin$estimate[1] #???? do i have to convert back from negative log?
maxMu <- mymin$estimate[2]

sink("Ludwisiak_out.txt", append=TRUE) # save and append next portion of results

cat("\nPart E output:\n")
cat("The maximum likelihood estimates, which minimise the negative log likelihood, are:\n")
cat("Tau parameter: ",round(maxTau,4), " and mu parameter: ",round(maxMu,4), "\n")

# ========================== Part F -- STANDARD ERROR =============================?????
# Obtain and print out approximate standard errors for these estimates.

hess <- mymin$hessian

# find variance of parameter components:
var <- diag(solve(hess)) 
# (solve finds the inverse of the matrix and diag returns its diagonal elements)

# assuming by se we mean sd compute the standard error (standard deviation of the sample mena based on the population mean)
se <- sqrt(var)

cat("\nPart F output:\n")
cat("The associated standard error on Tau is approx.",round(se[1],4), "and on Mu approx.",round(se[2],4),"\n") 

# save output to file in cwd
sink()
