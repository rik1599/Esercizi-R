numeric.graph <- function(data, variables, breaks = rep(10, length(variables))) {
  for (var in variables) {
    par(mfrow=c(1,2))
    i=1
    name <- names(data)[var]
    dens <- density(data[, var])
    hist(
      data[, var], 
      breaks = breaks[i], 
      probability = T, 
      xlab = name,
      main = paste("Histogram of", name),
    )
    lines(dens)
    abline(v=mean(data[, var]), col = "red")
    abline(v=median(data[, var]), col = "green")
    
    boxplot(data[, var], xlab = name)
    
    par(mfrow=c(1,1))
    i = i+1
  }
  
  library(moments)
  
  print("Skewness indexes")
  print(skewness(data[, variables]))
  
  print("Kurtosis indexes - 3")
  print(kurtosis(data[, variables]) - 3)
}

categorical.graph <- function(data, variables) {
  for (var in variables) {
    table.var <- table(data[, var])
    name <- names(data)[var]
    par(mfrow=c(1,2))
    
    barplot(table.var, xlab = name, main = paste("Boxplot of", name))
    pie(table.var, main = paste("Piechart of", name))
    
    par(mfrow=c(1,1))
  }
}

scatterplots.graph <- function(response.var, esplicative.vars = c(), data) {
  ylab = names(data)[response.var]
  
  for (var in esplicative.vars) {
    xlab = names(data)[var]
    plot(data[, response.var]~data[, var], xlab = xlab, ylab = ylab)
    lines(lowess(data[, response.var]~data[, var]), lwd = 2)
    abline(lm(data[, response.var]~data[, var]), col="red")
  }
}

boxplots.graph <- function(response.var, esplicative.vars = c(), data) {
  ylab = names(data)[response.var]
  for (var in esplicative.vars) {
    xlab = names(data)[var]
    boxplot(data[, response.var]~data[, var], xlab = xlab, ylab = ylab)
  }
}

qqplot.graph <- function(data, variables) {
  for (var in variables) {
    name <- names(data)[var]
    qqnorm(
      data[, var], 
      main = paste("Normality of", name), 
      xlab = "Quantiles", ylab = "Values"
    )
    
    qqline(data[, var], col = "red", lwd = 2)
  }
}

transformResponseWithBoxCox <- function(lm.object, dataset, response.var) {
  library(MASS)
  boxcox(lm.object)
  lambda <- boxcox(lm.object, plotit = F)
  #Estrae il valore di x in corrispondenza in corrispondenza del valore massimo di y
  lambda.max = lambda$x[which.max(lambda$y)]
  print(paste("Ideal lambda for", response.var, ":", lambda.max))
  return((dataset[, response.var]^lambda.max - 1)/lambda.max)
}