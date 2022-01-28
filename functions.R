numeric.graph <- function(data, variables, breaks = rep(10, length(variables))) {
  for (var in variables) {
    par(mfrow=c(1,2))
    i=1
    name <- names(data)[var]
    hist(
      data[, var], 
      breaks = breaks[i], 
      probability = T, 
      xlab = name,
      main = paste("Histogram of", name)
    )
    lines(density(data[, var]))
    abline(v=mean(data[, var]), col = "red")
    abline(v=median(data[, var]), col = "green")
    
    boxplot(data[, var], xlab = name)
    
    par(mfrow=c(1,1))
    i = i+1
  }
  
  library(moments)
  #print("Standard errors")
  #print(sqrt(var(abs(data[, variables]))))
  
  print("Skewness indexes")
  print(skewness(data[, variables]))
  
  print("3 - Kurtosis indexes")
  print(3 - kurtosis(data[, variables]))
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