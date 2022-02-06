numeric.graph <- function(data,
                          #Indici delle variabili nel dataset per cui fare gli istogrammi
                          variables,
                          #vettore di indici per cui verificare graficamente delle ipotesi di normalità (con densità e qqplot)
                          qqplot = c(),
                          draw.normal.curve = c(),
                          no.density = c()) {
  for (var in variables) {
    par(mfrow=c(1,2))
    name <- names(data)[var]
    
    MASS::hist.scott(
      data[, var], 
      prob = T,
      xlab = name,
      main = paste("Histogram of", name)
    )
    
    if (var %in% draw.normal.curve || var %in% qqplot) {
      xx <- seq(min(data[, var]), max(data[, var]), 0.1)
      lines(xx, dnorm(xx, mean = mean(data[, var]), sd = sd(data[, var])), col="blue", lwd=2)
    }
    
    if (!(var %in% no.density)) {
      dens <- density(data[, var])
      lines(dens, col="orange", lwd=2)
    }
    abline(v=mean(data[, var]), col = "red")
    abline(v=median(data[, var]), col = "green")
    
    boxplot(data[, var], xlab = name)
    par(mfrow=c(1,1))
    
    if (var %in% qqplot) {
      qqnorm(
        data[, var], 
        main = paste("Normal Q-Q Plot for", name)
      )
      qqline(data[, var], col = "red", lwd = 2)
    }
  }
  
  print("Skewness indexes")
  print(moments::skewness(data[, variables]))
  
  print("Kurtosis indexes - 3")
  print(moments::kurtosis(data[, variables]) - 3)
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
    xlab <- names(data)[var]
    plot(data[, response.var]~data[, var], xlab = xlab, ylab = ylab)
    lines(lowess(data[, response.var]~data[, var]), lwd = 2)
    abline(lm(data[, response.var]~data[, var]), col="red")
  }
}

boxplots.graph <- function(response.var, esplicative.vars = c(), data, inverse=F) {
  ylab <- names(data)[response.var]
  for (var in esplicative.vars) {
    xlab = names(data)[var]
    if (inverse) {
      boxplot(data[, var]~data[, response.var], xlab = ylab, ylab = xlab)
    } else {
      boxplot(data[, response.var]~data[, var], xlab = xlab, ylab = ylab)
    }
  }
}

xyplot.graph <- function(response.var, esplicative.var, groups.var, data) {
  levels <- levels(data[, groups.var])
  graph <- lattice::xyplot(
    data[, response.var] ~ data[, esplicative.var],
    groups = data[, groups.var], pch=19,
    key=lattice::simpleKey(text = as.character(levels),
                           space="top",
                           cex=1.2,
                           lwd=6,
                           columns=length(levels)),
    type=c("p","r"), lwd=2,
    xlab = names(data)[esplicative.var],
    ylab = names(data)[response.var])
  
  return(graph)
}

transformResponseWithBoxCox <- function(lm.object, dataset, response.var) {
  library(MASS)
  boxcox(lm.object)
  lambda <- boxcox(lm.object, plotit = F)
  #Estrae il valore di x in corrispondenza in corrispondenza del valore massimo di y
  lambda.max = lambda$x[which.max(lambda$y)]
  print(paste("Ideal lambda for", names(dataset)[response.var], ":", lambda.max))
  
  if (lambda.max == 0) {
    return(log(dataset[, response.var]))
  } else {
    return((dataset[, response.var]^lambda.max - 1)/lambda.max)
  }
}

barplot.graph <- function(response.var, esplicative.vars = c(), data) {
  ylab <- names(data)[response.var]
  for (var in esplicative.vars) {
    xlab <- names(data)[var]
    cont.table <- table(data[, var], data[, response.var])
    
    par(mfrow=c(1,2))
    mosaicplot(
      cont.table,
      ylab = ylab,
      xlab = xlab,
      main = paste(ylab, "~", xlab),
      )
    barplot(
      prop.table(t(cont.table), margin = 2), 
      width = 2, 
      beside = T, 
      legend.text = colnames(cont.table))
    par(mfrow=c(1,1))
  }
}

logitResponse.modelVerify <- function(lm.object, response.var) {
  plot(
    lm.object$data[, response.var],
    lm.object$fitted.values,
    pch=19,
    ylim=c(-0.1,1.1),
    xlab=names(lm.object$data)[response.var],
    ylab="Fitted values"
    )
  abline(0,0,col='red',lwd=2)
  abline(1,0,col='red',lwd=2)
  
  correctness <- lm.object$fitted.values >= 0 && lm.object$fitted.values <= 1
  if (correctness) {
    print("Il modello è corretto, tutti i fitted values sono compresi tra 0 e 1")
  } else {
    print("Il modello è errato, ci sono fitted values non compresi tra 0 e 1")
    print(lm.object$fitted.values[
      lm.object$fitted.values < 0 | lm.object$fitted.values > 1])
  }
  
  library(DAAG)
  CVbinary(lm.object)
}

logitResponse.modelVerify.verbose <- function(glm.object, 
                                              response.var, 
                                              regressors, 
                                              data, 
                                              K=10, B=1,
                                              values=c(1,0),
                                              roc = F) {
  pred <- ifelse(predict(glm.object, data, type='response') > 0.5, values[1], values[2])
  cm <- crossval::confusionMatrix(data[, response.var], pred, negative = values[2])
  de <- crossval::diagnosticErrors(cm)
  
  predfun.glm <- function(train.x, train.y, test.x, test.y, negative) {
    train.data <- data.frame(train.x, train.y)
    glm.fit <- glm(train.y~., binomial, train.data)
    ynew <- predict(glm.fit, newdata=data.frame(test.x, test.y), type="response")
    ynew <- as.numeric(ynew>0.5)
    out <- crossval::confusionMatrix(test.y, ynew, negative=values[2])
    return(out)
  }
  
  cv.out <- crossval::crossval(
    predfun.glm, 
    data[, regressors], 
    data[, response.var], 
    K=K, B=B, negative=values[2], verbose = FALSE)
  de.cv <- crossval::diagnosticErrors(cv.out$stat)
  
  tabe <- rbind(de[c(1,3,2)], de.cv[c(1,3,2)])
  rownames(tabe) <- c("training data","cross-validated")
  colnames(tabe) <- c("tot. accuracy","true negative","true positive")
  
  if (roc) {
    booleanResponse <- ifelse(data[, response.var] == values[1], 1, 0)
    verification::roc.plot(booleanResponse, fitted(glm.object), 
                           xlab='false positive rate = 1-true negative rate', 
                           ylab='true positive rate')
  }
  
  return(tabe)
}

calculateMSE <- function(glm.object, data) {
  print(paste(
    "Training MSE:", 
    sum(resid(glm.object)^2)/length(data[, 1])
  ))
  print(paste(
    "Cv estimate test MSE:",
    boot::cv.glm(data, glm.object)$delta[2]
  ))
}