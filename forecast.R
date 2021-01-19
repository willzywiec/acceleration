# housing.R

# initialize environment
if (!is.null(dev.list())) dev.off()
rm(list = ls())
cat('\014')

library(ggplot2)
library(keras)
library(magrittr)
library(mgcv)
library(tibble)

# setwd('C:/Users/Will/Desktop')

boston_housing <- dataset_boston_housing()

c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test

column_names <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'AA', 'LSTAT')

train_df <- as_tibble(train_data)
colnames(train_df) <- column_names

# normalize training data
train_data <- scale(train_data) 

# use means and standard deviations from training set to normalize test set
col_means_train <- attr(train_data, 'scaled:center') 
col_stddevs_train <- attr(train_data, 'scaled:scale')
test_data <- scale(test_data, center = col_means_train, scale = col_stddevs_train)

build_model <- function(loss) {
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = 'relu', input_shape = dim(train_data)[2]) %>%
    layer_dense(units = 64, activation = 'relu') %>%
    layer_dense(units = 1)
  model %>% compile(
    loss = loss,
    optimizer = optimizer_rmsprop(), # default learning rate = 0.001
    # optimizer = optimizer_sgd(),
    metrics = list('mean_absolute_error'))
  model
}

SSE <- function(y_true, y_pred) k_sum(k_pow(y_true - y_pred, 2))

model <- build_model('mean_squared_error') # SSE

model %>% summary()

LossHistory <- R6::R6Class('LossHistory',
                           inherit = KerasCallback,
                           public = list(
                              loss = NULL,
                              on_epoch_end = function(batch, logs = list()) {
                                self$loss<- c(self$loss, logs[["loss"]])
                              }))

WeightHistory <- R6::R6Class('WeightHistory',
                             inherit = KerasCallback,
                             public = list(
                                weight = list(),
                                on_epoch_end = function(batch, logs = list()) {
                                  self$weight <- c(self$weight, get_weights(model))
                                }))

# forecast weights and biases
Forecast <- function(distance, epochs, points, mode, weight.history) {

  weight.length <- length(weight.history$weight)

  layers <- weight.length / epochs

  weight.forecast <- list()

  for (i in 1:layers) {

    w <- weight.history$weight[seq(i, weight.length, layers)]

    fit <- numeric()

    for (j in 1:length(w[[1]])) {
      
      x <- c((epochs - points + 1):epochs)
      y <- unlist(lapply(w, '[[', j))
      y <- y[(epochs - points + 1):epochs]
      
      if (mode == 'lm') {
        fit[j] <- predict(lm(y ~ x), epochs + distance)
      } else if (mode == 'gam') {
        fit[j] <- predict(gam(y ~ s(x)), epochs + distance)
      }
      
    }

    if (i %% 2 != 0) {
      fit <- matrix(fit, nrow = dim(weight.history$weight[[weight.length - layers + i]])[1], ncol = dim(weight.history$weight[[weight.length - layers + i]])[2])
    } else {
      fit <- array(fit)
    }

    weight.forecast[[i]] <- fit

  }

  return(weight.forecast)

}

loss.history <- LossHistory$new()
weight.history <- WeightHistory$new()

loop <- TRUE

if (loop == TRUE) {
  
  epochs <- 50
  
  # start timer
  ptm <- proc.time()
  
  # fit model
  history <- model %>% fit(train_data, train_labels, epochs = epochs, validation_split = 0.2, shuffle = FALSE, verbose = TRUE, callbacks = list(loss.history, weight.history))
  
  # forecast         Forecast(distance, epochs, points, mode, weight.history)
  weight.forecast <- Forecast(data.frame(x = 50), epochs, 15, 'lm', weight.history)
  set_weights(model, weight.forecast)
  new.history <- model %>% fit(train_data, train_labels, epochs = 20, validation_split = 0.2, verbose = TRUE, callbacks = list(loss.history, weight.history))

  proc.time() - ptm
  
} else {
  
  epochs <- 100
  
  # start timer
  ptm <- proc.time()
  
  # fit model
  history <- model %>% fit(train_data, train_labels, epochs = epochs, validation_split = 0.2, verbose = TRUE, callbacks = list(loss.history, weight.history))
  
  proc.time() - ptm
  
}

test <- unlist(weight.history$weight[seq(6, length(weight.history$weight), 6)])
testy <- data.frame(x = 1:70, y = test)
testy <- head(testy, -20)

# ggplot(testy, aes(x, y)) + geom_point() + geom_smooth(method = gam, formula = y ~ s(x), se = FALSE)

plot(test, pch = 20)
# plot(unlist(weight.history$weight[seq(1, length(weight.history$weight), 6)]), pch = 20)
