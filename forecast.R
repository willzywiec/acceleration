# housing.R

# initialize environment
if (!is.null(dev.list())) dev.off()
rm(list = ls())
cat('\014')

library(ggplot2)
library(keras)
library(magrittr)
library(tibble)

epochs <- 10

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
    optimizer = optimizer_rmsprop(),
    # optimizer = optimizer_sgd(),
    metrics = list('mean_absolute_error'))
  model
}

SSE <- function(y_true, y_pred) k_sum(k_pow(y_true - y_pred, 2))

model <- build_model('mean_squared_error') # SSE

initial.weight <- get_weights

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

loss.history <- LossHistory$new()
weight.history <- WeightHistory$new()

# fit model
history <- model %>% fit(
  train_data,
  train_labels,
  epochs = epochs,
  # shuffle = FALSE,
  validation_split = 0.2,
  verbose = TRUE,
  callbacks = list(loss.history, weight.history))

# test_predictions <- model %>% predict(test_data)

# forecast weights and biases
Forecast <- function(distance, epochs, weight.history) {

  layers <- length(weight.history) / epochs

  weights <- fits <- list()

  j <- 0

  for (i in 1:layers) {

    weights[[i]] <- c(weight.history$weight[[epochs - 2 * layers + j]] %>% as.numeric(), weight.history$weight[[epochs - layers + j]] %>% as.numeric())

    print(weights)
    
    x <- c((epochs - 1), epochs)
    y <- weights[[i]]

    fits[[i]] <- predict(lm(y ~ x), distance)

    if (i %% 2 != 0) {
      fits[[i]] <- matrix(fits[[i]], nrow = dim(weight.history$weight[[epochs - layers + j]][1]), ncol = dim(weight.history$weight[[epochs - layers + j]])[2])
    } else {
      fits[[i]] <- array(fits[[i]])
    }

    j <- j + 1

  }

  return(fits)

}

new.model <- build_model('mean_squared_error')

loss.history <- LossHistory$new()
weight.history <- WeightHistory$new()

# fit new model
new.history <- new.model %>% fit(train_data, train_labels, epochs = 10, validation_split = 0.2, verbose = TRUE, callbacks = list(loss.history, weight.history))

weight.forecast <- Forecast(data.frame(x = 10), 10, weight.history)

new.model <- set_weights(build_model('mean_squared_error'), weight.forecast)

new.history <- new.model %>% fit(train_data, train_labels, epochs = 10, validation_split = 0.2, verbose = TRUE, callbacks = list(loss.history, weight.history))
