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

setwd('C:/Users/Will/Desktop')

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
  shuffle = FALSE,
  validation_split = 0.2,
  verbose = TRUE,
  callbacks = list(loss.history, weight.history)
)

# test_predictions <- model %>% predict(test_data)

w1 <- w2 <- w3 <- list()
b1 <- b2 <- b3 <- list()

for (i in 1:epochs) {

  for (j in 1:6) {

    if (j == 1) w1[[i]] <- weight.history$weight[[(i - 1) * 6 + j]] %>% as.numeric()
    if (j == 2) b1[[i]] <- weight.history$weight[[(i - 1) * 6 + j]] %>% as.numeric()
    if (j == 3) w2[[i]] <- weight.history$weight[[(i - 1) * 6 + j]] %>% as.numeric()
    if (j == 4) b2[[i]] <- weight.history$weight[[(i - 1) * 6 + j]] %>% as.numeric()
    if (j == 5) w3[[i]] <- weight.history$weight[[(i - 1) * 6 + j]] %>% as.numeric()
    if (j == 6) b3[[i]] <- weight.history$weight[[(i - 1) * 6 + j]] %>% as.numeric()

  }

}

epoch <- 1

w1int <- w2int <- w3int <- numeric()
w1coe <- w2coe <- w3coe <- numeric()

b1int <- b2int <- b3int <- numeric()
b1coe <- b2coe <- b3coe <- numeric()

for (i in 1:length(w1[[1]])) {
  fit <- lm(c(w1[[epoch]][i], w1[[epoch + 1]][i]) ~ c(epoch, (epoch + 1)))
  w1int[i] <- fit$coefficients[1]
  w1coe[i] <- fit$coefficients[2]
}

for (i in 1:length(w2[[1]])) {
  fit <- lm(c(w2[[epoch]][i], w2[[epoch + 1]][i]) ~ c(epoch, (epoch + 1)))
  w2int[i] <- fit$coefficients[1]
  w2coe[i] <- fit$coefficients[2]
}

for (i in 1:length(w3[[1]])) {
  fit <- lm(c(w3[[epoch]][i], w3[[epoch + 1]][i]) ~ c(epoch, (epoch + 1)))
  w3int[i] <- fit$coefficients[1]
  w3coe[i] <- fit$coefficients[2]
}


for (i in 1:length(b1[[1]])) {
  fit <- lm(c(b1[[epoch]][i], b1[[epoch + 1]][i]) ~ c(epoch, (epoch + 1)))
  b1int[i] <- fit$coefficients[1]
  b1coe[i] <- fit$coefficients[2]
}

for (i in 1:length(b2[[1]])) {
  fit <- lm(c(b2[[epoch]][i], b2[[epoch + 1]][i]) ~ c(epoch, (epoch + 1)))
  b2int[i] <- fit$coefficients[1]
  b2coe[i] <- fit$coefficients[2]
}

for (i in 1:length(b3[[1]])) {
  fit <- lm(c(b3[[epoch]][i], b3[[epoch + 1]][i]) ~ c(epoch, (epoch + 1)))
  b3int[i] <- fit$coefficients[1]
  b3coe[i] <- fit$coefficients[2]
}
