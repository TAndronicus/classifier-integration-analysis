# install.packages('readxl')
# install.packages('e1071')

library("readxl")
library("e1071")

# setwd("D:\\Workspace\\R")

print("Reading data")
data <- read_excel("datasets.xlsx", sheet = 7, col_names = FALSE)

print("Preparing data")
X = cbind(data[1], data[2])
X = sapply(X, as.numeric)
y = sapply(data[length(data)], as.numeric)
y = as.factor(y)
train.data = data.frame(X, y)

print("Training model")
# model <- svm(X, y, kernel = "linear", type = "C-classification")
model <- svm(y ~ ., data = train.data, kernel = "linear")
print(names(model))
print( t(model$coefs) %*% model$SV)
coefs = model$x.scale[['scaled:scale']]
print(summary(model))
print(names(train.data))
# my.svm <- svm(train.data[, "X__9"] ~ , kernel = "linear", data = train.data, type = "C-classification")
# my.svm <- svm(~ ., kernel = "linear", data = train.data, type = "C-classification")
# print(summary(my.svm))
# plot(my.svm, train.data, X__2 ~ X__1)

print("Plotting model")
plot(model, train.data, X__2 ~ X__1, xlim = c(-0.5, 1.5), ylim = c(-0.5, 1.5), fill = TRUE)
# plot(model, train.data, X__2 ~ X__1)
print(summary(model))

print("Checking accuracy")
pred <- fitted(model)
print(table(pred, y))

print("Checking decission values")
pred <- predict(model, X, decision.values = TRUE)
# print(attr(pred, "decision.values"))
