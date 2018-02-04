library("readxl")
library("e1071")

setwd("D:\\Workspace\\R")

data <- read_excel("datasets.xlsx", sheet = 13, col_names = FALSE)

X = cbind(data[1], data[2])
X = sapply(X, as.numeric)
y = sapply(data[length(data)], as.numeric)
y = as.factor(y)
train.data = data.frame(X, y)

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
plot(model, train.data, X__2 ~ X__1, xlim = c(-5, 0), ylim = c(0, 5), fill = TRUE)
print(summary(model))