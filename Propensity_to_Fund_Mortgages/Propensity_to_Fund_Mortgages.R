install.packages(c('caret', 'skimr', 'RANN', 'randomForest', 'fastAdaboost', 'gbm', 'xgboost', 'caretEnsemble', 'C50', 'earth'))


##---- working directory
setwd("C:/Users/bend/Dropbox/PROJECTS/2.COMPETITIONS/Propensity_to_Fund_Mortgages")

##---- Library 
library(caret)

##---- Input data
data <- read.table("data/CAX_MortgageModeling_Train.csv", header = TRUE, sep = ",")
head(data)
#train <- select(data,-c("Unique_ID", "RESULT", "MORTGAGE.NUMBER"))

# Create the training and test datasets
set.seed(100)
trainRowNumbers <- createDataPartition(data$RESULT, p=0.8, list=FALSE)
trainData <- data[trainRowNumbers,]
testData <- data[-trainRowNumbers,]

# Store X and Y for later use.
cols_to_drop <- c("Unique_ID", "RESULT", "MORTGAGE.NUMBER")
x = trainData[, -c(1,2, 22)]
head(x)
y = trainData$RESULT


##---- Descriptive statistics
library(skimr)
skimmed <- skim_to_wide(trainData)
skimmed[, c(1:5, 9:11, 13, 15:16)]


##----  Impute missing values using preProcess
# Create the knn imputation model on the training data
#preProcess_missingdata_model <- preProcess(trainData, method='knnImpute')
#preProcess_missingdata_model

# Use the imputation model to predict the values of missing data points
#library(RANN)  # required for knnInpute
#trainData <- predict(preProcess_missingdata_model, newdata = trainData)
#anyNA(trainData)


##---- One-Hot Encoding
dummies_model <- dummyVars(RESULT ~ ., data=trainData)
trainData_mat <- predict(dummies_model, newdata = trainData)
trainData <- data.frame(trainData_mat)


##---- How to preprocess to transform the data?
preProcess_range_model <- preProcess(trainData, method='range')
trainData <- predict(preProcess_range_model, newdata = trainData)

# Append the Y variable
trainData$RESULT <- y

apply(trainData[, 1:10], 2, FUN=function(x){c('min'=min(x), 'max'=max(x))})


##---- visualize the importance of variables using featurePlot()

featurePlot(x = trainData[, 1:18], 
            y = trainData$RESULT, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))


featurePlot(x = trainData[, 1:18], 
            y = trainData$RESULT, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))


##---- recursive feature elimination

set.seed(100)
options(warn=-1)

subsets <- c(1:22)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x=trainData[, 1:18], y=trainData$RESULT,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile


##---- Training and Tuning the model
# See available algorithms in caret
modelnames <- paste(names(getModelInfo()), collapse=',  ')
modelnames

modelLookup('earth')

# Set the seed for reproducibility
set.seed(100)

# Train the model using randomForest and predict on the training data itself.
model_mars = train(RESULT ~ ., data=trainData, method='earth')
fitted <- predict(model_mars)
model_mars

plot(model_mars, main="Model Accuracies with MARS")


##---- variable importance
varimp_mars <- varImp(model_mars)
plot(varimp_mars, main="Variable Importance with MARS")


##---- Prepare the test dataset and predict
# Step 1: Impute missing values 
testData2 <- predict(preProcess_missingdata_model, testData)  

# Step 2: Create one-hot encodings (dummy variables)
testData3 <- predict(dummies_model, testData2)

# Step 3: Transform the features to range between 0 and 1
testData4 <- predict(preProcess_range_model, testData3)
head(testData4[, 1:10])


##---- Predict on testData
# Predict on testData
predicted <- predict(model_mars, testData4)
head(predicted)

# Compute the confusion matrix
confusionMatrix(reference = testData$RESULT, data = predicted, mode='everything', positive='MM')


##---- hyperparameter tuning
# Define the training control
fitControl <- trainControl(
    method = 'cv',                   # k-fold cross validation
    number = 5,                      # number of folds
    savePredictions = 'final',       # saves predictions for optimal tuning parameter
    classProbs = T,                  # should class probabilities be returned
    summaryFunction=twoClassSummary  # results summary function
)


# Step 1: Tune hyper parameters by setting tuneLength
set.seed(100)
model_mars2 = train(RESULT ~ ., data=trainData, method='earth', tuneLength = 5, metric='ROC', trControl = fitControl)
model_mars2

# Step 2: Predict on testData and Compute the confusion matrix
predicted2 <- predict(model_mars2, testData4)
confusionMatrix(reference = testData$RESULT, data = predicted2, mode='everything', positive='MM')


# Step 1: Define the tuneGrid
marsGrid <-  expand.grid(nprune = c(2, 4, 6, 8, 10), 
                        degree = c(1, 2, 3))

# Step 2: Tune hyper parameters by setting tuneGrid
set.seed(100)
model_mars3 = train(RESULT ~ ., data=trainData, method='earth', metric='ROC', tuneGrid = marsGrid, trControl = fitControl)
model_mars3

# Step 3: Predict on testData and Compute the confusion matrix
predicted3 <- predict(model_mars3, testData4)
confusionMatrix(reference = testData$RESULT, data = predicted3, mode='everything', positive='MM')


##---- evaluate performance of multiple machine learning algorithms?

# Train the model using adaboost
set.seed(100)
model_adaboost = train(RESULT ~ ., data=trainData, method='adaboost', tuneLength=2, trControl = fitControl)
model_adaboost

# Train the model using rf
set.seed(100)
model_rf = train(RESULT ~ ., data=trainData, method='rf', tuneLength=5, trControl = fitControl)
model_rf

# Train the model using MARS
set.seed(100)
model_xgbDART = train(RESULT ~ ., data=trainData, method='xgbDART', tuneLength=5, trControl = fitControl, verbose=F)
model_xgbDART

# Train the model using MARS
set.seed(100)
model_svmRadial = train(RESULT ~ ., data=trainData, method='svmRadial', tuneLength=15, trControl = fitControl)
model_svmRadial

# Compare model performances using resample()
models_compare <- resamples(list(ADABOOST=model_adaboost, RF=model_rf, XGBDART=model_xgbDART, MARS=model_mars3, SVM=model_svmRadial))

# Summary of the models performances
summary(models_compare)

# Draw box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)


##---- Ensembling the predictions
library(caretEnsemble)

# Stacking Algorithms - Run multiple algos in one call.
trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

algorithmList <- c('rf', 'adaboost', 'earth', 'xgbDART', 'svmRadial')

set.seed(100)
models <- caretList(RESULT ~ ., data=trainData, trControl=trainControl, methodList=algorithmList) 
results <- resamples(models)
summary(results)

# Box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)


##----  combine the predictions of multiple models to form a final prediction
# Create the trainControl
set.seed(101)
stackControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

# Ensemble the predictions of `models` to form a new combined prediction based on glm
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

# Predict on testData
stack_predicteds <- predict(stack.glm, newdata=testData4)
head(stack_predicteds)