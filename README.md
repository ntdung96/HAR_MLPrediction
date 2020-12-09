# Human Activity Recognition - A Model Stacking Prediction Approach

## 1. Background
-------------

### a. Sypnosis

Using devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* it is
now possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how *much* of a particular activity they do,
but they rarely quantify *how well they do it*. In this project, your
goal will be to use data from accelerometers on the belt, forearm, arm,
and dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<a href="http://groupware.les.inf.puc-rio.br/har" class="uri">http://groupware.les.inf.puc-rio.br/har</a>
(see the section on the Weight Lifting Exercise Dataset).

### b. Prediction goal

The goal is to predict the manner in which they did the exercise. This
is the “classe” variable in the training set.

This report covers how the model was built, what the expected
out-of-sample error is, and evaluates the prediction model.

## 2. Data
-------

### a. Training set and testing set

The training set is downloaded and stored as “pml-training.csv”. I load
the data and partition it into a training and test set.

    #Load the caret package
    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    #Download and read and store the csv data used for training to train_dat
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                  destfile = "pml-training.csv",
                  method = "curl")
    train_dat <- read.csv("pml-training.csv", 
                          na.strings = c("NA","#DIV/0!", ""))
    train_dat$classe <- as.factor(train_dat$classe) #coerce into factor

    #Remove the first 7 columns of the data since they are the personal information and clearly of no use for the prediction. Also remove "Near Zero variables" and columns with more than 60% of missing data
    train_dat <- train_dat[,-c(1:7)]
    train_dat <- train_dat[,-nearZeroVar(train_dat)]
    tempData <- colSums(is.na(train_dat)) <= 0.6*nrow(train_dat)
    train_dat <- train_dat[, tempData]

    #Create data partitions: train and test for training and testing data, with 75% and 25% of the data points respectively.
    trainIndex <- createDataPartition(train_dat$classe,
                                      p=0.75, list = FALSE)
    train <- train_dat[trainIndex,]
    test <- train_dat[-trainIndex,]

### b. Evaluation set

The evaluation set contains of 20 different test cases.The final model
will be evaluate using this evaluation set.

    #Download and load evaluation data
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                  destfile = "pml-testing.csv",
                  method = "curl")
    eva <- read.csv("pml-testing.csv",
                    na.strings = c("NA","#DIV/0!", ""))[,-c(1:7)]

## 3. Prediction algorithm
-----------------------

### a. Prediction approach

My approach to predict “classe” from the Human Activity Recognition
dataset is rather simple and naive. I will choose a model from the
linear model family and a tree-based algorithm, estimate their expected
out-of-sample error using cross-validation and blend (stacking) them. I
will use Linear Discriminant Analysis (LDA) and Random Forest. I will
stack them using Support Vector Machine (SVM).

### b. Fitting models and prediction algorithms

Throughout the model building process, I will use 7-fold cross
validation integrated in caret to estimate accuracy

#### Model 1: Linear discriminant analysis (LDA)

    #Set seed and set cross validation parameters
    set.seed(4796)
    t.control <- trainControl(method = "repeatedcv", number = 7, repeats = 3,
                                  allowParallel = TRUE)

    #Fit Linear Discriminant Analysis
    model_lda <- train(classe ~ ., method = "lda", trControl = t.control, data = train)
    model_lda

    ## Linear Discriminant Analysis 
    ## 
    ## 14718 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (7 fold, repeated 3 times) 
    ## Summary of sample sizes: 12616, 12616, 12615, 12616, 12616, 12615, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.6984189  0.6181963

The accuracy of LDA model is 0.6984189. This accuracy is estimated from
repeated 7-fold cross-validation with 3 repeats. The estimated
out-of-sample error = 1 - estimated out-of-sample accuracy: 0.3015811.

#### Model 2: Random Forest

    #Fit Random Forest with total 128 trees grown and 7 variables each split with parallel computing
    library(doParallel)

    ## Loading required package: foreach

    ## Loading required package: iterators

    ## Loading required package: parallel

    cl = makeCluster(4)
    registerDoParallel(cl)

    model_rf <- train(classe ~ ., method = "rf", ntree=128,
                      tuneGrid=data.frame(.mtry = 7),
                      trControl = t.control, 
                      data = train)
    stopCluster(cl)
    remove(cl)
    registerDoSEQ()
    model_rf

    ## Random Forest 
    ## 
    ## 14718 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (7 fold, repeated 3 times) 
    ## Summary of sample sizes: 12618, 12616, 12614, 12615, 12615, 12616, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.9941792  0.9926367
    ## 
    ## Tuning parameter 'mtry' was held constant at a value of 7

The accuracy of Random Forest model is 0.9941792. This accuracy is
estimated from repeated 7-fold cross-validation with 3 repeats. The
estimated out-of-sample error = 1 - estimated out-of-sample accuracy:
0.0058208.

#### Model stacking

Now we can prepare and compare the accuracy of predicted classes
produced from 2 models.

    #Prediction from LDA
    pred_lda <- predict(model_lda, newdata = test)

    #Prediction from Random Forest
    pred_rf <- predict(model_rf, newdata = test)

    #Data frame contains the actual classe value in testing set and 2 prediction values from 2 models
    meta <- data.frame(classe_sb = test$classe,
                       lda = pred_lda, 
                       rf = pred_rf)

Now I compared the accuracy of 2 models to the test data.

    confusionMatrix(pred_lda, test$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1140  139   85   51   28
    ##          B   34  597   88   22  153
    ##          C  112  118  570   80   79
    ##          D  103   46   95  630   88
    ##          E    6   49   17   21  553
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7117          
    ##                  95% CI : (0.6988, 0.7243)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6353          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8172   0.6291   0.6667   0.7836   0.6138
    ## Specificity            0.9137   0.9249   0.9039   0.9190   0.9768
    ## Pos Pred Value         0.7900   0.6678   0.5944   0.6549   0.8560
    ## Neg Pred Value         0.9263   0.9122   0.9278   0.9559   0.9183
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2325   0.1217   0.1162   0.1285   0.1128
    ## Detection Prevalence   0.2942   0.1823   0.1956   0.1962   0.1317
    ## Balanced Accuracy      0.8654   0.7770   0.7853   0.8513   0.7953

    confusionMatrix(pred_rf, test$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1394    5    0    0    0
    ##          B    1  941   11    0    0
    ##          C    0    3  843    5    0
    ##          D    0    0    1  798    1
    ##          E    0    0    0    1  900
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9943          
    ##                  95% CI : (0.9918, 0.9962)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9928          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9993   0.9916   0.9860   0.9925   0.9989
    ## Specificity            0.9986   0.9970   0.9980   0.9995   0.9998
    ## Pos Pred Value         0.9964   0.9874   0.9906   0.9975   0.9989
    ## Neg Pred Value         0.9997   0.9980   0.9970   0.9985   0.9998
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2843   0.1919   0.1719   0.1627   0.1835
    ## Detection Prevalence   0.2853   0.1943   0.1735   0.1631   0.1837
    ## Balanced Accuracy      0.9989   0.9943   0.9920   0.9960   0.9993

From the plot above, the LDA model seems to classify not so good
compared to the Random Forest model.

To stack 2 models for even better prediction, I combine 2 models using
Support Vector Machine (SVM). This is “meta-learning” actually.

    #Create prediction from 2 models using SVM
    model_svm <- train(classe_sb ~ ., data = meta, method = "svmLinear",
                       trControl = t.control)
    pred_svm <- predict(model_svm, data = meta)

    #Result
    confusionMatrix(pred_svm, test$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1394    5    0    0    0
    ##          B    1  941   11    0    0
    ##          C    0    3  843    5    0
    ##          D    0    0    1  798    1
    ##          E    0    0    0    1  900
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9943          
    ##                  95% CI : (0.9918, 0.9962)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9928          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9993   0.9916   0.9860   0.9925   0.9989
    ## Specificity            0.9986   0.9970   0.9980   0.9995   0.9998
    ## Pos Pred Value         0.9964   0.9874   0.9906   0.9975   0.9989
    ## Neg Pred Value         0.9997   0.9980   0.9970   0.9985   0.9998
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2843   0.1919   0.1719   0.1627   0.1835
    ## Detection Prevalence   0.2853   0.1943   0.1735   0.1631   0.1837
    ## Balanced Accuracy      0.9989   0.9943   0.9920   0.9960   0.9993

The accuracy of the stacking model is 0.9942891. This accuracy is
estimated from repeated 7-fold cross-validation with 3 repeats. The
estimated out-of-sample error = 1 - estimated out-of-sample accuracy:
0.0057109.

    #Compare models
    compare <- resamples(list(LDA=model_lda, RF=model_rf, stacking=model_svm))
    bwplot(compare, metric = "Accuracy")

![](HAR_Prediction_files/figure-markdown_strict/unnamed-chunk-8-1.png)

## 4. Submission

In the evaluation set, I will apply 2 models of LDA and Random Forest
and stack them using SVM model.

I will make prediction for 20 cases using model 1 (LDA), model 2 (Random
Forest) and final model (stacking model using SVM to blend LDA and
Random Forest)

    #Prediction from 2 models
    eva_mod_lda <- predict(model_lda, eva)
    eva_mod_rf <- predict(model_rf, eva)

    #Model stacking
    stacking <- data.frame(lda = eva_mod_lda,
                           rf = eva_mod_rf)
    pred <- predict(model_svm, newdata = stacking)
    data.frame(eva_mod_lda, eva_mod_rf, pred)

    ##    eva_mod_lda eva_mod_rf pred
    ## 1            B          B    B
    ## 2            A          A    A
    ## 3            B          B    B
    ## 4            C          A    A
    ## 5            C          A    A
    ## 6            E          E    E
    ## 7            D          D    D
    ## 8            D          B    B
    ## 9            A          A    A
    ## 10           A          A    A
    ## 11           D          B    B
    ## 12           A          C    C
    ## 13           B          B    B
    ## 14           A          A    A
    ## 15           E          E    E
    ## 16           A          E    E
    ## 17           A          A    A
    ## 18           B          B    B
    ## 19           B          B    B
    ## 20           B          B    B

The prediction using the final stacking model gives the same result with
the prediction from the Random Forest model.
