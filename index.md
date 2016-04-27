# Classifying Common Mistakes in Weight Lifting Motion Using Fitness Tracker Data
Wayne Witzke  


```r
# This analysis presumes that there is a specific directory structure in place.
# That structure is defined here.
knitr::opts_chunk$set( fig.path = "figure/" );
raw.data.dir = "raw";
raw.training.file = file.path( raw.data.dir, "RawTrainingData" );
raw.test.file = file.path( raw.data.dir, "RawTestData" );
suppressMessages(
{
    library(data.table);
    library(caret);
    library(randomForest);
    library(doParallel);
});
set.seed(1869304218);
cluster = makeCluster( detectCores() - 1 );
registerDoParallel( cluster );
```

## Synopsis
We explore the possibility of using personal fitness tracker data to classify
errors in exercise performance. Existing data on weight lifting exercises was
used to create several machine learning models, which were then tested against
out-of-sample data to determine accuracy. A ensemble model approach was found
to yield extremely good results.

## Data Processing

### Data collection and processing overview
This analysis uses data from the Weight Lifting Exercise dataset created by
Velloso, E et al. Training data from this dataset is programmatically
downloaded from [this](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)
location, and test data (also from this dataset) is programmatically downloaded
from [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).
More information about this data and the group that collected it can be found
[here](http://groupware.les.inf.puc-rio.br/har).


```r
if ( !dir.exists( raw.data.dir ) )
{
    dir.create( raw.data.dir, recursive = TRUE );
}

# This function will fetch a file from an online location, timestamp it, and
# then return the timestamp. It only does this if the `output.file` doesn't
# already exist. If it does, then it will merely return the timestamp for the
# already saved raw data set.
GetRawFile = function( url, output.file )
{
    timestamp = "";
    timestamp.file = paste( output.file, "timestamp", sep = "." );

    if ( !file.exists( output.file ) )
    {
        cat( paste( "Downloading", output.file, "\n" ) );
        timestamp = date();
        download.file(
            url = url,
            destfile = output.file
        );
        writeLines( timestamp, timestamp.file );
    }
    else
    {
        cat( paste( "Using existing", output.file, "\n" ) );
        timestamp = readLines( timestamp.file );
    }

    suppressWarnings(
        unzip( output.file, overwrite = FALSE, exdir = raw.data.dir )
    );

    return(timestamp);
}

cat(
    "Raw training data downloaded on",
    GetRawFile("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
    raw.training.file),
    "\n"
);
cat(
    "Raw test data downloaded on",
    GetRawFile("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
    raw.test.file),
    "\n"
);
```

```
## Using existing raw/RawTrainingData 
## Raw training data downloaded on Tue Apr 26 08:33:53 2016 
## Using existing raw/RawTestData 
## Raw test data downloaded on Tue Apr 26 08:33:57 2016
```

The data is comprised of a number of accelerometer and gyroscope measurements,
time sequence information, some categorical values, and summary rows that
divide the observations up into different collection windows. It describes the
movements associated with a specific exercise repeatedly performed by test
subjects in several different ways. Our outcome variable, `classe`, classifies
how the exercises were pereformed, either correctly (`classe = A`), or
incorrectly (`classe != A`), where the exercise was performed with one of four
common mistakes. The full structure of the data can be seen in figure 1.

#####Figure 1: Dataset structure

```r
# Dropping the first column of row labels.
na.st = c("","NA");
exercise.table =
    fread(
        raw.training.file,
        na.strings = na.st,
        drop = c(1),
        showProgress = FALSE
    );
exercise.test =
    fread(
        raw.test.file,
        na.strings = na.st,
        drop = c(1),
        showProgress = FALSE
    );
str(exercise.table, list.len = 1000000000);
```

```
## Classes 'data.table' and 'data.frame':	19622 obs. of  159 variables:
##  $ user_name               : chr  "carlitos" "carlitos" "carlitos" "carlitos" ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : chr  "05/12/2011 11:23" "05/12/2011 11:23" "05/12/2011 11:23" "05/12/2011 11:23" ...
##  $ new_window              : chr  "no" "no" "no" "no" ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : chr  NA NA NA NA ...
##  $ kurtosis_picth_belt     : chr  NA NA NA NA ...
##  $ kurtosis_yaw_belt       : chr  NA NA NA NA ...
##  $ skewness_roll_belt      : chr  NA NA NA NA ...
##  $ skewness_roll_belt.1    : chr  NA NA NA NA ...
##  $ skewness_yaw_belt       : chr  NA NA NA NA ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : chr  NA NA NA NA ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_belt            : chr  NA NA NA NA ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : chr  NA NA NA NA ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ kurtosis_roll_arm       : chr  NA NA NA NA ...
##  $ kurtosis_picth_arm      : chr  NA NA NA NA ...
##  $ kurtosis_yaw_arm        : chr  NA NA NA NA ...
##  $ skewness_roll_arm       : chr  NA NA NA NA ...
##  $ skewness_pitch_arm      : chr  NA NA NA NA ...
##  $ skewness_yaw_arm        : chr  NA NA NA NA ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ kurtosis_roll_dumbbell  : chr  NA NA NA NA ...
##  $ kurtosis_picth_dumbbell : chr  NA NA NA NA ...
##  $ kurtosis_yaw_dumbbell   : chr  NA NA NA NA ...
##  $ skewness_roll_dumbbell  : chr  NA NA NA NA ...
##  $ skewness_pitch_dumbbell : chr  NA NA NA NA ...
##  $ skewness_yaw_dumbbell   : chr  NA NA NA NA ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : chr  NA NA NA NA ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : chr  NA NA NA NA ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_dumbbell: num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_dumbbell  : chr  NA NA NA NA ...
##  $ total_accel_dumbbell    : int  37 37 37 37 37 37 37 37 37 37 ...
##  $ var_accel_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_dumbbell    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_dumbbell   : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_dumbbell     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_dumbbell_x        : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ gyros_dumbbell_y        : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z        : num  0 0 0 -0.02 0 0 0 0 0 0 ...
##  $ accel_dumbbell_x        : int  -234 -233 -232 -232 -233 -234 -232 -234 -232 -235 ...
##  $ accel_dumbbell_y        : int  47 47 46 48 48 48 47 46 47 48 ...
##  $ accel_dumbbell_z        : int  -271 -269 -270 -269 -270 -269 -270 -272 -269 -270 ...
##  $ magnet_dumbbell_x       : int  -559 -555 -561 -552 -554 -558 -551 -555 -549 -558 ...
##  $ magnet_dumbbell_y       : int  293 296 298 303 292 294 295 300 292 291 ...
##  $ magnet_dumbbell_z       : num  -65 -64 -63 -60 -68 -66 -70 -74 -65 -69 ...
##  $ roll_forearm            : num  28.4 28.3 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 ...
##  $ pitch_forearm           : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 ...
##  $ yaw_forearm             : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -152 ...
##  $ kurtosis_roll_forearm   : chr  NA NA NA NA ...
##  $ kurtosis_picth_forearm  : chr  NA NA NA NA ...
##  $ kurtosis_yaw_forearm    : chr  NA NA NA NA ...
##  $ skewness_roll_forearm   : chr  NA NA NA NA ...
##  $ skewness_pitch_forearm  : chr  NA NA NA NA ...
##  $ skewness_yaw_forearm    : chr  NA NA NA NA ...
##  $ max_roll_forearm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_forearm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_forearm         : chr  NA NA NA NA ...
##  $ min_roll_forearm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_forearm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_forearm         : chr  NA NA NA NA ...
##  $ amplitude_roll_forearm  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_forearm : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_forearm   : chr  NA NA NA NA ...
##  $ total_accel_forearm     : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ var_accel_forearm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_forearm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_forearm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_forearm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_forearm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_forearm    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_forearm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_forearm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_forearm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_forearm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_forearm_x         : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
##  $ gyros_forearm_y         : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
##  $ gyros_forearm_z         : num  -0.02 -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 ...
##  $ accel_forearm_x         : int  192 192 196 189 189 193 195 193 193 190 ...
##  $ accel_forearm_y         : int  203 203 204 206 206 203 205 205 204 205 ...
##  $ accel_forearm_z         : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
##  $ magnet_forearm_x        : int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
##  $ magnet_forearm_y        : num  654 661 658 658 655 660 659 660 653 656 ...
##  $ magnet_forearm_z        : num  476 473 469 469 473 478 470 474 476 473 ...
##  $ classe                  : chr  "A" "A" "A" "A" ...
##  - attr(*, ".internal.selfref")=<externalptr>
```

### Creating Data Slices
Before we begin to tidy the data in this analysis, we segment the data into
training and validation sets. Based on the structure of the data and the nature
of the final test, and because we wish to perform multiple tests, the data will
have a simple 80/20 slice, with 80% of the data used for training, and 20% used
for validation. The selection of validation data is done using random sampling.
The final test data was selected independent of this analysis.


```r
inTrain = createDataPartition( exercise.table$classe, p = 0.8, list = FALSE );
exercise.train = exercise.table[inTrain, ];
exercise.validate = exercise.table[-inTrain, ];
```

In addition to creating our own validation set, we will be doing
10-fold-cross-validation as part of every model (hence the need for only 20%
for the final validation step).


```r
#   This appears to do 10-fold cross validation by default, and allow parallel
#   processing to boot. We do, however, appear to need to calculate a bunch of
#   seeds for reproducability.
seeds = vector( mode = "list", length = 11 ); # Apparently the length is
                                              # number*repeat, which are 10 and
                                              # 1 by default in trainControl.
for( ii in 1:11 )
{
    seeds[[ii]] = sample.int(.Machine$integer.max, ncol(exercise.train)-1);
}
#   And apparently we want one value in each seed vector for each tuning
#   parameters, which is just the number of variables in our data right now.
fitControl = trainControl(method = "cv", seed = seeds);
```

### Tidying the data
A cursory examination of this data shows a number of potential problems. For 
instance, many of the variables have a large number of `NA` values. We remove
any variables that have more than 80% of their values missing. 


```r
keep.cols =
    which(
        unlist(
            exercise.train[ , lapply(.SD,function(x) sum(is.na(x)) < 0.2*.N)]
        )
    );
exercise.train = exercise.train[ , keep.cols, with=FALSE];
```

This leaves no missing values in the training set.

```r
cat(
    "There are",
    sum(sapply(exercise.train, function(x) sum(is.na(x)))),
    "NAs remaining.\n"
);
```

```
## There are 0 NAs remaining.
```

In addition, several variables were loaded as character data. One should be a
date/time variable, while the others should be factor variables.


```r
exercise.train$cvtd_timestamp =
    as.POSIXct(exercise.train$cvtd_timestamp, format="%d/%m/%Y %H:%M");
exercise.train$user_name = factor(exercise.train$user_name);
exercise.train$new_window = factor(exercise.train$new_window);
exercise.train$classe = factor(exercise.train$classe);
```

## Model Creation
Because our outcome variable is categorical, we use random forests, boosting,
and linear discriminant analysis to find the best model for prediction. We also
combine these predictors to create an aggregate model to compare against the
individual models.


```r
model.RF =
    suppressMessages(
        train(
            classe ~ .,
            data = exercise.train,
            method = "rf",
            trControl = fitControl 
        )
    );
pred.train.RF = predict(model.RF, exercise.train);
```

```r
model.GBM = 
    suppressMessages(
        train(
            classe ~ .,
            data = exercise.train,
            method = "gbm",
            trControl = fitControl,
            verbose = FALSE
        )
    );
pred.train.GBM = predict(model.GBM, exercise.train);
```

```r
model.LDA = 
    suppressMessages(
        suppressWarnings(
            train(
                classe ~ .,
                data = exercise.train,
                method = "lda",
                trControl = fitControl 
            )
        )
    );
pred.train.LDA = predict(model.LDA, exercise.train);
```

```r
combined.predict =
    data.frame(
        p1 = pred.train.RF,
        p2 = pred.train.GBM,
        p3 = pred.train.LDA,
        classe = exercise.train$classe
    );

model.combined = 
    suppressMessages(
        train(
            classe ~ .,
            data = combined.predict,
            method = "gbm",
            trControl = fitControl,
            verbose = FALSE
        )
    );
pred.train.combined = predict(model.combined, combined.predict);
```

```r
#   We'll clean up here to avoid warning messages.
stopCluster(cluster);
```

Figure 2 shows the in-sample accuracy of the various methods. Clearly, with
100% in-sample accuracy, both random forest and the combined methods seem to
produce the strongest results.

#####Figure 2: Model fit in-sample accuracy

```r
cat("Random Forest model:\n");
confusionMatrix(pred.train.RF, exercise.train$classe);
cat("\nBoosted model:\n");
confusionMatrix(pred.train.GBM, exercise.train$classe);
cat("\nLinear Discriminant Analysis model:\n");
confusionMatrix(pred.train.LDA, exercise.train$classe);
cat("\nCombined model:\n");
confusionMatrix(pred.train.combined, exercise.train$classe);
```

```
## Random Forest model:
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4464    0    0    0    0
##          B    0 3038    0    0    0
##          C    0    0 2738    0    0
##          D    0    0    0 2573    0
##          E    0    0    0    0 2886
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
## 
## Boosted model:
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4464    2    0    0    0
##          B    0 3033    1    0    0
##          C    0    3 2731    3    0
##          D    0    0    6 2567    3
##          E    0    0    0    3 2883
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9987         
##                  95% CI : (0.998, 0.9992)
##     No Information Rate : 0.2843         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9983         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9984   0.9974   0.9977   0.9990
## Specificity            0.9998   0.9999   0.9995   0.9993   0.9998
## Pos Pred Value         0.9996   0.9997   0.9978   0.9965   0.9990
## Neg Pred Value         1.0000   0.9996   0.9995   0.9995   0.9998
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1932   0.1740   0.1635   0.1836
## Detection Prevalence   0.2845   0.1933   0.1743   0.1641   0.1838
## Balanced Accuracy      0.9999   0.9991   0.9985   0.9985   0.9994
## 
## Linear Discriminant Analysis model:
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3823  420  244  152  109
##          B  126 2024  245   95  335
##          C  225  384 1915  304  173
##          D  288  102  294 1983  247
##          E    2  108   40   39 2022
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7495          
##                  95% CI : (0.7427, 0.7563)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6828          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8564   0.6662   0.6994   0.7707   0.7006
## Specificity            0.9177   0.9367   0.9162   0.9291   0.9852
## Pos Pred Value         0.8052   0.7165   0.6381   0.6805   0.9145
## Neg Pred Value         0.9415   0.9212   0.9352   0.9539   0.9359
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2435   0.1289   0.1220   0.1263   0.1288
## Detection Prevalence   0.3024   0.1799   0.1912   0.1856   0.1408
## Balanced Accuracy      0.8870   0.8015   0.8078   0.8499   0.8429
## 
## Combined model:
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4464    0    0    0    0
##          B    0 3038    0    0    0
##          C    0    0 2738    0    0
##          D    0    0    0 2573    0
##          E    0    0    0    0 2886
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

## Model Validation and Selection
In order to select the model that is most likely to fit external data, we
now validate the models using the validation dataset captured earlier. Before
we can do this, we have to prepare it in the same way the training data was
prepared.


```r
exercise.validate = exercise.validate[ , keep.cols, with=FALSE];

exercise.validate$cvtd_timestamp =
    as.POSIXct(exercise.validate$cvtd_timestamp, format="%d/%m/%Y %H:%M");
exercise.validate$user_name = factor(exercise.validate$user_name);
exercise.validate$new_window = factor(exercise.validate$new_window);
exercise.validate$classe = factor(exercise.validate$classe);
```

As can be seen in figure 3, once again both the random forest and combined
models perform identically well, and better than the other two models. Since,
theoretically, it should be a better predictor, the combined model will be used
on the final test data.

#####Figure 3: Estimated out of sample accuracy

```r
pred.validate.RF = predict(model.RF, exercise.validate);
pred.validate.GBM = predict(model.GBM, exercise.validate);
pred.validate.LDA = predict(model.LDA, exercise.validate);
combined.validate.predict =
    data.frame(
        p1 = pred.validate.RF,
        p2 = pred.validate.GBM,
        p3 = pred.validate.LDA,
        classe = exercise.validate$classe
    );
pred.validate.combined = predict(model.combined, combined.validate.predict);
cat("Random Forest model:\n");
confusionMatrix(pred.validate.RF, exercise.validate$classe);
cat("\nBoosted model:\n");
confusionMatrix(pred.validate.GBM, exercise.validate$classe);
cat("\nLinear Discriminant Analysis model:\n");
confusionMatrix(pred.validate.LDA, exercise.validate$classe);
cat("\nCombined model:\n");
confusionMatrix(pred.validate.combined, exercise.validate$classe);
```

```
## Random Forest model:
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    1    0    0    0
##          B    0  757    1    0    0
##          C    0    1  683    0    0
##          D    0    0    0  643    1
##          E    0    0    0    0  720
## 
## Overall Statistics
##                                           
##                Accuracy : 0.999           
##                  95% CI : (0.9974, 0.9997)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9987          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9974   0.9985   1.0000   0.9986
## Specificity            0.9996   0.9997   0.9997   0.9997   1.0000
## Pos Pred Value         0.9991   0.9987   0.9985   0.9984   1.0000
## Neg Pred Value         1.0000   0.9994   0.9997   1.0000   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1930   0.1741   0.1639   0.1835
## Detection Prevalence   0.2847   0.1932   0.1744   0.1642   0.1835
## Balanced Accuracy      0.9998   0.9985   0.9991   0.9998   0.9993
## 
## Boosted model:
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    4    0    0    0
##          B    1  752    1    0    0
##          C    0    3  681    1    0
##          D    0    0    2  639    3
##          E    0    0    0    3  718
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9954          
##                  95% CI : (0.9928, 0.9973)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9942          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9908   0.9956   0.9938   0.9958
## Specificity            0.9986   0.9994   0.9988   0.9985   0.9991
## Pos Pred Value         0.9964   0.9973   0.9942   0.9922   0.9958
## Neg Pred Value         0.9996   0.9978   0.9991   0.9988   0.9991
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1917   0.1736   0.1629   0.1830
## Detection Prevalence   0.2852   0.1922   0.1746   0.1642   0.1838
## Balanced Accuracy      0.9988   0.9951   0.9972   0.9961   0.9975
## 
## Linear Discriminant Analysis model:
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 955 103  64  38  26
##          B  26 504  65  27  68
##          C  52  98 451  82  52
##          D  82  22  95 484  54
##          E   1  32   9  12 521
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7431          
##                  95% CI : (0.7291, 0.7567)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6746          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8557   0.6640   0.6594   0.7527   0.7226
## Specificity            0.9177   0.9412   0.9123   0.9229   0.9831
## Pos Pred Value         0.8052   0.7304   0.6136   0.6567   0.9061
## Neg Pred Value         0.9412   0.9211   0.9269   0.9501   0.9403
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2434   0.1285   0.1150   0.1234   0.1328
## Detection Prevalence   0.3023   0.1759   0.1874   0.1879   0.1466
## Balanced Accuracy      0.8867   0.8026   0.7858   0.8378   0.8529
## 
## Combined model:
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    1    0    0    0
##          B    0  757    1    0    0
##          C    0    1  683    0    0
##          D    0    0    0  643    1
##          E    0    0    0    0  720
## 
## Overall Statistics
##                                           
##                Accuracy : 0.999           
##                  95% CI : (0.9974, 0.9997)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9987          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9974   0.9985   1.0000   0.9986
## Specificity            0.9996   0.9997   0.9997   0.9997   1.0000
## Pos Pred Value         0.9991   0.9987   0.9985   0.9984   1.0000
## Neg Pred Value         1.0000   0.9994   0.9997   1.0000   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1930   0.1741   0.1639   0.1835
## Detection Prevalence   0.2847   0.1932   0.1744   0.1642   0.1835
## Balanced Accuracy      0.9998   0.9985   0.9991   0.9998   0.9993
```
## Conclusions: Final Test Data Prediction and Model Evaluation
Applying this model to the test set, we get the following predictions.


```r
exercise.test = exercise.test[ , keep.cols, with=FALSE];

exercise.test$cvtd_timestamp =
    as.POSIXct(exercise.test$cvtd_timestamp, format="%d/%m/%Y %H:%M");
exercise.test$user_name = factor(exercise.test$user_name);
exercise.test$new_window = factor(exercise.test$new_window);

pred.test.RF = predict(model.RF, exercise.test);
pred.test.GBM = predict(model.GBM, exercise.test);
pred.test.LDA = predict(model.LDA, exercise.test);
combined.test.predict =
    data.frame(
        p1 = pred.test.RF,
        p2 = pred.test.GBM,
        p3 = pred.test.LDA
    );
final.prediction = predict(model.combined, combined.test.predict);
names(final.prediction) = 1:20;
print(final.prediction);
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

This is apparently 100% correct.

It is no surprise that we were able to get such highly accurate results. The
test data apparently comes directly from the original dataset, which, as a time
series, has highly correlated data windows in it. Likely if we were to apply
these models to data generated outside of this initial study, we would not see
such high performance in the model fit.

## Appendix

#### System Information
This analysis was performed using the hardware and software listed in this
section.


```r
sessionInfo();
```

```
## R version 3.2.5 (2016-04-14)
## Platform: x86_64-pc-linux-gnu (64-bit)
## Running under: Ubuntu 15.10
## 
## locale:
##  [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C              
##  [3] LC_TIME=en_US.UTF-8        LC_COLLATE=en_US.UTF-8    
##  [5] LC_MONETARY=en_US.UTF-8    LC_MESSAGES=en_US.UTF-8   
##  [7] LC_PAPER=en_US.UTF-8       LC_NAME=C                 
##  [9] LC_ADDRESS=C               LC_TELEPHONE=C            
## [11] LC_MEASUREMENT=en_US.UTF-8 LC_IDENTIFICATION=C       
## 
## attached base packages:
## [1] splines   parallel  stats     graphics  grDevices utils     datasets 
## [8] methods   base     
## 
## other attached packages:
##  [1] mgcv_1.8-12         nlme_3.1-126        MASS_7.3-45        
##  [4] plyr_1.8.3          gbm_2.1.1           survival_2.38-3    
##  [7] doParallel_1.0.10   iterators_1.0.8     foreach_1.4.3      
## [10] randomForest_4.6-12 caret_6.0-68        ggplot2_2.1.0      
## [13] lattice_0.20-33     data.table_1.9.6    rmarkdown_0.9.5    
## 
## loaded via a namespace (and not attached):
##  [1] Rcpp_0.12.4        compiler_3.2.5     formatR_1.3       
##  [4] nloptr_1.0.4       class_7.3-14       tools_3.2.5       
##  [7] digest_0.6.9       lme4_1.1-11        evaluate_0.8.3    
## [10] gtable_0.2.0       Matrix_1.2-4       yaml_2.1.13       
## [13] SparseM_1.7        e1071_1.6-7        stringr_1.0.0     
## [16] knitr_1.12.3       MatrixModels_0.4-1 stats4_3.2.5      
## [19] grid_3.2.5         nnet_7.3-12        minqa_1.2.4       
## [22] reshape2_1.4.1     car_2.1-2          magrittr_1.5      
## [25] scales_0.4.0       codetools_0.2-14   htmltools_0.3.5   
## [28] pbkrtest_0.4-6     colorspace_1.2-6   quantreg_5.21     
## [31] stringi_1.0-1      munsell_0.4.3      chron_2.3-47
```



