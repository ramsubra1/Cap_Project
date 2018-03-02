library(tidyverse)
library(data.table)
library(tidyverse)
library(tidytext)
library(text2vec)
library(quanteda)
library(tm)
library(tictoc)
library(xgboost)
library(caret) #for dummyVars
library(Metrics) #for performance evaluation
library(foreach)
library(doSNOW)
library(doParallel)
library(RecordLinkage)


train <- fread("C://Users/rsubra02/Documents/Mercari/train.tsv")
test <- fread("C://Users/rsubra02/Documents/Mercari/test.tsv")

#renaming required for binding dataframes
names(train)[1] <- "id"
names(test)[1] <- "id"
names(train)[7] <- "shippingT"
names(test)[6] <- "shippingT"
#Convert price to log(price)
train$price <- log(train$price + 1)

#storing some informations before combining
test_ids <- test$id
num_rows_train <- nrow(train)
train_ids <- train$id

#combining train and test dataframes using rbindlist instead of rbind
comb <- rbindlist(list(train, test), fill = TRUE, use.names = TRUE, idcol = NULL)
rm(train)
rm(test)
gc()

#Get top 3 categories from the category_name column
comb[, c("cat_1","cat_2") := tstrsplit(comb$category_name, split = "/", keep = c(1,2))]

#features from the item description

#feature 1: number of words in item_description
comb$num_words <- sapply(comb$item_description, function(x) length(unlist(strsplit(as.character(x), "\\W+"))))

#feature 2: length of words in description                        
comb$len_words <- str_length(comb$item_description)

#feature 3: average length of words in description                        
comb$avg_len_words <- comb$len_words / comb$num_words

#Cleanup Function
prep_fun = function(x) {
  x %>% 
    # make text lower case
    str_to_lower %>% 
    # remove non-alphanumeric symbols
    str_replace_all("[^[:alpha:]]", " ") %>% 
    # collapse multiple spaces
    str_replace_all("\\s+", " ")
}


#Log of price
#train$LogPrice <- log(train$price+1)
comb$Class <- ifelse(comb$price<=2.398,"Low",
                      ifelse(comb$price<=2.89,"Mid",
                             ifelse(comb$price<=3.401,"High",
                                    ifelse(comb$price<=5,"Higher","Highest"))))


comb$item_description <- prep_fun(comb$item_description)

#LDA Function
LDAFunc <- function(x){
  #x$item_description <- prep_fun(x$item_description)
  tokens = x$item_description %>% 
    tolower %>% 
    word_tokenizer
  it = itoken(tokens, progressbar = FALSE)
  v = create_vocabulary(it,stopwords = stop_words$word) %>% 
    prune_vocabulary(term_count_min = 100, doc_proportion_max = 0.3)
  vectorizer = vocab_vectorizer(v)
  dtm = create_dtm(it, vectorizer, type = "dgTMatrix")
  
  lda_model = LDA$new(n_topics = 5, doc_topic_prior = 0.1, topic_word_prior = 0.01)
  doc_topic_distr = 
    lda_model$fit_transform(x = dtm, n_iter = 1000, 
                            convergence_tol = 0.001, n_check_convergence = 25, 
                            progressbar = FALSE)
  #lda_model$get_top_words(n = 10, lambda = 1)
  unique(as.vector(lda_model$get_top_words(n = 10, lambda = 1)))
  
}

Low <- filter(comb, Class=='Low')
LowWords <- LDAFunc(Low)
Mid <- filter(comb, Class=='Mid')
MidWords <- LDAFunc(Mid)
High <- filter(comb, Class=='High')
HighWords <- LDAFunc(High)
Higher <- filter(comb, Class=='Higher')
HigherWords <- LDAFunc(Higher)
Highest <- filter(comb, Class=='Highest')
HighestWords <- LDAFunc(Highest)
rm(Low,Mid,High,Higher,Highest)

  train <- comb[1:num_rows_train,]
  test <- comb[(num_rows_train+1) : nrow(comb),]
  rm(comb)
    
  # test$item_description <- prep_fun(test$item_description)
    x <- list(LowWords,MidWords,HighWords,HigherWords,HighestWords)
    classes <- c("Low","Mid","High","Higher","Highest")
    # test$Class <- vector("character",nrow(test))
    scores <- vector("numeric",5)
    cores=detectCores()
    cl <- makeCluster(cores,outfile="ramout.txt")
    registerDoSNOW(cl)
    tic()
    foreach(i = 1:nrow(test)) %dopar% {
      for(j in 1:5){
        scores[j] <- sum(sapply(x[[j]],grepl,test$item_description[i]))
      }
      test$Class[i] <- classes[which.max(scores)]
      print(i)
    }
    stopCluster(cl)
    toc()

########################################################################
#Modeling


#Highly recommended to watch https://www.youtube.com/watch?v=Og7CGAfSr_Y&t=37s

#Select the variables
myTrain <- train %>% select(-id,-name,-category_name,-brand_name,-item_description,
                            -Class)
##  myTrain <- train %>% select(-id,-name,-category_name,-brand_name,-item_description,
##                              -level4cat,-level5cat,-Class,-LogPrice)
#Binarize all factors
tic()
dmy <- dummyVars("~.",data=myTrain)
myTrainDmy <- data.frame(predict(dmy, newdata = myTrain))
toc()

#Outcome name
outcomeName <- c('price')
#List of predictors
predictors <- names(myTrainDmy)[!names(myTrainDmy)%in%outcomeName]

#Train on 10% of the data
trainPortion <- floor(nrow(myTrainDmy)*0.1)
trainSet <- myTrainDmy[1:floor(trainPortion/2),]
testSet <- myTrainDmy[floor(trainPortion/2)+1:trainPortion,]

#Model Tuning
smallestError <- 100
for (depth in seq(1,20,1))  {
  for (rounds in seq(1,20,1)) {
    
    # train
    bst <- xgboost(data = as.matrix(trainSet[,predictors]),
                   label = trainSet[,outcomeName],
                   max.depth=depth, nround=rounds,
                   objective = "reg:linear", verbose=0,nthread=8)
    gc()
    
    # predict
    predictions <- abs(predict(bst, as.matrix(testSet[,predictors]), outputmargin=TRUE))
    err <- Metrics::rmsle((as.numeric(testSet[,outcomeName])+1), (as.numeric(predictions)+1))
    
    if (err < smallestError) {
      smallestError = err
      print(paste(depth,rounds,err))
    }     
  }
} 

#Now with cross validation
cv <- 30
trainSet <- myTrainDmy[1:trainPortion,]
cvDivider <- floor(nrow(trainSet) / (cv+1))

smallestError <- 100
for (depth in seq(1,10,1)) { 
  for (rounds in seq(1,20,1)) {
    totalError <- c()
    indexCount <- 1
    for (cv in seq(1:cv)) {
      # assign chunk to data test
      dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
      dataTest <- trainSet[dataTestIndex,]
      # everything else to train
      dataTrain <- trainSet[-dataTestIndex,]
      
      bst <- xgboost(data = as.matrix(dataTrain[,predictors]),
                     label = dataTrain[,outcomeName],
                     max.depth=depth, nround=rounds,
                     objective = "reg:linear", verbose=0,
                     nthread=8)
      gc()
      predictions <- predict(bst, as.matrix(dataTest[,predictors]), outputmargin=TRUE)
      
      err <- rmsle(as.numeric(dataTest[,outcomeName]), as.numeric(predictions))
      totalError <- c(totalError, err)
    }
    if (mean(totalError) < smallestError) {
      smallestError = mean(totalError)
      print(paste(depth,rounds,smallestError))
    }  
  }
} 

#Testing the models
trainSet <- myTrainDmy[ 1:trainPortion,]

# assign everything else to test
testSet <- myTrainDmy[(trainPortion+1):nrow(myTrainDmy),]

bst <- xgboost(data = as.matrix(trainSet[,predictors]),
               label = trainSet[,outcomeName],
               max.depth=8, nround=9, objective = "reg:linear", verbose=0)
pred <- predict(bst, as.matrix(testSet[,predictors]), outputmargin=TRUE)
rmsle(as.numeric(testSet[,outcomeName]), as.numeric(pred))
submission <- as.data.frame(cbind(train$id[(trainPortion+1):nrow(myTrainDmy)],pred))
colnames(submission) <- c("test_id","price")
write.csv(submission,"sub1_6_9.csv")
#Let's try this on the test dataset
#Select the variables
myTest <- test %>% select(item_condition_id,shipping,ItemDescLen,CategoryTags,level1cat,Class)

#Binarize all factors
dmy <- dummyVars("~.",data=myTest)
myTestDmy <- data.frame(predict(dmy, newdata = myTest))
pred <- predict(bst,as.matrix(myTestDmy[,predictors]),outputmargin = TRUE)

submission <- as.data.frame(cbind(test$test_id,pred))
colnames(submission) <- c("test_id","price")
write.csv(submission,"sub1_6_9.csv")


bst <- xgboost(data = as.matrix(trainSet[,predictors]),
               label = trainSet[,outcomeName],
               max.depth=3, nround=20, objective = "reg:linear", verbose=0)
pred <- predict(bst, as.matrix(testSet[,predictors]), outputmargin=TRUE)
rmsle(as.numeric(testSet[,outcomeName]), as.numeric(pred))


