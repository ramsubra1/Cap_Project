library(ggplot2)
library(scales)
library(stringr)
library(wordcloud)
library(RColorBrewer)
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

###Open the training file
df_train <- fread("C://Users/rsubra02/Documents/Mercari/train.tsv")
train <- df_train

### Data Overview
###Load the 5 Head and 5 tail records of the file
head(df_train)
tail(df_train)

##Check the number of records
dim(df_train)

###Check the size of the file
print(object.size(df_train), units = 'Mb')

###Check the summary of train data
summary(df_train)


##Varianle Analysis - Price
###Price Ranges
range(df_train$price)

ggplot(data = df_train, aes(x = log(price+1))) + 
  geom_histogram(fill = 'orangered2') +
  labs(title = 'Histogram of log item price + 1')

##Variable ANalysis - Item Condition

table(df_train$item_condition_id)

##number of items by condition

df_train[, .N, by = item_condition_id] %>%
  ggplot(aes(x = as.factor(item_condition_id), y = N/1000)) +
  geom_bar(stat = 'identity', fill = 'cyan2') + 
  labs(x = 'Item condition', y = 'Number of items (000s)', title = 'Number of items by condition category')

df_train[, .(.N, median_price = median(price)), by = item_condition_id][order(item_condition_id)]

ggplot(data = df_train, aes(x = as.factor(item_condition_id), y = log(price + 1))) + 
  geom_boxplot(fill = 'cyan2', color = 'darkgrey')

###shipping

table(df_train$shipping)

###Distribution of price by shipping where 1 is by seller and 0 is by purchaser
df_train %>%
  ggplot(aes(x = log(price+1), fill = factor(shipping))) + 
  geom_density(adjust = 6, alpha = 0.8) + 
  labs(x = 'Log price', y = '', title = 'Distribution of price by shipping')

#plot to find the top 25 median price by brandname
df_train[, .(median_price = median(price)), by = brand_name] %>%
  head(25) %>%
  ggplot(aes(x = reorder(brand_name, median_price), y = median_price)) + 
  geom_point(color = 'cyan2') + coord_flip()

#find the number of unique categories
length(unique((df_train$category_name)))
sort(table(df_train$category_name), decreasing = TRUE)[1:10]
##Median price by item category
df_train[, .(median = median(price)), by = category_name][order(median, decreasing = TRUE)][1:30] %>%
  ggplot(aes(x = reorder(category_name, median), y = median)) + 
  geom_point(color = 'orangered2') + 
  coord_flip()

##Finding out the top 10 categories

sort(table(df_train$category_name), decreasing = TRUE)[1:10]

##most of the categories are 3 levels. Let us break them to see how many unique level 1 and 
#level 2 categories are there and the numbers in each
# split the item category_name by '/' and get the first two category levels as separate columns
df_train[, c("level_1_cat", "level_2_cat") := tstrsplit(df_train$category_name, split = "/", keep = c(1,2))]

# peek at the first few rows to make sure this worked correctly. 
head(df_train[, c("level_1_cat", "level_2_cat")])

##finding the top level categories
table(df_train$level_1_cat)

##check for the price variation between categories
df_train %>%
  ggplot(aes(x = level_1_cat, y = log(price+1))) + 
  geom_boxplot(fill = 'cyan2', color = 'darkgrey') + 
  coord_flip() + 
  labs(x = '', y = 'Log price + 1', title = 'Boxplot of price by top-level category')

# get number of unique level 2 categories
length(unique(df_train$level_2_cat))
#Create a box plot for level 2 categories
df_train %>%
  ggplot(aes(x = level_2_cat, y = log(price+1))) + 
  geom_boxplot(fill = 'cyan2', color = 'darkgrey') + 
  coord_flip() + 
  labs(x = '', y = 'Log price + 1', title = 'Boxplot of price by second-level category')

p1 <-
  df_train[, .N, by = c('level_1_cat', 'item_condition_id')] %>%
  ggplot(aes(x = item_condition_id, y = level_1_cat, fill = N/1000)) +
  geom_tile() +
  scale_fill_gradient(low = 'lightblue', high = 'cyan4') +
  labs(x = 'Condition', y = '', fill = 'Number of items (000s)', title = 'Item count by category and condition') +
  theme_bw() + 
  theme(legend.position = 'bottom')

p2 <-
  df_train[, .(median_price = median(price)), by = c('level_1_cat', 'item_condition_id')] %>%
  ggplot(aes(x = item_condition_id, y = level_1_cat, fill = median_price)) +
  geom_tile() +
  scale_fill_gradient(low = 'lightblue', high = 'cyan4', labels = dollar) +
  labs(x = 'Condition', y = '', fill = 'Median price', title = 'Item price by category and condition') + 
  theme_bw() + 
  theme(legend.position = 'bottom', axis.text.y = element_blank())

grid.arrange(p1, p2, ncol = 2)

##Item Description

#Check if there is a relationship between the length of description and price
df_train[, desc_length := nchar(item_description)]

# set desc_length to NA where no description exists
df_train[item_description == 'No description yet', desc_length := NA]

cor(df_train$desc_length, df_train$price, use = 'complete.obs')

df_train[item_description == 'No description yet', item_description := NA]


dcorpus <- corpus(df_train$item_description)

dfm1 <- dfm(dcorpus, ngrams = 1, ignoredFeatures = c("rm", stopwords("english")), remove_punct = TRUE, remove_numbers = TRUE, stem = TRUE)

tf <- topfeatures(dfm1, n = 25)
data.frame(term = names(tf), freq = unname(tf)) %>%
  ggplot(aes(x = reorder(term, freq), y = freq/1000)) + 
  geom_bar(stat = 'identity', fill = 'orangered2') + 
  labs(x = '', y = 'Frequency (000s)', title = '25 most common description words') + 
  coord_flip() 
set.seed(100)
textplot_wordcloud(dfm1, min.freq = 3e4, random.order = FALSE,
                   rot.per = .25, 
                   colors = RColorBrewer::brewer.pal(8,"Dark2"))

dfm2 <- dcorpus %>%
  corpus_sample(size = floor(ndoc(dcorpus) * 0.15)) %>%
  dfm(
    ngrams = 2,
    ignoredFeatures = c("rm", stopwords("english")),
    remove_punct = TRUE,
    remove_numbers = TRUE,
    concatenator = " "
  )

# get 25 most common bigrams
tf <- topfeatures(dfm2, n = 25)

# convert to df and plot
data.frame(term = names(tf), freq = unname(tf)) %>%
  ggplot(aes(x = reorder(term, freq), y = freq/1000)) + 
  geom_bar(stat = 'identity', fill = 'orangered2') + 
  labs(x = '', y = 'Frequency (000s)', title = '25 most common description bigrams') + 
  coord_flip() 

set.seed(100)
textplot_wordcloud(dfm2, min.freq = 2000, random.order = FALSE,
                   rot.per = .25, 
                   colors = RColorBrewer::brewer.pal(8,"Dark2"))

# get 25 most common trigrams
dfm3 <- dcorpus %>%
  corpus_sample(size = floor(ndoc(dcorpus) * 0.15)) %>%
  dfm(
    ngrams = 3,
    ignoredFeatures = c("rm", stopwords("english")),
    remove_punct = TRUE,
    remove_numbers = TRUE,
    concatenator = " "
  )

tf <- topfeatures(dfm3, n = 25)

# convert to df and plot
data.frame(term = names(tf), freq = unname(tf)) %>%
  ggplot(aes(x = reorder(term, freq), y = freq/1000)) + 
  geom_bar(stat = 'identity', fill = 'orangered2') + 
  labs(x = '', y = 'Frequency (000s)', title = '25 most common description 3-grams') + 
  coord_flip() 

set.seed(100)
textplot_wordcloud(dfm3, min.freq = 100, random.order = FALSE,
                   rot.per = .25, 
                   colors = RColorBrewer::brewer.pal(8,"Dark2"))

##Add other features as docvars
docvars(dcorpus, "price") <- df_train$price
docvars(dcorpus, "brand_name") <- df_train$brand_name
docvars(dcorpus, "item_condition_id") <- df_train$item_condition_id
docvars(dcorpus, "level_1_cat") <- df_train$level_1_cat
docvars(dcorpus, "level_2_cat") <- df_train$level_2_cat

p1 <- summary(dcorpus) %>%
  ggplot(aes(x = level_1_cat, y = Tokens)) +
  geom_boxplot(aes(fill = level_1_cat), color = 'grey') +
  coord_flip() +
  theme(legend.position = 'bottom') + 
  labs(x = '', y = 'Number of tokens in description')

p2 <- summary(dcorpus) %>%
  ggplot(aes(x = Tokens)) +
  geom_density(aes(fill = level_1_cat), color = 'grey') + 
  facet_wrap(~level_1_cat) + 
  theme(legend.position = "none") + 
  labs(x = 'Number of tokens in description')

grid.arrange(p1, p2, ncol = 2)


test <- fread("test.tsv")

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

numRowTrain =num_rows_train * 0.1


train <- comb[1:round(numRowTrain,digits=0),]
test <- comb[round(numRowTrain,digits=0) + 1 : round(numRowTrain,digits=0),]
##test <- comb[(num_rows_train+1) : nrow(comb),]
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
    err <- Metrics::rmse((as.numeric(testSet[,outcomeName])+1), (as.numeric(predictions)+1))
    
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
      
      err <- rmse(as.numeric(dataTest[,outcomeName]), as.numeric(predictions))
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
testSet <- myTrainDmy[(trainPortion+1):nrow(myTrainDmy)/2,]

bst_final <- xgboost(data = as.matrix(trainSet[,predictors]),
                     label = trainSet[,outcomeName],
                     max.depth=7, nround=20, objective = "reg:linear", verbose=0)

importance_matrix <- xgb.importance(predictors, model = bst_final)

xgb.plot.importance(importance_matrix[1:15], rel_to_first = TRUE, xlab = "Relative importance")


(gg <- xgb.ggplot.importance(importance_matrix[1:15], measure = "Frequency", rel_to_first = TRUE))
gg + ggplot2::ylab("Frequency")
topPredictors <- cat(toString(importance_matrix[1:15,1]))
#pred <- predict(bst, as.matrix(testSet[,topPredictors]), outputmargin=TRUE)
pred <- predict(bst_final, as.matrix(testSet[,c("shippingT", "item_condition_id", "len_words", "cat_2Shoes", "cat_2Women.s.Handbags", "avg_len_words", "num_words", "cat_1Women", "cat_2Tops...Blouses", "cat_1Electronics", "cat_1Men", "cat_2Jewelry", "cat_2Athletic.Apparel", "cat_1Handmade", "cat_2Computers...Tablets")]), outputmargin=TRUE)
rmse(as.numeric(testSet[,outcomeName]), as.numeric(pred))
submission <- as.data.frame(cbind(train$id[(trainPortion+1):nrow(myTrainDmy)/2],exp(pred), pred))
colnames(submission) <- c("test_id","price", "log price")
####=========================================================
bst <- xgboost(data = as.matrix(trainSet[,predictors]),
               label = trainSet[,outcomeName],
               max.depth=8, nround=9, objective = "reg:linear", verbose=0)
pred <- predict(bst, as.matrix(testSet[,predictors]), outputmargin=TRUE)
rmse(as.numeric(testSet[,outcomeName]), as.numeric(pred))

submission <- as.data.frame(cbind(train$id[(trainPortion+1):nrow(myTrainDmy)/2],exp(pred), pred))
colnames(submission) <- c("test_id","price", "log price")
write.csv(submission,"sub1_6_9_1.csv")



