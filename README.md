# Starbucks-customer-segmentation
Medium blog post linke is [here](https://medium.com/p/8a81fea1f5c2/edit).

## Data overview

* The program used to create the data simulates how people make purchasing decisions and how those decisions are influenced by promotional offers.
* ach person in the simulation has some hidden traits that influence their purchasing patterns and are associated with their observable traits. People produce various events, including receiving offers, opening offers, and making purchases.
* As a simplification, there are no explicit products to track. Only the amounts of each transaction or offer are recorded.
* There are three types of offers that can be sent: buy-one-get-one (BOGO), discount, and informational. In a BOGO offer, a user needs to spend a certain amount to get a reward equal to that threshold amount. In a discount, a user gains a reward equal to a fraction of the amount spent. In an informational offer, there is no reward, but neither is there a requisite amount that the user is expected to spend. Offers can be delivered via multiple channels.
* The basic task is to use the data to identify which groups of people are most responsive to each type of offer, and how best to present each type of offer.

## Data Dictionary
### profile.json
Rewards program users (17000 users x 5 fields)

* gender: (categorical) M, F, O, or null
* age: (numeric) missing value encoded as 118
* id: (string/hash)
* became_member_on: (date) format YYYYMMDD
* income: (numeric)

### portfolio.json
Offers sent during 30-day test period (10 offers x 6 fields)

* reward: (numeric) money awarded for the amount spent
* channels: (list) web, email, mobile, social
* difficulty: (numeric) money required to be spent to receive reward
* duration: (numeric) time for offer to be open, in days
* offer_type: (string) bogo, discount, informational
* id: (string/hash)

### transcript.json
Event log (306648 events x 4 fields)

* person: (string/hash)
* event: (string) offer received, offer viewed, transaction, offer completed
* value: (dictionary) different values depending on event type
  * offer id: (string/hash) not associated with any "transaction"
  * amount: (numeric) money spent in "transaction"
  * reward: (numeric) money gained from "offer completed"
* time: (numeric) hours after start of test


## Data cleaning process
Cleaned data is stored in data directory as clean_data.csv.

The granular data is messy, and are stored in several different tables. The aims of the data cleaning process are
1. remove missing data and outliers
2. one-hot-encode categorical variables
3. merge tables based on customer ids
4. create labels based on activities

### remove missing data and outliers
We can see that many customers did not disclose age or income, which are not informative enough for our modelling purpose. Therefore, they are removed so that the rest of the customers are more comparable and we can extract as much information as possible from the remaining customers.

Outliers such as extremely large age numbers are removed. Some of these might be due to some pre-processing work done before to create the data tables. Previously they might be missing values, but they are replaced by special number to indicate that they are missing. However, as suggested above, they might not contain enough information, and we want our model to be predictive based on the data we have, so they are removed as well.

### OHE categorical variables
As suggested on [this post](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/), categorical variables might be troublesome to handle. And for categorical variables with relatively low cardinality, one hot encoding can help the model understand better what's going on. For this project, OHE has been applied to gender, age group of the customer, and channel of the offer.

### merge tables based on customer ids
Using a particular customer id as a reference, we can find all activities associated with that customer in the transcript. Merging the tables gives us a clearer picture of the progress made by a particular customer in the period of an offer.

### Create labels based on activities
The label indicate whether an offer has successfully resulted in a transaction.

## Modelling
The model is stored under the models directory.

The data was fit using naive classifier, a logistic regression, and a random forrest classifier. In the end, the random forrest gives the best performance at 0.729 accuracy and 0.721 F1-score on the testing set. The top 3 features are
1. reward
2. difficulty
3. duration
This makes sense, because the most important features affecting the success rate of an offer is its own mechanism.

The top 3 customer related features are
1. income
2. time with Starbucks
3. gender
This corresponds to the analysis performed because a customer's income is crucial in determining whether the customer is likely to convert the offer to a transaction. Also, the newest customers (joined in 2018) are the most responsive ones. This might be due to the fact that they are more excited about converting an offer because they have just recently joind the membership programme. The 3rd most important feature is gender, which corresponds to the part of the analysis where we find that offers are likely to have a higher success rate on female as compared to male customers.

## Conclusion
The outcome of an offer is tightly related to the internal mechanism of the offer itself - its duration, difficulty to achieve, and the reward. Putting ourselves into a customer's shoes, we can see that for offers with different duration, difficulty and reward, we might also tend to have different willingness to convert the offer to a transaction.

The trained model has good performance on predicting the success rate of an offer on a particular customer. Using the model we can see which customer might be more responsive to a particular offer. Using the model scores (predicted likelihoods), we can also rank order the customers to determine the groups of customers to advertised to. For example, if Starbucks aims to advertise an offer to 70% of its customer base with a membership, Starbucks can choose the customers based on the rank ordering by choosing the top 70% customers with highest likelihood to convert the offer. In this way, Starbucks can maximise its revenue associated with that particular offer using this method.
