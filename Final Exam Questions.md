(MRQ) Which of the following statements about Linear Models are TRUE?

Group of answer choices

Linear Regression uses the identity function with the mean squared error loss

Linear Regression uses the identity function with the cross-entropy loss

Logistic Regression uses the sigmoid function with the mean squared error loss

None of the other

Logistic Regression uses the sigmoid function with the cross entropy loss



(MRQ) Which of the following statements about Linear Regression are always TRUE?

Group of answer choices

It is possible for a dataset containing samples with a lot of noise to get an MSE loss of 0

Feature scaling (e.g., z-score standardization) will improve the prediction results

None of the other

We find the best Linear Regression model for a dataset of the MSE loss is 0

To train a Linear Regression model, the data must show a linear relationship


(MRQ) Assume two different regression datasets D1 and D2 with the same number of training samples and the same number of features. After training a Linear Regression model over each dataset, the two results sets of parameters 
are almost identical for both models. Which conclusions can you derive from this information?

Group of answer choices

Both models will show very similar MSE losses over their training data (D1 or D2)

Both models will make very similar predictions for unseen data samples

Both models will show a very similar MSE loss over the same test dataset

None of the other

The datasets D1 and D2 have very similar distributions



(MRQ) Recall that we introduced regularization as an additional term to a loss function (e.g., MSE or Cross-Entropy). This additional term included the regularization parameter λ, which is a hyperparameter that specifies the strength of the regularization. Which of the following statements about the value of λ are TRUE?

Group of answer choices

We should pick the λ value that minimizes the validation error

Increasing λ may cause more training examples to be classified incorrectly

Increasing λ may cause the model to overfit

We should pick the λ value that minimizes the test error

None of the other



(MRQ) Which of the following statements about Gradient Descent are always TRUE (independent from the exact model and loss function)

Group of answer choices

Gradient Descent will converge to the same solution no matter the initial choice of 𝜃

The update direction in Gradient Descent is determined by the negative gradient of the loss function

The loss will decrease in each iteration

Gradient Descent will eventually alway find the global minimum

None of the other



(MRQ) Which of the following factors directly affect the PageRank score of a webpage W?

Group of answer choices

The total number of incoming links to W

The PageRank scores of pages linking to W

The total number of outgoing links on pages linking to W

The total number of outgoing links from W to other pages

None of the other



(MCQ) Given is the directed and unweighted graph below with Nodes A-E. You are also given the PageRank scores for Nodes A and C (you can ignore where this two values are coming from)

cs5228-2410-quiz-pagerank-1.png

Assuming 
, what will be the PageRank score of Node B? Hint: You should first calculate the PageRank scores for Nodes D and E.

Group of answer choices

9

11

10

None of the other

12



(MRQ) Assume you have a directed and unweighted Graph G(V,E). After applying the PageRank algorithm to G, all the nodes have the same PageRank score. What does this information tell you about G?

Group of answer choices

The set of edges E must be empty

G must have multiple components

None of the other

The PageRank algorithm must have been run with alpha=0

G must be a fully connected graph (i.e., E contains all possible directed edges)



(MRQ) Which of the following statements about Principal Component Analysis (PCA) are TRUE?

Group of answer choices

PCA can only be applicable for unlabeled datasets

PCA seeks to maximize the variance of the data in the transformed feature space

None of the other

Feature scaling (e.g., z-score standardization) will not affect the ouput of PCA

PCA creates new features which are linear combinations of the original features



(MRQ) Which of the following statements about T-distributed Stochastic Neighbor Embedding (t-SNE) are TRUE?

Group of answer choices

t-SNE assumes labeled datasets

Running t-SNE twice over the same dataset may yield different results

None of the other

t-SNE allows to map any new data point into the lower-dimensional space

t-SNE does not assume linear data distributions



(MCQ) Assume you are using Reservoir Sampling with a reservoir size B=20. After seeing the first 100 items 
, what is the probability of an item 
 NOT being in the sample?

Group of answer choices

0.4

None of the other

0.2

0.8

0.6



(MCQ) Assume you have a Bloom Filter with a bit array B. This also means that your memory requirements are |B| bits to store the bit array (1 bit per bucket). Now you want to convert your Bloom Filter to a Counting Bloom Filter, where each bucket should be able to hold a count of up to 10. How does this affect your memory requirements (compared to the initial (basic) Bloom Filter)?

Group of answer choices

10x more memory needed

3x more memory needed

4x more memory needed

2x more memory needed

None of the other



(MRQ) Assume you have a Bloom Filter with a bit array of size B. After inserting all keys of set S using m hash functions, you see a false positive rate of 8%. What steps can you take that are guaranteed to lower the false positive rate?

Group of answer choices

Decrease the size of the bit array B

None of the other

Increase the number of hash functions

Decrease the number of hash functions

Increase the size of the bit array B




MRQ) What are impossible values for a single estimate (i.e., using only a single hash function) of the Flajolet-Martin algorithm for counting distinct items?



(MRQ) Assume you have dataset with 50 unique data samples, and each data sample represented by 500 feature values. Given this information, which of the following conclusions can you make when training a basic Linear Regression model? (Note: in the answers below 
)

Group of answer choices

The model will have infinite solutions

The model with have a unique solution

B' is a matrix of size 50x50

B' is a matrix of size 500x500

None of the other
