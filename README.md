## Introduction
As an undergraduate student with a packed schedule, cooking is both a creative outlet, a balancing act, and an opportunity to not have dining hall meals. When I cook, time is often of the essence, and the number of steps in a recipe can determine whether a meal is quick and satisfying or a time-consuming challenge. With this in mind, this report delves into how the complexity and duration of recipes might influence ratings of recipes, particularly for those who, like me, juggle academics, commitments, and the desire for homemade meals. In order to conduct this analysis, two datasets of recipes and ratings posted to [food.com](https://www.food.com/) since 2008 were used. The datasets were originally scraped and used by the authors of the paper, [Generating Personalized Recipes from Historical User Preferences](https://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19c.pdf). 


The `recipes` dataset consisted of 83,782 rows of unique recipes and 12 columns. The columns and their descriptions are below:

| Column         | Description                                                                                                                                                     |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`         | Recipe name                                                                                                                                                     |
| `id`           | Recipe ID                                                                                                                                                       |
| `minutes`      | Minutes to prepare the recipe                                                                                                                                   |
| `contributor_id`| User ID who submitted the recipe                                                                                                                               |
| `submitted`    | Date the recipe was submitted                                                                                                                                   |
| `tags`         | Food.com tags for the recipe                                                                                                                                    |
| `nutrition`    | Nutrition information in the form `[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]`; PDV stands for "percentage of daily value" |
| `n_steps`      | Number of steps in the recipe                                                                                                                                   |
| `steps`        | Text for recipe steps, in order                                                                                                                                 |
| `description`  | User-provided description of the recipe                                                                                                                         |
| `ingredients`  | List of ingredients used in the recipe                                                                                                                          |
| `n_ingredients`| Number of ingredients used in the recipe                                                                                                                        |


The `interactions` dataset consisted of 731,927 rows of ratings/reviews of recipes and 5 columns. The columns and their descriptions are as follows:

| Column     | Description                       |
|------------|-----------------------------------|
| `user_id`  | User ID                           |
| `recipe_id`| Recipe ID                         |
| `date`     | Date of interaction               |
| `rating`   | Rating given                      |
| `review`   | Review text                       |

But beyond the personal convenience mentioned above, an interesting question arises from these datasets: **Does the complexity of a recipe—measured by the number of steps or ingredients—affect how it is rated by others?** This report explores whether the number of steps, preparation time, or ingredients in a recipe impacts its likelihood of receiving higher ratings. Are recipes with fewer steps more highly rated due to their simplicity, or do elaborate recipes earn higher reviews for their complexity? Is it possible to accurately predict the rating of a recipe based on these factors? 

Not all columns are relevant to this analysis, so the focus will be on key columns: `minutes`, `n_steps`, `n_ingredients`, and `rating`.

Now, without further ado, let’s dive into the analysis.


## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

The data cleaning process began by merging the two datasets and then proceeding with the necessary steps to prepare the data for analysis. The steps taken are as follows:

1. **Merging the Datasets**: The recipes and interactions datasets were left-merged on the `id` and `recipe_id` columns.
2. **Handling Missing Ratings**: In the merged dataset, all ratings of 0 were replaced with `np.nan`. Since the rating scale ranges from 1 to 5, a rating of 0 indicates missing data. This replacement is crucial to avoid bias in the analysis.
3. **Calculating Average Ratings**: A new column, `avg_rating`, was created to store the average rating for each unique recipe. This step ensures a comprehensive understanding of the ratings, as some recipes have multiple ratings.
4. **Converting List Columns**: Some columns, such as `nutrition`, were stored as strings, making list operations impossible. These columns were converted into lists for easier manipulation. 
5. **Breaking Down the `nutrition` Column**: The `nutrition` column, which contains multiple values, was split into individual columns for each nutritional component (e.g., `calories`, `protein`, `sugar`). This allows for more detailed analysis of the nutritional data.
6. **Converting Date Columns**: The `submitted` and `date` columns, which contain date information, were originally stored as strings. These were converted to datetime objects, enabling accurate analysis of temporal data.

After these data cleaning steps, the dataset now contains 234,429 rows and 28 columns. Below is the first 5 unique recipes of the cleaned DataFrame, with only the columns relevant to the analysis:

| name                                 |     id |   minutes | submitted           | tags_list                                                                                                                                                                                                                                                                                          |   n_steps |   n_ingredients |   avg_rating |   calories |   protein |
|:-------------------------------------|-------:|----------:|:--------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|----------------:|-------------:|-----------:|----------:|
| 1 brownies in the world    best ever | 333281 |        40 | 2008-10-27 00:00:00 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings']                                                                        |        10 |               9 |            4 |      138.4 |         3 |
| 1 in canada chocolate chip cookies   | 453467 |        45 | 2011-04-11 00:00:00 | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                                                                                                      |        12 |              11 |            5 |      595.1 |        13 |
| 412 broccoli casserole               | 306168 |        40 | 2008-05-30 00:00:00 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                                                                                               |         6 |               9 |            5 |      194.8 |        22 |
| millionaire pound cake               | 286009 |       120 | 2008-02-12 00:00:00 | ['time-to-make', 'course', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'american', 'southern-united-states', 'dinner-party', 'holiday-event', 'cakes', 'dietary', 'christmas', 'thanksgiving', 'low-sodium', 'low-in-something', 'taste-mood', 'sweet', '4-hours-or-less'] |         7 |               7 |            5 |      878.3 |        20 |
| 2000 meatloaf                        | 475785 |        90 | 2012-03-06 00:00:00 | ['time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'potatoes', 'vegetables', '4-hours-or-less', 'meatloaf', 'simply-potatoes2']                                                                                                                                             |        17 |              13 |            5 |      267   |        29 |

For additional context, the table below outlines the data types for each column displayed above:

| Column        | Type                                     |
|:--------------|:-----------------------------------------|
| name          | nominal categorical                      |
| id            | nominal categorical                      |
| minutes       | numerical discrete                       |
| submitted     | numerical discrete                       |
| tags          | nominal categorical                      |
| n_steps       | numerical discrete                       |
| n_ingredients | numerical discrete                       |
| avg_rating    | numerical continuous/ordinal categorical |
| calories      | numerical continuous                     |
| protein       | numerical continuous                     |


### Univariate Analysis

The plot below visualizes the distribution of `n_steps` in the recipes, providing insight into the typical number of steps involved. We can see that the distribution is heavily right-skewed. The median number of steps appears to be close to 7, which suggests that most recipes are relatively simple, with a smaller spread towards recipes with more complexity/steps.

<iframe
  src="assets/univariate-n_steps-distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Now, if we visualize the distribution of avg_rating we can gain valuable insight into the overall rating trends. The graphs displays a pronounced left skew, with most average ratings clustering around 5. This tells us that the majority of recipes in the dataset are highly rated (on average). Additionally, we can see that a significant portion of the data is concentrated near the upper end of the x-axis, with average ratings between 4 and 5. This further reinforces the dataset’s overall favorable ratings.

<iframe
  src="assets/univariate-avg_rating-distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

With these two plots, we are starting to address the main question posed in the introduction. The distribution of `n_steps` suggests that simplicity is common in the recipes, while the distribution of `avg_rating` shows that these simpler recipes tend to receive high ratings. Together, these insights could imply that complexity may not be the primary driver of high ratings. People might appreciate and rate recipes higher that are more accessible and straightforward. This can hint that the fact the recipes with fewer steps can perform just as well, if not better, than more complex recipes. 


### Bivariate Analysis

To futher explore the relationship between `n_steps` and `avg_rating`, they are plotted together. In order to create this plot, the recipes were grouped into bins based on the `n_steps`. The plot shows us that most recipes, regardless of the number of steps, seems to have high average ratings. This supports our univariate analysis findings. We can see this by looking at the interquartile ranges (IQR) of each group. Interestingly, recipes with a higher number of steps do not show a significant drop in their average ratings, suggesting that the number of steps may not strongly affect the overall rating. 

<iframe
  src="assets/bivariate-n_steps-boxplot.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Next, let's look at another relationship, this time between the average rating of a recipe and the time it takes to make (`minutes`). A similar analysis was performed but here recipes were grouped by `minutes`. This plot shows that most recipes regardless of the `minutes` also seem to have high average ratings. This makes sense due to the large clustering we saw in our univariate analysis. However, when we look at the IQR for each group, we see the median graduadlly decreasing as the time increases. This suggests that recipes with higher cooking time might receive lower ratings. 
<iframe
  src="assets/bivariate-minutes-boxplot.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

With this bivariate analysis, we can futher answer our main question.


### Interesting Aggregates
Embed at least one grouped table or pivot table in your website and explain its significance.

Here, we can further examine the relationship between `n_steps` and key recipe characteristics, such as `avg_rating`, `n_ingredients`, and `minutes`. The table and plot below summarizes the mean `avg_rating`, `n_ingredients`, and `minutes` for each `n_steps` group. We can see how the variables change as the complexity of the recipe increases. If we zoom in on the `n_ingredients` column we can see that there is an increase in the avgerage number of ingreidents as the number of steps increase. We also see this pattern in the `minutes` column. However, if we look at the avg_rating column, it is a bit diffult to make the same conclusion as we see a lot more fluctuation in both the table and the plot as the number of steps increases.


| n_steps | avg_rating | n_ingredients | minutes |
|--------:|------------:|--------------:|--------:|
|       1 |        4.72 |          5.73 |   22.20 |
|       2 |        4.72 |          5.96 |   38.42 |
|       3 |        4.71 |          6.53 |   46.47 |
|       4 |        4.70 |          7.07 |   72.68 |
|       5 |        4.66 |          7.53 |   94.82 |
|     ... |         ... |           ... |     ... |
|      87 |        5.00 |          9.00 |  195.00 |
|      88 |        4.20 |          9.50 | 2610.00 |
|      93 |        5.00 |         13.00 |  360.00 |
|      98 |        5.00 |         18.00 | 2930.00 |
|     100 |        5.00 |         19.00 | 1680.00 |

<iframe
  src="assets/interesting-aggregate-n_steps-lineplot
.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

To look at the fluction examined above even deeper, we can look at the mean, median, maximum, and minimum values for each grouping. This highlights the variability within each group. We can observe in the plot that while the majority of recipes maintain high ratings, there are some lower ratings in more complex recipes (e.g., `n_steps`: 88), which gives us some insight into the ratings of recipes.

| n_steps | Mean | Median | Max  | Min  |
|--------:|-----:|-------:|-----:|-----:|
|       1 | 4.72 |   5.00 | 5.00 | 1.00 |
|       2 | 4.72 |   4.92 | 5.00 | 1.00 |
|       3 | 4.71 |   4.90 | 5.00 | 1.00 |
|       4 | 4.70 |   4.88 | 5.00 | 1.00 |
|       5 | 4.66 |   4.83 | 5.00 | 1.00 |
|     ... |  ... |    ... |  ... |  ... |
|      87 | 5.00 |   5.00 | 5.00 | 5.00 |
|      88 | 4.20 |   4.33 | 4.33 | 3.00 |
|      93 | 5.00 |   5.00 | 5.00 | 5.00 |
|      98 | 5.00 |   5.00 | 5.00 | 5.00 |
|     100 | 5.00 |   5.00 | 5.00 | 5.00 |

<iframe
  src="assets/interesting-aggregate-n_steps-avg_rating-lineplot.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

For the line plots above, feel free to interact (double click) with the legend to view each line clearly.

## Assessment of Missingness

### NMAR Analysis
State whether you believe there is a column in your dataset that is NMAR. Explain your reasoning and any additional data you might want to obtain that could explain the missingness (thereby making it MAR). Make sure to explicitly use the term “NMAR.”


### Missingness Dependency
Present and interpret the results of your missingness permutation tests with respect to your data and question. Embed a plotly plot related to your missingness exploration; ideas include:
• The distribution of column Y when column X is missing and the distribution of column Y when column X is not missing, as was done in Lecture 8.
• The empirical distribution of the test statistic used in one of your permutation tests, along with the observed statistic.

<iframe
  src="assets/missingness-n_steps_reject_null.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

## Hypothesis Testing
Clearly state your null and alternative hypotheses, your choice of test statistic and significance level, the resulting 
p
-value, and your conclusion. Justify why these choices are good choices for answering the question you are trying to answer.

Optional: Embed a visualization related to your hypothesis test in your website.

Tip: When making writing your conclusions to the statistical tests in this project, never use language that implies an absolute conclusion; since we are performing statistical tests and not randomized controlled trials, we cannot prove that either hypothesis is 100% true or false.


## Framing a Prediction Problem
Clearly state your prediction problem and type (classification or regression). If you are building a classifier, make sure to state whether you are performing binary classification or multiclass classification. Report the response variable (i.e. the variable you are predicting) and why you chose it, the metric you are using to evaluate your model and why you chose it over other suitable metrics (e.g. accuracy vs. F1-score).

Note: Make sure to justify what information you would know at the “time of prediction” and to only train your model using those features. For instance, if we wanted to predict your final exam grade, we couldn’t use your Final Project grade, because the project is only due after the final exam! Feel free to ask questions if you’re not sure.


## Baseline Model
Describe your model and state the features in your model, including how many are quantitative, ordinal, and nominal, and how you performed any necessary encodings. Report the performance of your model and whether or not you believe your current model is “good” and why.

Tip: Make sure to hit all of the points above: many projects in the past have lost points for not doing so.


## Final Model
State the features you added and why they are good for the data and prediction task. Note that you can’t simply state “these features improved my accuracy”, since you’d need to choose these features and fit a model before noticing that – instead, talk about why you believe these features improved your model’s performance from the perspective of the data generating process.

Describe the modeling algorithm you chose, the hyperparameters that ended up performing the best, and the method you used to select hyperparameters and your overall model. Describe how your Final Model’s performance is an improvement over your Baseline Model’s performance.

Optional: Include a visualization that describes your model’s performance, e.g. a confusion matrix, if applicable.


## Fairness Analysis
Clearly state your choice of Group X and Group Y, your evaluation metric, your null and alternative hypotheses, your choice of test statistic and significance level, the resulting 
p
-value, and your conclusion.

Optional: Embed a visualization related to your permutation test in your website.
