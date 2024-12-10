# Recipe Ratings Unveiled: Predicting Success Through Complexity

Author: Norah Kerendian

## Introduction
As an undergraduate student with a packed schedule, cooking is both a creative outlet, a balancing act, and an opportunity to not have dining hall meals. When I cook, time is often of the essence, and the number of steps in a recipe can determine whether a meal is quick and satisfying or a time-consuming challenge. With this in mind, this report delves into how the complexity and duration of recipes might influence ratings of recipes, particularly for those who, like me, juggle academics, commitments, and the desire for homemade meals. In order to conduct this analysis, two datasets of recipes and ratings were posted to [food.com](https://www.food.com/) since 2008 were used. The datasets were originally scraped and used by the authors of the paper, [Generating Personalized Recipes from Historical User Preferences](https://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19c.pdf). 


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

But beyond the personal convenience mentioned above, an interesting question arises from these datasets: **Does the complexity of a recipe—measured by the number of steps or minutes—affect how it is rated by others?** This report explores whether the number of steps or the preparation time in a recipe impacts its likelihood of receiving higher ratings. Are recipes with fewer steps more highly rated due to their simplicity, or do elaborate recipes earn higher reviews for their complexity? Is it possible to accurately predict the rating of a recipe based on these factors? 

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

After these data-cleaning steps, the dataset now contains 234,429 rows and 28 columns. Below are the first 5 unique recipes of the cleaned DataFrame, with only the columns relevant to the analysis:

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
  width="620"
  height="450"
  frameborder="0"
></iframe>

Now, if we visualize the distribution of avg_rating we can gain valuable insight into the overall rating trends. The graph displays a pronounced left skew, with most average ratings clustering around 5. This tells us that the majority of recipes in the dataset are highly rated (on average). Additionally, we can see that a significant portion of the data is concentrated near the upper end of the x-axis, with average ratings between 4 and 5. This further reinforces the dataset’s overall favorable ratings.

<iframe
  src="assets/univariate-avg_rating-distribution.html"
  width="620"
  height="450"
  frameborder="0"
></iframe>

With these two plots, we are starting to address the main question posed in the introduction. The distribution of `n_steps` suggests that simplicity is common in the recipes, while the distribution of `avg_rating` shows that these simpler recipes tend to receive high ratings. Together, these insights could imply that complexity may not be the primary driver of high ratings. People might appreciate and rate recipes higher that are more accessible and straightforward. This can hint that the fact the recipes with fewer steps can perform just as well, if not better, than more complex recipes. 


### Bivariate Analysis

To further explore the relationship between `n_steps` and `avg_rating`, they are plotted together. In order to create this plot, the recipes were grouped into bins based on the `n_steps`. The plot shows us that most recipes, regardless of the number of steps, seem to have high average ratings. This supports our univariate analysis findings. We can see this by looking at the interquartile ranges (IQR) of each group. Interestingly, recipes with a higher number of steps do not show a significant drop in their average ratings, suggesting that the number of steps may not strongly affect the overall rating. 

<iframe
  src="assets/bivariate-n_steps-boxplot.html"
  width="620"
  height="450"
  frameborder="0"
></iframe>

Next, let's look at another relationship, this time between the average rating of a recipe and the time it takes to make (`minutes`). A similar analysis was performed but here recipes were grouped by `minutes`. This plot shows that most recipes regardless of the `minutes` also seem to have high average ratings. This makes sense due to the large clustering we saw in our univariate analysis. However, when we look at the IQR for each group, we see the median gradually decreasing as time increases. This suggests that recipes with higher cooking time might receive lower ratings. 

<iframe
  src="assets/bivariate-minutes-boxplot.html"
  width="620"
  height="450"
  frameborder="0"
></iframe>

With this bivariate analysis, we can further answer our main question.


### Interesting Aggregates

Here, we can further examine the relationship between `n_steps` and key recipe characteristics, such as `avg_rating`, `n_ingredients`, and `minutes`. The table and plot below summarize the mean `avg_rating`, `n_ingredients`, and `minutes` for each `n_steps` group. We can see how the variables change as the complexity of the recipe increases. If we zoom in on the `n_ingredients` column we can see that there is an increase in the average number of ingredients as the number of steps increase. We also see this pattern in the `minutes` column. However, if we look at the avg_rating column, it is a bit difficult to make the same conclusion as we see a lot more fluctuation in both the table and the plot as the number of steps increases.


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
  width="620"
  height="450"
  frameborder="0"
></iframe>

To look at the fluctuation examined above even deeper, we can look at the mean, median, maximum, and minimum values for each grouping. This highlights the variability within each group. We can observe in the plot that while the majority of recipes maintain high ratings, there are some lower ratings in more complex recipes (e.g., `n_steps`: 88), which gives us some insight into the ratings of recipes.

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
  width="620"
  height="450"
  frameborder="0"
></iframe>

For the line plots above, feel free to interact (double-click) with the legend to view each line clearly.

## Assessment of Missingness

### NMAR Analysis

Looking at the dataset, there are three columns that have missing values: `description`, `rating` (and `avg_rating`), and `review`. Due to this, it will be meaningful to assess the missingness of the columns. Looking at the missing columns, there is reason to believe that the column `review` is considered Missing Not At Random (NMAR). The `review` column contains written reviews for recipes that could be missing due to various factors, one being an emotional one. For example, someone may only leave a review if the recipe was either a disaster or exceptional, while moderate experiences might be underrepresented. For instance, a failed recipe might prompt a negative review due to frustration, whereas a successful but unremarkable experience may result in no review at all. An additional piece of data that could change the classification of missingness from NMAR to MAR (Missing At Random), is if we had a column that indicated if the recipe succeeded, failed, or had moderate results. This added variable could help predict which reviews could be missing depending on the success rate of the attempted recipe. 

### Missingness Dependency

Now, let's assess the missingness of the `rating` column to determine if it is dependent on another column. 

> Rating and Number of Steps

To start, there is reason to believe that ratings might be missing depending on the number of steps in a recipe. Intuitively, if a recipe has a moderate number of steps and is fairly straightforward, then people might not leave a review because the recipe met expectations without eliciting a strong reaction. To determine if missing ratings are dependent on the number of steps, a permutation test can be performed. 

**Null Hypothesis**: The missingness of the `rating` column does not depend on the number of steps (`n_steps`) in the recipe.

**Alternative Hypothesis**: The missingness of the `rating` column does depend on the number of steps (`n_steps`) in the recipe.

**Test Statistic**: Absolute Difference in Means and Kolmogorov-Smirnov (two separate tests)

**Significance Level**: 0.05

<iframe
  src="assets/missingness-n_steps_reject_null.html"
  width="620"
  height="450"
  frameborder="0"
></iframe>

<iframe
  src="assets/missingness-n_steps-kde_map-reject-null.html"
  width="620"
  height="450"
  frameborder="0"
></iframe>

Both the Kolmogorov-Smirnov (KS) test and the Absolute Difference in Means test yielded p-values of **0.0** which is less than the stated significance level of 0.05. Thus, I reject the null hypothesis which concludes that the missingness of the rating column does depend on the number of steps.

> Rating and Minutes

Now the question arises: Is there a column that the `rating` column is not dependent on? To answer this question, let's run another permutation test. This permutation test asks whether the rating column's missingness depends on the `minutes` column.

**Null Hypothesis**: The missingness of the `rating` column does not depend on the number of minutes (`minutes`) in the recipe.

**Alternative Hypothesis**: The missingness of the `rating` column does depend on the number of minutes (`minutes`) in the recipe.

**Test Statistic**: Absolute Difference in Means

**Significance Level**: 0.05

<iframe
  src="assets/missingness-n_steps_fail-to-reject-null.html"
  width="620"
  height="450"
  frameborder="0"
></iframe>

The Absolute Difference in Means test produced a p-value of **0.125**, which is greater than the significance level of 0.05. Thus, I fail to reject the null hypothesis, which suggests that the missingness of the rating column does not depend on the number of minutes in the recipe.

## Hypothesis Testing

To further explore the main question-*Does the complexity of a recipe—measured by the number of steps or minutes—affect how it is rated by others?*-it would be worth while to see the relationship between the number of steps and the rating. To investigate this, it will be useful to run a permutation test. A permutation test is preferred over a standard hypothesis test because the main goal is to determine whether the two groups-recipes with `n_steps` less than or equal to the median number of steps and recipes with `n_steps` greater than the median number of steps-look like they come from the same population. 

There is reason to hypothesize that recipes with more steps are rated lower on average. This could be due to their complexity intimidating or frustrating users, leading to lower ratings, or because such recipes are prone to execution errors, resulting in unsatisfactory outcomes. To test this, the test statistic of the difference of means between the two groups was calculated. This is appropriate because the alternative hypothesis has a certain direction posits a specific direction: recipes with more steps tend to receive lower ratings.

During this test, a column named `lower_median_steps` was created which consists of boolean values to help create the two groups. This column will be used in other sections of this report.

To investigate this, the following was conducted:

**Null Hypothesis**: The number of steps in a recipe does not affect the recipe's average rating.

**Alternative Hypothesis**: Recipes with more steps are rated lower on average.

**Test Statistic**: Difference in Means between two groups: recipes with `n_steps` less than or equal to the median and recipes with `n_steps` greater than the median.

**Significance Level**: 0.05

<iframe
  src="assets/hypothesis-testing-n_steps.html"
  width="620"
  height="450"
  frameborder="0"
></iframe>

The permutation test produced a p-value of **0.045**, which is less than the significance level of 0.05. Thus, I reject the null hypothesis, which suggests that there is sufficient evidence to support the alternative hypothesis: recipes with more steps are rated lower on average. This adds some insight to the relationship between the number of steps of a recipe and its rating. 

## Framing a Prediction Problem

For the remainder of this report, I will address a prediction problem: predicting recipe ratings on a scale of 1 to 5. This is a **multiclass classification** problem because the ratings are discrete and fall into one of five categories (1, 2, 3, 4, or 5). I will use the average rating as the response variable since it provides a meaningful measure of recipe quality based on user feedback. However, the current `avg_rating` column is a continuous variable so I will use a rounded version instead. Predicting recipe ratings is valuable because it will help add insight to the main question of the report. To predict ratings, I will use features available at the time of prediction, such as the number of steps or minutes in a recipe, as there is evidence of correlation between these features and the response variable from previous analyses.

To evaluate my model, I will use **F1 score** over accuracy because the dataset has an imbalanced distribution of ratings, with a higher concentration in higher ratings. Since the F1 score balances precision and recall, it is better suited for imbalanced datasets like the one we have. The F1 score will provide a more nuanced understanding of model performance compared to accuracy alone.

I plan to create a baseline model using two features, `minutes` and `n_steps`, to predict ratings. From there, I will enhance the model by incorporating additional features and advanced techniques.

## Baseline Model

The first step in tackling the prediction problem is creating a baseline model to later improve upon. The baseline model in this section predicts the rounded average ratings of recipes based on two features: the number of steps (`n_steps`) and the cooking time (`minutes`). As previously mentioned, the original average ratings column contained continuous variables, but to approach the prediction problem, they were converted to categorical variables through rounding. The data was further cleaned in this section by dropping all the rows with missing values in the target column of rounded average ratings. I made the decision to drop these rows rather than impute them as they made up less than 1% of the data, and imputation would introduce randomness, which I am trying to minimize. 

For the baseline model, two main features were used:

- `n_steps`: A quantitative feature representing the number of steps in the recipe.
- `minutes`: A quantitative feature representing the cooking time in minutes.

Both of the features used are numerical quantitiative variables. There were no ordinal or nominal features included in the baseline model. The target column, as describe earlier, is now a categorical ordinal variable. 

Regarding column transformation, `minutes` was standardized using `StandardScalar` since the column contained many outliers. Standardizing it, although it does not drastically affect the results of the model, will help bring the column into a comparable range. The `n_steps` column was not transformed for this model. 

As for the actual model, I decided to use a Random Forest Classifier. The Random Forest Classifier was chosen for its strengths in handling numerical and categorical variables (which will be implemented in the final model), its robustness to overfitting (which can be further advanced by tuning its hyperparameters), and its ability to capture non-linear relationships in the data which is useful for predicting ratings.

The model was then trained on the training data and evaluated on the test data using the F1 score. As previously mentioned, the F1 score was chosen due to the skewness of the higher ratings as seen below:

|   avg_rating_rounded |   count |
|---------------------:|--------:|
|                    5 |  171448 |
|                    4 |   53696 |
|                    3 |    4601 |
|                    2 |    1207 |
|                    1 |     700 |

The F1 score for the baseline model was **0.67**, which indicates moderate performance in predicting the ratings. To further understand the F1 score for each category of rating, here are the individual scores: 

- Rating 1: F1 score = 0.01
- Rating 2: F1 score = 0.10
- Rating 3: F1 score = 0.09
- Rating 4: F1 score = 0.18
- Rating 5: F1 score = 0.86

As we can see, the baseline model does not perform well for lower ratings (1, 2, and 3) but it performs very well for higher ratings, specifically ratings of 5. This suggests that the model is good at make predictions for higher ratings but less effective for lower ratings. This will be important to keep in mind when feature engineering for the final model. 

In conclusion, while this model performs decently for higher ratings, there is lots of room for improvement, especially when focusing on the lower ratings. I believe that the current model is moderate and could definitely be improved, particularly given the very low scores for the lower ratings. It is important for the model to make accurate predictions for lower ratings as well, as it will help individuals searching for recipes avoid the "bad" ones. 


## Final Model

In the final model, I decided to focus on the following columns as features: `minutes`, `submitted`, `lower_median_steps`, `calories`, `protein`, and `tags_list`. Below, I will discuss the feature engineering applied and why each column is relevant for improving my baseline model. 

`minutes`

The `minutes` column was selected due to its observed relationship with the `avg_rating` column. Specifically, during the bivariate analysis, it was noticed that median ratings gradually decreased as minutes increased. This trend suggests that recipes with longer cooking time might receive lower ratings. With this, using the minutes column would be beneficial for predicting both lower and higher ratings. In the baseline model, this column was included and standardized using `StandardScalar`. To improve upon this, I applied a different yet similar transformer to `minutes`, `RobustScaler()`. This will normalize the variability of the `minutes` column while also mitigating the influence of the extreme outliers we noticed in the baseline model. This adjustment should improve my model's performance as the relationship between minutes and ratings is an informative feature for predicting outcomes.

`submitted`

The `submitted` column was originally selected due to speculation that ratings might vary by the year or season in which a recipe was submitted. To look into this, analysis was conducted which concluded that more recent submissions tend to have lower average ratings. This could be due to newer recipes having received less attention or reviews. This information is helpful in improving the model. So, I feature-engineered the `submitted` column. A function was created to extract the year of each value. This was incorporated into the pipeline using a `FunctionTransformer`, followed by `OneHotEncoder` to represent the year as categorical data. I believe this will help improve my model's performance as there seems to be a trend between the year of the `submitted` column and the ratings of recipes.

`lower_median_steps`

The `lower_median_steps` column was created during the hypothesis test conducted earlier. This column was originally created to aid in the permutation test of exploring whether recipes with more steps tend to have lower ratings. The test concluded that there is sufficient evidence to support this hypothesis, as recipes with more steps are rated lower on average. This means that using the `lower_median_steps` column could further improve my model. A `OneHotEncoder` was used to categorize the column and treat each category equally. Incorporating this feature should improve the model’s ability to predict ratings by leveraging the relationship between `n_steps` and ratings.

`calories` and `protein`

The `calories` and `protein` columns were selected based on speculation that recipes with higher protein might receive higher ratings. This was further inspected through a scatter plot of `protein` against `avg_rating_rounded`. The plot confirmed my speculation as higher protein recipes tended to have better ratings. To balance potential confounding due to serving size, I calculated the proportion of protein relative to `calories` for each recipe. A function was created to calculate the protein proportions and was applied using `FunctionTransformer`. This new feature will add another layer of information and depth to my final model as there seems to be a relationship between protein levels and recipe ratings.

`tags_list`

The `tags_list` column was included due to speculation that recipes with more tags might receive higher ratings. Tags function like hashtags on this website, potentially increasing visibility and engagement with certain recipes. This was further investigated using a bar plot that compared the number of tags with the average rating. My speculation was supported and the bar plot showed that recipes with longer tag lists tend to have slightly higher average ratings. To incorporate this, a function was created to extract the length of each list in `tags_list` and was used in my model's pipeline. I believe this information will help predict more accurate ratings, particularly on the higher end of the scale.

As for the actual model, I continued to use a `RandomForestClassifier`. To optimize performance I applied `GridSearchCV` to perform cross-validation and hyperparameter tuning. The hyperparameters used were `max_depth` and `n_estimators`. The best parameters found were the following:

- `max_depth`: 24
- `n_estimators`: 350

To evaluate my model, I continued to use the F1 score to best compare the improvement. The F1 score for the final model was **0.86**, which is a 0.19 increase from the baseline model. To further understand the F1 score for each category of rating, here are the individual scores: 

- Rating 1: F1 score = 0.19
- Rating 2: F1 score = 0.49
- Rating 3: F1 score = 0.50
- Rating 4: F1 score = 0.73
- Rating 5: F1 score = 0.92

Not only did the overall F1 score increase but each individual category also saw significant improvement. This can be further visualized the the Confusion Matrix below:

<iframe
  src="assets/final-model-confusion-matrix.html"
  width="900"
  height="720"
  frameborder="0"
></iframe>

In conclusion, the final model demonstrates significant improvement over the baseline model due to the addition of feature engineering and optimization of hyperparameters. These adjustments allowed the model to better capture the relationships in the data between various columns, leading to improved accuracy and predictive power.

## Fairness Analysis

In this analysis, let's explore whether the model exhibits bias when it comes to predicting ratings for two groups: 

- High Protein: Recipes with more than 18 grams of protein
- Low Protein: Recipes with less than or equal to 18 grams of protein

The threshold of 18 grams of protein represents the median amount of protein across all recipes. The median amount of protein was chosen over the mean because the data contains some outliers that could skew the mean. The box plot below highlights these outliers.

<iframe
  src="assets/fairness-analysis-protein-outliers.html"
  width="620"
  height="450"
  frameborder="0"
></iframe>

The evaluation metric used will be **precision parity** for the two groups. Precision is prioritized for this model compared to other metrics since correctly identifying the recipe rating among all instances of that recipe rating is more important. False positives can mislead individuals with inaccurate ratings. For an individual who is selecting recipes based on their protein levels, precision will make sure that when a model predicts a recipe as highly rated (4,5), the recipe will also be satisfactory for the person. This specific grouping could be very helpful for individuals who are concerned with their protein intake and rely on these predictions to make good choices about their diet. Having higher false positives (incorrect high ratings) could lead individuals to be disappointed. 

Focusing on precision parity helps determine whether the model is biased towards one group over the other when making predictions. 

**Null Hypothesis**: The model is fair. Its precision for recipes with higher protein levels and lower protein levels is roughly the same, and any differences are due to random chance.

**Alternative Hypothesis**: The model is unfair. Its precision for recipes with lower protein levels is lower than that for recipes with higher protein levels.

**Test Statistic**: Difference in Precision (Low Protein - High Protein)

**Significance Level**: 0.05

<iframe
  src="assets/fairness-analysis-protein.html"
  width="620"
  height="450"
  frameborder="0"
></iframe>

After 1,000 samples, the permutation test produced a p-value of **0.0**, which is less than the significance level of 0.05. Thus, I reject the null hypothesis that the model is fair. This suggests that the model’s precision for recipes with lower protein is lower than its precision for recipes with higher protein.

