# Info 📌
https://www.reddit.com/r/MachineLearning/comments/hnn1vv/p_gridsearchcv_20_up_to_10x_faster_than_sklearn/

# Post header 📝
GridSearchCV 2.0 - Up to 10x faster than sklearn - tune-sklearn  
I'm one of the developers that have been working on a package that enables faster hyperparameter tuning for machine learning models. We recognized that sklearn's GridSearchCV is too slow, especially for today's larger models and datasets, so we're introducing tune-sklearn. Just 1 line of code to superpower Grid/Random Search with
- Bayesian Optimization
- Early Stopping
- Distributed Execution using Ray Tune
- GPU support

https://medium.com/distributed-computing-with-ray/gridsearchcv-2-0-new-and-improved-ee56644cbabf
https://towardsdatascience.com/5x-faster-scikit-learn-parameter-tuning-in-5-lines-of-code-be6bdd21833c

# Comments section 👂🏻
>Scikit-optimize’s BayesOptSearch is very similar to our TuneSearchCV API.
>The core benefits of tune-sklearn are GPU support and early stopping which make us much better suited to integrate with deep learning scikit learn adapters such as KerasClassifier, Skorch, and XGBoost.

# Notes ✍🏻
- Cutting edge hyperparameter tuning techniques (bayesian optimization, early stopping, distributed execution) can provide significant speedups over grid search and random search.

# Thoughts 💭
- Complete hyperparameter optimization framework