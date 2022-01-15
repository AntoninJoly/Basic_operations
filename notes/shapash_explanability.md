# Info 📌
https://www.reddit.com/r/MachineLearning/comments/kxrld8/p_introducing_shapash_a_new_python_library_makes/

# Post header 📝
Introducing Shapash, a new Python library : makes Machine Learning models transparent and understandable by everyone  
Key features:
-Provides easy-to-read visualizations and webApp for Global and Local explainability
-Displays results with appropriate wording (preprocessing inverse/postprocessing)
-Summarizes local explanability to answer operational needs
-Uses explanability from Exploration to Production

Is compatible with many Python lib: Explaining (Shap/Lime), ML models, encoding features
# Comments section 👂🏻
>So to be clear, this is essentially a wrapper around other explainability tools (with a particular focus on shapley values) that uses them to build a plotly dashboard for visualizing the outputs of those tools, yeah?

>There was an ICML 2020 paper basically saying we should stop using Shapley vals for explainability (in most cases): https://paperswithcode.com/paper/problems-with-shapley-value-based

>Other papers discussing issues with SHAP are Fooling Lime and SHAP and "How do I fool you?". The former discusses adversarial attacks on the SHAP algorithm, and the latter discusses adversarial "attacks" on humans who interpret the results of SHAP feature attributions.

# Notes ✍🏻
- Only for tabular data.
- Not so good reception from practitionners as shapley values used in this framework fail to achieve their goal because they have bias and pushed forward desired source of problem.
- Botting / boosted accounts

# Thoughts 💭
- As model explanability is not mature yet, skip for the moment.