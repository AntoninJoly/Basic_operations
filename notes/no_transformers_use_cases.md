# Info 📌
https://www.reddit.com/r/MachineLearning/comments/pxz1iw/d_nlp_or_computer_vision_tasks_where_transformers/

# Post header 📝
NLP or computer vision tasks where transformers are less recommended  
Do you know an application or task (NLP or Computer vision) where Transformer-based models (like BERT) do not perform well?

# Comments section 👂🏻
>I know that, for a lot of Social Media analysis stuff, using transformers comes with the cost of needing to re-train models to account for every new trend, so hand-engineered features end up being much easier to work with

>Small data will generally be dominated by non-Transformer models--i.e., anywhere where more explicitly hand-engineered priors (in the architecture) are going to be superior to the higher degrees of freedom Transformer offers. (This would also appear to hold on the CV side?)

>Your argument about small data is interesting - my observation is probably entirely the other way around; if you have a lot of data then you have many methods to choose from, you might go with all kinds of interesting approaches to learn from that data, including various non-neural-network models; but if you have really small data, then learning from that data alone is not an option no matter what architecture/priors you choose, and you absolutely need transfer learning from something pretrained on a very large corpus, which probably means transfomer-based models.

>However, in some cases, you cannot transfer learn, for whatever reason; in those scenarios, there are frequently non-Transformer architectures which are superior.

# Notes ✍🏻
None

# Thoughts 💭
- Mixed reviews.
- Transformers seem to be the solution as you have large number of data, while you will end doing transfer learning on traditionnal model (usually non transformers) on small dataset.