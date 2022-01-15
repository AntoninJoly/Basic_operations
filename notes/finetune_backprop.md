# Info 📌
https://www.reddit.com/r/MachineLearning/comments/mas3sz/p_backprop_a_library_to_easily_finetune_and_use/

# Post header 📝
Backprop: a library to easily finetune and use state-of-the-art models  
I'd like to share Backprop, a Python library I've been co-authoring for the last few months. Our goal is to make finetuning and using models as easy as possible, even without extensive ML experience.  
We've currently got support for text and image-based tasks, with wrappers around models like Google's T5, OpenAI's CLIP, and Facebook's BART, among others.  
Once you've got your training data, you can just import your model/task, and then finetune with a single line of code.

# Comments section 👂🏻
>We actually use Hugging Face's library under the hood for some models.
>One way we're trying to make ourselves distinct is in terms of accessibility/usability for people who may be less familiar with ML -- so, rather than setting up a model, tokenizer, training loop, etc., you can just pick a model and start training with your data.
>Additionally, we're aiming to be beyond just NLP. We've got some image-based tasks already, and are actively working to extend to other domains.

>Everything we use is PyTorch under the hood.

# Notes ✍🏻
None

# Thoughts 💭
- Worth giving it a try.
- Wonder about range of accepted models.