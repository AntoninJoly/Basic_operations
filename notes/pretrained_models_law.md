# Info 📌
https://www.reddit.com/r/MachineLearning/comments/id4394/d_is_it_legal_to_use_models_pretrained_on/

# Post header 📝
Is it legal to use models pre-trained on ImageNet for commercial purposes?  

# Comments section 👂🏻
> Japan explicitly made this universally legal a couple years ago, allegedly in hopes of attracting ML talent. So at least in theory, you can scrape random images off Google or whatever, train your model on them, and use that model commercially without any legal worries.

>From my experience at a FAANG we were not permitted to use imagenet models because there is a lack of known license for every image in The Training set. Lawyers err on the side of caution and assume the model is derivative. Hence we were advised to use models trained on datasets like OpenImages.

>OpenImages doesn't fix that. From their page:
>The images are listed as having a CC BY 2.0 license.
>Note: while we tried to identify images that are licensed under a Creative Commons Attribution license, we make no representations or warranties regarding the license status of each image and you should verify the license for each image yourself.
>Which is the exact same problem that you have with ImageNet.

>Well it would fall under the Apache 2 license in the GitHub repo so it would be usable for commercial use. Anything in a GitHub repo with a license means that everything in the repo (besides a few exceptions like sub modules) fall under the license in the repo.

# Notes ✍🏻
- From various github repo answers
Pytorch:
>The short answer is that licenses for the underlying data are generally pretty generous about this type of derivative use, but they occasionally vary. Our legal team have said that this is entirely a matter of your own or your legal team's comfort with the use you have in mind and the underlying license, so there really isn't a boxed answer here unfortunately. I'd encourage you to check with your legal teams to see what they're comfortable with.
Tensorflow:
>Short answer - Yes. The repo is listed under the Apache 2.0 license which allows for commercial use.
>Pytorch is on another permissive license, though it is different... The Apache 2.0 license is one of the most permissive licenses in the industry.

# Thoughts 💭
- Overall agreement but still gray zone as the repo are under free to use license (even commercially) but data part is unclear.