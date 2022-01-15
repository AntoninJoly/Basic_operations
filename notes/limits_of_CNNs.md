# Info 📌
https://www.reddit.com/r/MachineLearning/comments/pyti87/d_have_we_reached_some_limit_in_the_advancement/

# Post header 📝
Have we reached some limit in the advancement of CNNs?  
It seems that every modern CNN architecture is just some variant of ResNet or the residual block idea.  
With all the work that companies like Google or Microsoft put into research in CNN and ImageNet, have we reached some kind of stopping point with CNNs for the foreseeable future? Are Residual blocks the answer in a sense?

# Comments section 👂🏻
>We probably pushed extracting meaningful representations from a single label per image to a limit. Most future progress will probably come from self supervision, multi task/modal training, videos/simulation and much larger datasets.

>CNN architecture is adjusted to specific hardware, originated from GPU. Progress in hardware, like less strict parallelization or some changes in bit depth, or different type/size/latency of local memory coul potentially produce quite different architecture. Including dropping CNN paradigm partially or completely as already happens with transformer.

>Now, you asked if we reached "some" limits. Yes, many. When we can reach 90% top1 accuracy on ImageNet, we can't really improve to 100% because some labels are wrong anyway.

>I mean Imagenet is nice and all but it still has issues and it kinda grinds my gears when people talk about visual classification as pretty much "solved"

>Yes. 2016 was the last big advancement in CNN architecture. One of the biggest problems is ImageNet. New architectures are focused on minimal advances on a ImageNet and we have reached a point where we can’t improve anymore.
>There’s been a lot of focus on attention, but that didn’t improve performance on ImageNet by much so fell out of favour. Now many are looking at transformer architecture, but that again isn’t able to improve ImageNet performance yet. It seems the next big thing in CNNs will be efficient architectures.

# Notes ✍🏻
None

# Thoughts 💭
- Reach limit in CNN progress
- Data might be the problem are they are not perfeclty annotated.
- In order to make progress, new mechanism are proposed (attention, transformers,...)