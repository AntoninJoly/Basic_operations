# Info 📌
https://www.reddit.com/r/MachineLearning/comments/g7ntb8/r_background_matting_the_world_is_your_green/

# Post header 📝
Background Matting: The World is Your Green Screen
Link to git:  
https://github.com/senguptaumd/Background-Matting

# Comments section 👂🏻
>Our goal is really to provide professional quality results with any background, including backgrounds with movements and lighting changes. This is something which is still not really possible with traditional methods.

# Notes ✍🏻
- Although segmentation has made huge strides in recent years, it does not solve the full matting equation. Segmentation assigns a binary (0,1) label to each pixel in order to represent foreground and background instead of solving for a continuous alpha value.
-Therefore, the binary nature of segmentation creates a harsh boundary around the foreground, leaving visible artifacts. Solving for the partial transparency and foreground color allows much better compositing in the second frame.
- Although our method works with some background perturbations, it is still better when the background is constant and best in indoor settings. For example, it does not work in the presence of highly noticeable shadows cast by the subject, moving backgrounds (e.g. water, cars, trees), or large exposure changes.
- What’s also useful about the GAN is that you can train the generator on your own images to improve results at test time. Suppose you run the network and the output is not very good. You can update the weights of the generator on that exact data in order to better fool the discriminator. This will overfit to your data, but will improve the results on the images you provided.

# Thoughts 💭
- GAN based method that seems robust but not to extreme backgroung variations.
- Inference time might also be a problem.