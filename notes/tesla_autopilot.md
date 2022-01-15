# Info 📌
https://www.reddit.com/r/MachineLearning/comments/mq3led/d_how_is_tesla_autopilot_trained/

# Post header 📝
How is Tesla autopilot trained?  
To my understanding, they have a big NN (does anyone know the architecture?) trained (from scratch before every release?) on a set of milions of image (each handchosen by humans?) that is evolving all the time, e.g images are taken off and taken on based on interaction of the model with the fleet of cars like humans corrections etc. Is that correct?

# Comments section 👂🏻
>For visual perception, it’s a giant multi-task CNN-Transformer.
>Predicts shit like road segmentation, lane lines, object bounding boxes, etc. There’s nothing fancy.
>Most of the useful logic resides in hard-coded control rules. Such as, if a car is on my left in lane #1, while I’m on freeway lane #2, if the driver wants to turn left, wait until 100 meters of clearance. If no clearance for 5 seconds, accelerate to change lanes. These specific parameters are mostly determined by human annotations, although a RL algorithm attempts to provide initial estimates from the fleet data.
>It’s not some giant AGI hive-mind neural network bullshit that Elon implies. It has a lot of hard-coded rules. Very similar to Waymo.

# Notes ✍🏻
- Informative youtube video that explains it (AI Tesla head speak):  
https://www.youtube.com/watch?v=oBklltKXtDE
- Use Pytorch to train their models. In-house hardware deployment (own GPU, dojo).
- Autopilot named software (lane keep, distance keep). No Lidar, just 8 cameras. A lot f CNNs.
- 960x1280 resolution image, ResNet-50 shared backbone to FPN / DeepLab v3 / UNet heads architecture called HydraNets. 15 tasks in total.
- Two rounds processing to estimate additionnal information from multiple cameras (so 8 HydraNets in total)(depth for example).
- A lot of BEV trasnformation from multiple camera informations.
- Memory issue as they have to handle 4096 image for a single inference. Solved by data & model parallelization.

# Thoughts 💭
