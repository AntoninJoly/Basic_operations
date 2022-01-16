# Info 📌
https://www.reddit.com/r/MachineLearning/comments/eftv1o/d_what_frustrates_you_about_ml_tools_libraries/

# Post header 📝
What frustrates you about ML tools / libraries that people don’t talk enough about?  
I’ll start - sparse matrices in Pandas that aren’t supported in SKLearn. Both tools are great but for some reason it took hours to find out that SKLearn will by default inflate the sparse matrix (in my case 30MB>>20GB) without throwing any warning...

# Comments section 👂🏻
>Some papers are almost unreadable because the notation is total bananas. Mathematicians and theoretical physicists have spent centuries getting their notation right. 

> Memory management. Training my model with seemingly innocuous operations, and suddenly memory use blows up with no obvious way to solve it.

> 
# Notes ✍🏻
- Haven't tried but apparently Modin speeds up Pandas. There's also Vaex  
https://github.com/modin-project/modin
-

# Thoughts 💭
