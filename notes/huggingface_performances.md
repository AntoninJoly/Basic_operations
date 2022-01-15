# Info 📌
https://www.reddit.com/r/MachineLearning/comments/r0y56t/p_python_library_to_optimize_hugging_face/

# Post header 📝
Python library to optimize Hugging Face transformer for inference: < 0.5 ms latency / 2850 infer/sec  
We just launched a new open source Python library to help in optimizing Transformer model inference and prepare deployment in production.  
It’s a follow up of a proof of concept shared on Reddit. Scripts have been converted to a Python library (Apache 2 license) to be used in any NLP project.
Basically, most tutorials on how to deploy in production a transformer model tell you to take FastAPI and put Pytorch inside. There are many reasons why it’s a bad idea, first of all, the inference performance is very low.

# Comments section 👂🏻
>CPUs don't benefit from batching - they do BS1 inference just fine. So there's not much benefit using a model server, as opposed to just wrapping ONNRuntime in FastAPI

>Thank you for your question. TBH we already got this request a few times, but it's not obvious to us how many people are deploying transformer on CPU.

>Is it for economical/cost reasons? Or have you a use case with very small architecture and short sequences?

>Like said by u/JustOneAvailableName Triton supports CPU, the challenge is all about converting ONNX model to OpenVINO format (not especially hard) or just use CPU backend from ONNX Runtime (slightly less optimized however).

# Notes ✍🏻
None

# Thoughts 💭
- Efficient for GPU based inference.