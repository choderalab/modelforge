Profiling
================

Profiling: Overview
------------------------------------------

It is common to profile models to identify bottlenecks and optimize performance. *Modelforge* provides a simple interface to profile models using the `torch.profiler` module. The profiler can be used to profile the forward pass, backward pass, or both, and can be used to profile the model on a single batch or multiple batches.