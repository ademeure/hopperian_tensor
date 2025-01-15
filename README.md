# Insanely Fast H100 Matrix Multiplication & Hardened LLM Processing

Goal: Persistent "hardened" kernel that handles everything that isn't attention in a GPT2/Llama2-like LLM for both forwards and backwards, with easy programmability for both "elementwise" and "row operations", including automatic FP8 with 256x256 scaling factors.

Originally forked from the H100 matrix multiplication code of https://github.com/pranjalssh/fast.cu/