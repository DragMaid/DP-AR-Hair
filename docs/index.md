# AR Hair Transfer (Based on *Hair Shifter*)

Welcome to the documentation for the **AR Hair Transfer** project — an applied implementation inspired by the research paper *Hair Shifter*. This project explores real-time and offline hair appearance transfer, enabling users to map hairstyle attributes from one subject to another using modern computer vision, deep learning, and differentiable rendering techniques.

## Overview

The system aims to perform:

* **Hair appearance transfer** — mapping texture, color, and style cues onto a target portrait.
* **High-fidelity rendering** — preserving occlusion boundaries, lighting, and semantic consistency.
* **Modular model loading** — using a registry-based loader for PyTorch models and weight management.
* **Dataset-agnostic integration** — supporting COCO, OpenImages, and other datasets via a transformer-driven inserter.

While the underlying model ideas are informed by *Hair Shifter*, this implementation is designed to be practical and compatibility-focused, supporting lighter GPUs and efficient inference pathways.

## Key Features

* **Optimized model registry**: Unified interface for loading architectures and optionally loading pretrained weights.
* **Flexible weight loader**: Standardized weight paths and auto-download support.
* **Dataset transformers**: Allow multiple dataset formats to be converted into a COCO-like structure.
* **Lazy, threaded URL validation**: Ensures only accessible image references are inserted.
* **Dockerized environment**: Ensures reproducible builds for both development and deployment.
* **Redis-backed task management**: Supports rate-limited async email handling for web integrations.

## Technical Goals

* Support **real-time hair transfer** on mid-range GPUs.
* Modularize the codebase so every component can be swapped or extended.
* Maintain compatibility with research workflows while keeping the codebase production-friendly.
* Prioritize stability, predictable behavior, and clean architecture.

## Why This Project Exists

Since the Hair Shifter paper did not leave any link to their implementation, this is a re-creation of what they proposed on their paper in addition with these following features:

* **Practical usability**
* **Ease of extension**
* **Efficient inference**
* **Clear documentation**

If you are exploring AR, image-based rendering, or fashion/appearance modeling, this project provides a structured and understandable foundation.

Actually the main reason I'm doing this is to go to Seoul

## Getting Started

* Explore the model architecture and registry under `loaders/`.
* Review the dataset insertion pipeline in `inserter/`.
* Check the examples section for common usage patterns.
* Use the Docker setup for consistent deployment.

## Project Vision

The long-term goal is to evolve this into a general **AR appearance modification engine**, capable of handling hairstyles, facial attributes, accessories, and more — all modular and dataset-agnostic.

Continue to the next sections to learn about installation, architecture design, and usage examples.
