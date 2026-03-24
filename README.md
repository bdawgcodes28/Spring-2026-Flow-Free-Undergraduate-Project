# Spring-2026-Flow-Free-Undergraduate-Project
Machine Learning Model trained to solve 5x5 Flow Free puzzle boards.

Tech Stack:
- Python OpenCV
- TensorFlow
- MobileNetV2

The goal of the project is to assess how well Machine Learning applications can be used to solve Flow Free puzzle boards. Progress made from this project can be used to further research focused on Machine Learning applications in VLSI wire routing.

Approach:
- Solve FlowFree puzzles manually (5x5 only)
- create training data by:
-   extracting data from unsolved and solved Flow Free screenshots using OpenCV
-   creating synthetic solutions using numpyarrays
- Train MobileNetV2 model with extracted and synthetic datasets
- Test overall accuracy of model
