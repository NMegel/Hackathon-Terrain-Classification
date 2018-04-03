# Hackathon Terrain Classification

This work has been done as part of a hackathon at ISAE Supaero, Toulouse, France.
It has been organized by IRT Saint Exupery and computation has been achieved through Google Cloud virtualization and Spark paralelization

The objective was to classify 16x16 image patches among 23 classes corresponding to land types in South America
For details, refer to "terrain_classification.pdf". The presented method is adaptable to any classification task on labeled image database

Feature engineering method: refer to code
- "feature_computation.py" for modular calculation of 62 different imagery features from massive h5 files using PySpark
- "feature_selection.ipynb" to go directly to feature selection and training phases tutorial with Random Forests and Adaboost

This computation is designed for any image database and paralelized with PySpark
A relevant try is to practice on ImageNet
See at http://image-net.org/download-images
