This work has been done as part of a hackathon in engineering school ISAE Supaero, Toulouse, France.
It has been organized by IRT Saint Exupery and computation has been achieved through Google Cloud virtualization.

Objective: Classify 16x16 image patches among 23 classes corresponding to land types in South America
For details, refer to "terrain_classification.pdf"

Feature engineering method: refer to notebooks
- "feature_computation.py" for modular calculation of 62 different imagery features from massive h5 files using PySpark
- "feature_selection.ipynb" to go directly to feature selection and training phases with Random Forests and Adaboost

NB: feature selection and training is performed on an extract of 3M patches with previously calculated features in file features.h5