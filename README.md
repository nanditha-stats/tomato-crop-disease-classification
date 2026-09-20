# ML-Based Tomato Crop Disease Classification & Feature Importance

An end-to-end Machine Learning and statistical framework designed to classify 10+ distinct tomato plant diseases using precision agriculture datasets.

## 📊 Project Highlights
* **Dataset**: PlantVillage Dataset (18,160 leaf profiles across 10 classes).
* **Peak Accuracy Achieved: 87.42%**
* **Validation Rigor**: Stratified 5-Fold Cross-Validation, Inverse-Frequency Class Weighting, and SMOTE resampling.

## 🛠️ Technical Stack & Tools
* **Languages & Frameworks**: Python, PyTorch, scikit-learn
* **Libraries**: NumPy, Pandas, OpenCV, Scikit-Image (GLCM texture maps), Seaborn, Graphviz
* **Advanced Analytics**: PCA Dimensionality Selection, DeLong's ROC Significance Matrices, Bayesian Information Criterion (BIC) Complexity Penalties

## 🧬 Architectural Interpretability
This framework shifts away from traditional "black-box" models by implementing direct explainability layers:
1. **Inverse Loading Matrix Projection**: Linear algebraic mapping (`np.abs(pca.components_.T @ importance)`) to project abstract mathematical principal components back into raw physical image features (Color variance vs Texture maps).
2. **DeLong's Hypothesis Testing**: Pairwise evaluation from scratch utilizing Mann-Whitney variance structures to mathematically verify if model variations are truly statistically significant.
3. **Per-Class Granular Profiles**: Categorical breakdown loops to isolate precisely which visual signatures drive individual phenotype classification.
