# Semi-Supervised Multi-Class Car Classification
## A Comparative Study of Classification Approaches

---

# Presentation Outline
- Introduction to Semi-Supervised Learning
- Dataset Overview & Preprocessing
- Methodology
- Implementation Details
- Exploratory Data Analysis
- Model Performance Comparison
- Impact of Labeled Data
- Feature Importance Analysis
- Error Analysis
- Conclusions

---

# Introduction to Semi-Supervised Learning

## What is Semi-Supervised Learning?
- Machine learning paradigm that uses **both labeled and unlabeled data**
- Particularly valuable when labeled data is **scarce or expensive**
- Leverages patterns in unlabeled data to improve classification

## Project Objective
- Implement multi-class classification of car manufacturers using semi-supervised learning
- Compare performance of supervised vs. semi-supervised approaches
- Determine optimal amount of labeled data needed
- Identify which algorithms benefit most from semi-supervised learning

---

# Dataset Overview & Preprocessing

## Automobile Dataset
- **205 instances** with **26 features**
- **22 different car manufacturers** (makes)
- Features include dimensions, engine specs, performance metrics, and pricing
- **Class imbalance**: Some makes have very few samples
- **Missing values** in 7 features

## Preprocessing Steps
- Handling missing values using mean/mode imputation
- Encoding categorical features
- Removing makes with fewer than 3 samples (renault, mercury)
- **Final dataset**: 202 instances, 20 classes

## Distribution of Car Makes
[IMAGE: car_makes_distribution.png]

---

# Methodology

## Semi-Supervised Learning Approach
1. Split data into **train/test** sets (80/20 split)
2. From training data, create **labeled/unlabeled** portions (20/80 split)
3. Train **base model** on labeled data only
4. Use base model to generate **pseudo-labels** for unlabeled data
5. Train **final model** on combined data (labeled + pseudo-labeled)
6. Evaluate on held-out test set

## Experimental Setup
- **Algorithms**: Random Forest, SVM, Neural Network, XGBoost
- **Evaluation metrics**: Accuracy, F1-score
- **Labeled data experiments**: Varied from 30% to 80% to find optimal ratio

---

# Implementation Details

## Data Pipeline
```python
# Data preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])
```

## Semi-Supervised Learning Implementation
```python
# Base model on labeled data only
base_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])
base_pipeline.fit(X_labeled, y_labeled)

# Generate pseudo-labels
pseudo_labels = base_pipeline.predict(X_unlabeled)

# Train on combined data
X_combined = pd.concat([X_labeled, X_unlabeled])
y_combined = np.concatenate([y_labeled, pseudo_labels])

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])
final_pipeline.fit(X_combined, y_combined)
```

---

# Exploratory Data Analysis

## Price Distribution by Car Make
[IMAGE: price_distribution.png]

### Key Observations:
- **Luxury brands** (Mercedes-Benz, BMW, Jaguar) have distinctly higher pricing
- **Economy brands** (Chevrolet, Plymouth, Dodge) cluster at lower price points
- **Mid-tier manufacturers** (Volvo, Audi) show moderate pricing
- Several outliers indicate specialty models

---

# Feature Correlations & Relationships

## Correlation Heatmap
[IMAGE: correlation_heatmap.png]

### Key Relationships:
- **Strong correlation (0.87)** between dimensions (length, width) and curb-weight
- **Negative correlation (-0.80)** between engine characteristics and fuel efficiency
- Physical dimensions highly correlated with each other
- Engine performance metrics form distinct clusters
- These relationships reflect fundamental automotive design philosophies

---

# Model Performance Comparison

## Algorithm Performance Analysis
[IMAGE: algorithm_comparison.png]

## Key Findings
| Algorithm | Supervised | Semi-Supervised | Improvement |
|---|---|---|---|
| Random Forest | 65.9% | 61.0% | -4.9% |
| SVM | 26.8% | 48.8% | +21.9% |
| Neural Network | 41.5% | 56.1% | +14.6% |
| XGBoost | N/A | N/A | N/A |

### Observations:
- **Random Forest**: Best overall accuracy but slight decrease with semi-supervised learning
- **SVM**: Dramatic improvement with semi-supervised approach
- **Neural Network**: Significant improvement with semi-supervised approach
- **Average improvement**: +10.6% across algorithms

---

# Impact of Labeled Data Percentage

## Effect of Labeled Data on Accuracy
[IMAGE: labeled_data_impact.png]

## Key Insights
- **Minimum required labeled percentage**: 12.4% (due to class constraints)
- **Optimal performance at 70%** labeled data for both approaches
- Semi-supervised approach more beneficial with **less labeled data**
- Performance of both approaches **converges at higher labeled percentages**
- At 80%, semi-supervised learning slightly outperforms supervised learning

---

# Feature Importance Analysis

## Top Features for Classification
[IMAGE: feature_importance.png]

## Most Important Features:
1. **Curb-weight**: Strongest predictor of car manufacturer
2. **Length**: Second most important feature
3. **Height**: Third most important feature
4. **Normalized-losses**: Fourth most important feature

### Significance:
- Physical dimensions strongly reflect manufacturer design philosophies
- Different brands have characteristic size and weight profiles
- These features are more stable indicators than performance metrics

---

# Error Analysis

## Confusion Matrices
[IMAGE: confusion_matrices.png]

## Common Misclassifications
[IMAGE: common_misclassifications.png]

## Patterns in Errors
- BMW often misclassified as Toyota
- Related manufacturers confused with each other (Dodge ↔ Plymouth)
- Similar market segment vehicles confused (Honda ↔ Mitsubishi)
- Certain makes (Alfa-Romeo, Chevrolet, Jaguar) consistently misclassified
- Perfect classification for some brands (Audi, Porsche, Subaru, Volkswagen, Volvo)

---

# Classification Reports Comparison

## Supervised Model Performance (Base)
```
               precision    recall  f1-score   support
  alfa-romero       0.00      0.00      0.00         1
         audi       1.00      1.00      1.00         1
          bmw       0.50      0.50      0.50         2
    chevrolet       0.00      0.00      0.00         1
        dodge       0.00      0.00      0.00         2
        honda       1.00      0.33      0.50         3
        isuzu       0.50      1.00      0.67         1
       jaguar       0.00      0.00      0.00         1
        mazda       0.75      1.00      0.86         3
mercedes-benz       0.50      0.50      0.50         2
   mitsubishi       0.33      0.67      0.44         3
       nissan       1.00      0.75      0.86         4
     accuracy                           0.66        41
```

## Semi-Supervised Model Performance
```
               precision    recall  f1-score   support
  alfa-romero       0.00      0.00      0.00         1
         audi       1.00      1.00      1.00         1
          bmw       0.50      0.50      0.50         2
    chevrolet       0.00      0.00      0.00         1
        dodge       0.00      0.00      0.00         2
        honda       1.00      0.33      0.50         3
        isuzu       0.00      0.00      0.00         1
       jaguar       0.00      0.00      0.00         1
        mazda       0.50      0.67      0.57         3
     accuracy                           0.61        41
```

---

# Conclusions

## Semi-Supervised Learning Summary
- **Best Algorithm**: Random Forest (60.9% accuracy)
- **Largest Improvement**: SVM (+21.9% with semi-supervised approach)
- **Most Important Feature**: curb-weight
- **Optimal Labeled Data Percentage**: 70%
- **Minimum labeled data required**: 12.4% due to class distribution constraints

## Practical Implications
- Semi-supervised learning provides significant benefits when labeled data is limited
- Different algorithms respond differently to pseudo-labeling
  - SVM and Neural Networks gain the most
  - Random Forest slightly degraded with pseudo-labels
- Physical characteristics (dimensions, weight) are strong indicators of car manufacturer
- Challenge: Many car makes have too few samples for reliable classification

---

# Future Directions

## Potential Improvements
- **Advanced semi-supervised techniques**:
  - Label propagation
  - Graph-based methods
  - Consistency regularization

- **Data augmentation** to address class imbalance

- **Transfer learning** from pre-trained automotive models

- **Active learning** to optimize which samples to label

## Applications
- Automated vehicle inventory categorization
- Competitive analysis of vehicle specifications
- Insurance and pricing model development

---

# Thank You!

## Contact Information
Firoz Thapa
gyawat.magar@gmail.com

### Questions?
