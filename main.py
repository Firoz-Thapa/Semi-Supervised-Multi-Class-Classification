# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
import os
import urllib.request
warnings.filterwarnings('ignore')


if not os.path.exists('imports-85.data'):
    print("Downloading the dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
    urllib.request.urlretrieve(url, 'imports-85.data')
    print("Dataset downloaded successfully!")
# Load the dataset
df = pd.read_csv('imports-85.data', header=None)

# Assign column names
column_names = [
    'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
    'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
    'wheel-base', 'length', 'width', 'height', 'curb-weight',
    'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system',
    'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
    'city-mpg', 'highway-mpg', 'price'
]
df.columns = column_names

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Number of unique car makes: {df['make'].nunique()}")
print(f"Car makes in dataset: {df['make'].unique()}")

# EDA: Handling missing values
# Replace '?' with NaN
df = df.replace('?', np.nan)

# Check missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0])

# Data type conversion for numeric columns
numeric_columns = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 
                  'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 
                  'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# EDA: Visualizations for understanding the data
# Distribution of car makes
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='make', order=df['make'].value_counts().index)
plt.title('Distribution of Car Makes')
plt.xlabel('Count')
plt.ylabel('Make')
plt.savefig('car_makes_distribution.png')
plt.close()

# Price distribution by make
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, y='make', x='price', order=df['make'].value_counts().index)
plt.title('Price Distribution by Car Make')
plt.xlabel('Price')
plt.ylabel('Make')
plt.savefig('price_distribution_by_make.png')
plt.close()

# Correlation heatmap
plt.figure(figsize=(14, 12))
numerical_df = df.select_dtypes(include=[np.number])
correlation = numerical_df.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', mask=mask)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# Semi-Supervised Learning Implementation

# Step 1: Prepare the data
make_counts = df['make'].value_counts()
print("\nCar make distribution:")
print(make_counts)

# Identify makes with too few samples (less than 3)
rare_makes = make_counts[make_counts < 3].index.tolist()
print(f"\nMakes with fewer than 3 samples: {rare_makes}")

# Option 1: Remove rare classes before splitting
# Create a filtered dataset excluding rare makes
df_filtered = df[~df['make'].isin(rare_makes)]
print(f"\nFiltered dataset shape: {df_filtered.shape}")

# Re-encode classes
le = LabelEncoder()
y_filtered = le.fit_transform(df_filtered['make'])
X_filtered = df_filtered.drop('make', axis=1)
print(f"Number of classes after filtering: {len(le.classes_)}")

# Now perform the train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
)

# Identify categorical and numerical columns
categorical_cols = X_filtered.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_filtered.select_dtypes(include=[np.number]).columns.tolist()

# Step 2: Create preprocessing pipeline
# Numerical pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 3: Split training data into labeled and unlabeled sets (simulating semi-supervised scenario)
X_labeled, X_unlabeled, y_labeled, y_unlabeled_true = train_test_split(
    X_train, y_train, test_size=0.8, random_state=42, stratify=y_train
)

print(f"Labeled data size: {len(X_labeled)}")
print(f"Unlabeled data size: {len(X_unlabeled)}")
print(f"Test data size: {len(X_test)}")

# Step 4: Train base model on labeled data
base_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

base_model.fit(X_labeled, y_labeled)
y_pred_base = base_model.predict(X_test)
base_accuracy = accuracy_score(y_test, y_pred_base)
base_f1 = f1_score(y_test, y_pred_base, average='weighted')

print(f"Base model accuracy: {base_accuracy:.4f}")
print(f"Base model F1 score: {base_f1:.4f}")

# Step 5: Self-training (pseudo-labeling)
# Generate pseudo-labels for unlabeled data
pseudo_labels = base_model.predict(X_unlabeled)

# Combine labeled and pseudo-labeled data
X_combined = pd.concat([X_labeled, X_unlabeled])
y_combined = np.concatenate([y_labeled, pseudo_labels])

# Step 6: Train final model on combined data
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

final_model.fit(X_combined, y_combined)
y_pred_final = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final, average='weighted')

print(f"Final model accuracy: {final_accuracy:.4f}")
print(f"Final model F1 score: {final_f1:.4f}")

# Step 7: Evaluate and compare models
print("\nClassification Report for Base Model:")
print(classification_report(y_test, y_pred_base, target_names=le.classes_))

print("\nClassification Report for Semi-Supervised Model:")
print(classification_report(y_test, y_pred_final, target_names=le.classes_))

# Step 8: Confusion Matrix visualization
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
cm_base = confusion_matrix(y_test, y_pred_base)
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Base Model Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=90)

plt.subplot(1, 2, 2)
cm_final = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Semi-Supervised Model Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# Step 9: Compare different algorithms with semi-supervised approach
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),  # Modified parameters
    'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
}

results = []

for name, classifier in models.items():
    try:
        # Base model on labeled data only
        base_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        base_pipeline.fit(X_labeled, y_labeled)
        base_score = base_pipeline.score(X_test, y_test)
        
        # Generate pseudo-labels
        pseudo_labels = base_pipeline.predict(X_unlabeled)
        
        # Train on combined data
        final_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        final_pipeline.fit(X_combined, y_combined)
        final_score = final_pipeline.score(X_test, y_test)
        
        results.append({
            'Algorithm': name,
            'Supervised Score': base_score,
            'Semi-Supervised Score': final_score,
            'Improvement': final_score - base_score
        })
        
        print(f"Successfully completed {name}")
        
    except Exception as e:
        print(f"Error with {name}: {e}")
        # Add the algorithm with N/A results if it fails
        results.append({
            'Algorithm': name,
            'Supervised Score': float('nan'),
            'Semi-Supervised Score': float('nan'),
            'Improvement': float('nan')
        })

results_df = pd.DataFrame(results)
print(results_df)

# Visualize algorithm comparison
plt.figure(figsize=(12, 6))
results_df_melted = pd.melt(results_df, id_vars=['Algorithm'], 
                           value_vars=['Supervised Score', 'Semi-Supervised Score'],
                           var_name='Method', value_name='Accuracy')

sns.barplot(data=results_df_melted, x='Algorithm', y='Accuracy', hue='Method')
plt.title('Algorithm Performance Comparison: Supervised vs. Semi-Supervised')
plt.ylabel('Accuracy')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('algorithm_comparison.png')
plt.close()

preprocessor.fit(X_combined)
feature_names = []

# Get numeric feature names
if hasattr(preprocessor.named_transformers_['num'], 'get_feature_names_out'):
    feature_names.extend(preprocessor.named_transformers_['num'].get_feature_names_out(numerical_cols))
else:
    feature_names.extend(numerical_cols)

# Get one-hot encoded feature names
if hasattr(preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
    feature_names.extend(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols))

# Extract feature importances from the Random Forest model
rf_model = final_model.named_steps['classifier']
importances = rf_model.feature_importances_

# Create DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(15)

# Visualize feature importances
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature')
plt.title('Top 15 Features for Car Classification')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Step 11: Analyze relationship between accuracy and amount of labeled data (Fixed)
# Create different labeled data percentages - adjusted to ensure enough samples
# Minimum percentage needs to be at least (number of classes / training samples)
min_percentage = len(le.classes_) / len(X_train)
print(f"\nMinimum required labeled percentage: {min_percentage*100:.2f}%")

# Adjust the percentages to ensure they're all above the minimum
labeled_percentages = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
accuracy_results = []

for percentage in labeled_percentages:
    try:
        # Split training data according to current percentage
        X_labeled_subset, X_unlabeled_subset, y_labeled_subset, _ = train_test_split(
            X_train, y_train, test_size=(1-percentage), random_state=42, stratify=y_train
        )
        
        print(f"Processing {percentage*100:.0f}% labeled data - {len(X_labeled_subset)} samples")
        
        # Train base model
        base_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        base_model.fit(X_labeled_subset, y_labeled_subset)
        base_score = base_model.score(X_test, y_test)
        
        # Generate pseudo-labels
        pseudo_labels = base_model.predict(X_unlabeled_subset)
        
        # Combine data
        X_combined_subset = pd.concat([X_labeled_subset, X_unlabeled_subset])
        y_combined_subset = np.concatenate([y_labeled_subset, pseudo_labels])
        
        # Train final model
        final_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        final_model.fit(X_combined_subset, y_combined_subset)
        final_score = final_model.score(X_test, y_test)
        
        accuracy_results.append({
            'Labeled Percentage': percentage * 100,
            'Supervised Score': base_score,
            'Semi-Supervised Score': final_score,
            'Improvement': final_score - base_score
        })
        
    except Exception as e:
        print(f"Error with {percentage*100:.0f}% labeled data: {e}")

# Create DataFrame and visualize (only if we have results)
if accuracy_results:
    accuracy_df = pd.DataFrame(accuracy_results)
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_df['Labeled Percentage'], accuracy_df['Supervised Score'], 
             marker='o', label='Supervised Learning')
    plt.plot(accuracy_df['Labeled Percentage'], accuracy_df['Semi-Supervised Score'], 
             marker='s', label='Semi-Supervised Learning')
    plt.xlabel('Percentage of Labeled Data')
    plt.ylabel('Accuracy')
    plt.title('Effect of Labeled Data Percentage on Model Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('labeled_data_impact.png')
    plt.close()

    print(accuracy_df)
else:
    print("No valid percentage results to display")

# Step 12: Analyze misclassifications
# Get predictions
y_pred = final_model.predict(X_test)
misclassified_indices = np.where(y_pred != y_test)[0]

# Get original data for these indices
X_test_reset = X_test.reset_index(drop=True)
misclassified_data = X_test_reset.iloc[misclassified_indices]
true_labels = le.inverse_transform(y_test[misclassified_indices])
pred_labels = le.inverse_transform(y_pred[misclassified_indices])

# Create a DataFrame with misclassification info
misclassified_df = pd.DataFrame({
    'True Make': true_labels,
    'Predicted Make': pred_labels
})

# Count most common misclassifications
misclass_counts = misclassified_df.groupby(['True Make', 'Predicted Make']).size().reset_index(name='Count')
misclass_counts = misclass_counts.sort_values('Count', ascending=False).head(10)

print("Top 10 Misclassifications:")
print(misclass_counts)

# Visualize common misclassifications
plt.figure(figsize=(12, 6))
misclass_counts['Misclassification'] = misclass_counts['True Make'] + ' → ' + misclass_counts['Predicted Make']
sns.barplot(data=misclass_counts, x='Count', y='Misclassification')
plt.title('Top 10 Common Misclassifications')
plt.tight_layout()
plt.savefig('common_misclassifications.png')
plt.close()

# Step 13: Price analysis by car make
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='price', y='make', order=df['make'].value_counts().index)
plt.title('Price Distribution by Car Make')
plt.xlabel('Price')
plt.ylabel('Make')
plt.savefig('price_distribution.png')
plt.close()

# Final insights and summary
print("\nSemi-Supervised Learning Summary:")
print(f"Best Algorithm: {results_df.loc[results_df['Semi-Supervised Score'].idxmax(), 'Algorithm']}")
print(f"Maximum Accuracy: {results_df['Semi-Supervised Score'].max():.4f}")
print(f"Average Improvement over Supervised: {results_df['Improvement'].mean():.4f}")
print(f"Most Important Feature: {feature_importance_df.iloc[0]['Feature']}")
print(f"Optimal Labeled Data Percentage: {accuracy_df.loc[accuracy_df['Semi-Supervised Score'].idxmax(), 'Labeled Percentage']}%")