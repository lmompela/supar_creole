import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Function to read CoNLL-U file and extract sentence structures
def read_conllu(filepath):
    sentences = []
    sentence = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            columns = line.split('\t')
            if len(columns) == 10:
                sentence.append(columns)
    if sentence:
        sentences.append(sentence)
    return sentences

# Compare gold standard with predictions
def compare_predictions(gold_sentences, pred_sentences):
    results = []
    for gold, pred in zip(gold_sentences, pred_sentences):
        for g_token, p_token in zip(gold, pred):
            g_id, g_form, g_head, g_deprel = g_token[0], g_token[1], g_token[6], g_token[7]
            p_head, p_deprel = p_token[6], p_token[7]
            
            results.append({
                'Token': g_form,
                'Gold_Head': g_head,
                'Pred_Head': p_head,
                'Gold_Deprel': g_deprel,
                'Pred_Deprel': p_deprel,
                'Correct_Head': int(g_head == p_head),
                'Correct_Deprel': int(g_deprel == p_deprel)
            })
    return pd.DataFrame(results)

# Perform regression analysis to find error patterns


def analyze_errors(df):
    print("Gold Label Distribution:")
    print(df['Gold_Deprel'].value_counts())
    print("Predicted Label Distribution:")
    print(df['Pred_Deprel'].value_counts())
    
    label_encoder = LabelEncoder()
    df['Gold_Deprel_Code'] = label_encoder.fit_transform(df['Gold_Deprel'])
    df['Pred_Deprel_Code'] = df['Pred_Deprel'].apply(
        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
    )

    X = df[['Gold_Deprel_Code']]
    y = df['Correct_Deprel']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_true_dep = df['Gold_Deprel']
    y_pred_dep = df['Pred_Deprel']

    unique_labels = sorted(set(y_true_dep) | set(y_pred_dep))
    print("Classification Report:")
    print(classification_report(y_true_dep, y_pred_dep, labels=unique_labels, zero_division=0))

    # Print support counts (how often each label appears in gold and predictions)
    gold_counts = y_true_dep.value_counts()
    pred_counts = y_pred_dep.value_counts()

    print("\nLabel Support in Gold vs. Prediction:")
    support_table = pd.DataFrame({'Gold_Count': gold_counts, 'Predicted_Count': pred_counts}).fillna(0)
    print(support_table)

    # Compute F1-score per label
    classification_stats = classification_report(y_true_dep, y_pred_dep, labels=unique_labels, output_dict=True, zero_division=0)

    # Extract support (gold count) and F1-score
    support = []
    f1_scores = []
    for label in unique_labels:
        support.append(support_table.loc[label, "Gold_Count"] if label in support_table.index else 0)
        f1_scores.append(classification_stats[label]["f1-score"] if label in classification_stats else 0)

    # Plot Support vs. F1-score
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(support, f1_scores, alpha=0.7, c=f1_scores, cmap="coolwarm", edgecolors="k")
    plt.colorbar(scatter, label="F1-Score")
    plt.xlabel("Number of Occurrences in Gold File (Support)")
    plt.ylabel("F1-Score")
    plt.title("Relationship Between Data Quantity and Prediction Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("support_vs_f1.png", dpi=300, bbox_inches="tight")
    plt.close()

    missing_labels = sorted(set(df['Gold_Deprel']) - set(df['Pred_Deprel']))
    print("Gold labels that were never predicted:", missing_labels)

    # Ensure missing_labels exist in value_counts before plotting
    missing_counts = [df['Gold_Deprel'].value_counts().get(label, 0) for label in missing_labels]

    # Plot missing labels
    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing_labels, y=missing_counts, hue=missing_labels, palette="Reds_r", legend=False)
    plt.xlabel("Missing Labels")
    plt.ylabel("Occurrences in Gold Data")
    plt.xticks(rotation=45, ha="right")
    plt.title("Gold Labels That Were Never Predicted")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.savefig("missing_labels.png", dpi=300, bbox_inches="tight")
    plt.close()


    #y_true_dep = df['Gold_Deprel_Code']
    #y_pred_dep = df['Pred_Deprel_Code']

    return model, label_encoder, y_true_dep, y_pred_dep, unique_labels

# Visualize model performance per dependency type
def plot_error_patterns(df, y_true_dep, y_pred_dep, label_encoder, unique_labels):
    error_rates = df.groupby('Gold_Deprel')['Correct_Deprel'].mean()
    error_rates = 1 - error_rates  # Convert to error rate
    
    plt.figure(figsize=(14, 8))
    ax= sns.barplot(x=error_rates.index, y=error_rates.values, hue=error_rates.index, palette="Blues_r", legend=False)
    plt.axhline(np.mean(error_rates.values), color='red', linestyle='--', label="Avg Error Rate")
    plt.legend()
    plt.xticks(rotation=90, ha="right")
    plt.xlabel("Dependency Label")
    plt.ylabel("Error Rate")
    plt.title("Model Performance Across Dependency Types")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.savefig("error_rates.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Compute confusion matrix
    cm = confusion_matrix(y_true_dep, y_pred_dep, labels=unique_labels)
    # Plot it as a heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xticks(rotation=90, ha="right")
    plt.yticks(rotation=0)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix of Dependency Predictions")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300,bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", help="Path to gold standard CoNLL-U file")
    parser.add_argument("pred", help="Path to predicted CoNLL-U file")
    args = parser.parse_args()
    
    gold_sentences = read_conllu(args.gold)
    pred_sentences = read_conllu(args.pred)
    
    results_df = compare_predictions(gold_sentences, pred_sentences)
    model, encoder,y_true_dep, y_pred_dep, unique_labels = analyze_errors(results_df)
    plot_error_patterns(results_df, y_true_dep, y_pred_dep, encoder, unique_labels)
