import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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

def compare_predictions(gold_sentences, pred_sentences):
    results = []
    for gold, pred in zip(gold_sentences, pred_sentences):
        sentence_length = len(gold)
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
                'Correct_Deprel': int(g_deprel == p_deprel),
                'Sentence_Length': sentence_length
            })
    return pd.DataFrame(results)

def analyze_errors(df):
    print("Gold Label Distribution:")
    print(df['Gold_Deprel'].value_counts())
    print("Predicted Label Distribution:")
    print(df['Pred_Deprel'].value_counts())

    label_encoder = LabelEncoder()
    df['Gold_Code'] = label_encoder.fit_transform(df['Gold_Deprel'])
    df['Pred_Code'] = df['Pred_Deprel'].apply(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1)

    X = df[['Gold_Code']]
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

    # Confusion matrix and adjacent confusions
    cm = confusion_matrix(y_true_dep, y_pred_dep, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    cm_df.to_csv("confusion_matrix.csv")
    print("\nConfusion matrix saved as confusion_matrix.csv")

    support_table = pd.DataFrame({
        'Gold_Count': y_true_dep.value_counts(),
        'Predicted_Count': y_pred_dep.value_counts()
    }).fillna(0)

    # Length-based error analysis
    length_bins = pd.cut(df['Sentence_Length'], bins=[0, 10, 20, 30, 100], labels=['0-10', '11-20', '21-30', '30+'])
    length_errors = df.groupby(length_bins)['Correct_Deprel'].mean().apply(lambda x: 1 - x)
    print("\nError rate by sentence length:")
    print(length_errors)

    return model, label_encoder, y_true_dep, y_pred_dep, unique_labels, cm

def plot_error_patterns(df, y_true_dep, y_pred_dep, label_encoder, unique_labels):
    error_rates = df.groupby('Gold_Deprel')['Correct_Deprel'].mean()
    error_rates = 1 - error_rates
    plt.figure(figsize=(14, 8))
    sns.barplot(x=error_rates.index, y=error_rates.values, palette="coolwarm")
    plt.axhline(np.mean(error_rates), color='red', linestyle='--', label="Avg Error Rate")
    plt.xticks(rotation=90)
    plt.title("Error Rate by Dependency Label")
    plt.savefig("error_rate_by_label.png", bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(y_true_dep, y_pred_dep, labels=unique_labels)
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xticks(rotation=90)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_heatmap.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", help="Path to gold CoNLL-U")
    parser.add_argument("pred", help="Path to predicted CoNLL-U")
    args = parser.parse_args()

    gold_sentences = read_conllu(args.gold)
    pred_sentences = read_conllu(args.pred)
    df = compare_predictions(gold_sentences, pred_sentences)
    model, encoder, y_true_dep, y_pred_dep, unique_labels, cm = analyze_errors(df)
    plot_error_patterns(df, y_true_dep, y_pred_dep, encoder, unique_labels)

    # Save the full confusion matrix to CSV
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    cm_df.to_csv("confusion_matrix.csv")

    # Identify top confused label pairs
    confused_pairs = []
    for i, gold in enumerate(unique_labels):
        for j, pred in enumerate(unique_labels):
            if gold != pred:
                count = cm[i][j]
                if count > 0:
                    confused_pairs.append((gold, pred, count))

    # Sort and select top 20 confused pairs
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    top_confused_df = pd.DataFrame(confused_pairs[:20], columns=["Gold_Label", "Predicted_Label", "Count"])

    # Print and save
    print("\nTop 20 Confused Label Pairs (Gold → Predicted):")
    print(top_confused_df.to_string(index=False))

    top_confused_df.to_csv("top_confused_pairs.csv", index=False)
