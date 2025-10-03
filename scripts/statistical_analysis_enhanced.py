import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import statsmodels.api as sm  # For LOESS smoothing

# Modified: Read CoNLL-U file and check if extra features are available (i.e. at least 10 columns)
def read_conllu(filepath):
    sentences = []
    extra_features_available = False
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
            # Mark extra features as available if token has at least 10 columns
            if len(columns) >= 10:
                extra_features_available = True
                sentence.append(columns)
            else:
                # If token doesn't have enough columns, still add the basic token info
                sentence.append(columns)
        if sentence:
            sentences.append(sentence)
    return sentences, extra_features_available

# Compare gold standard with predictions and enrich with additional features (if available)
def compare_predictions(gold_sentences, pred_sentences, extra_features_available):
    results = []
    function_tags = {"DET", "ADP", "CONJ", "PRON", "PART", "AUX", "SCONJ", "CCONJ"}
    for gold, pred in zip(gold_sentences, pred_sentences):
        sent_length = len(gold)
        for i, (g_token, p_token) in enumerate(zip(gold, pred)):
            # Token index (fallback to position if not integer)
            try:
                token_index = int(g_token[0])
            except ValueError:
                token_index = i + 1

            # Gold head (assume numeric; if not, use None)
            try:
                gold_head = int(g_token[6])
            except (ValueError, IndexError):
                gold_head = None
            head_distance = abs(token_index - gold_head) if gold_head is not None else np.nan

            # Only extract UPOS and MISC if extra features are available
            upos = g_token[3] if extra_features_available and len(g_token) > 3 else None
            misc = g_token[9] if extra_features_available and len(g_token) > 9 else ""
            
            # Extract language from MISC if available (e.g., "Lang=XX")
            language = None
            if extra_features_available and "Lang=" in misc:
                for item in misc.split("|"):
                    if item.startswith("Lang="):
                        language = item.split("=")[1]
                        break

            # Determine if token is a function word (only if UPOS is available)
            is_function = 1 if upos in function_tags else 0 if upos is not None else np.nan

            results.append({
                'Token': g_token[1] if len(g_token) > 1 else "",
                'Gold_Head': g_token[6] if len(g_token) > 6 else None,
                'Pred_Head': p_token[6] if len(p_token) > 6 else None,
                'Gold_Deprel': g_token[7] if len(g_token) > 7 else None,
                'Pred_Deprel': p_token[7] if len(p_token) > 7 else None,
                'Correct_Head': int(g_token[6] == p_token[6]) if len(g_token) > 6 and len(p_token) > 6 else 0,
                'Correct_Deprel': int(g_token[7] == p_token[7]) if len(g_token) > 7 and len(p_token) > 7 else 0,
                'Sentence_Length': sent_length if extra_features_available else np.nan,
                'Token_Index': token_index if extra_features_available else np.nan,
                'Head_Distance': head_distance if extra_features_available else np.nan,
                'UPOS': upos,
                'Is_Function': is_function,
                'Language': language
            })
    return pd.DataFrame(results)

# Original error analysis (with added grid search for hyperparameters and LOESS smoothing for support vs F1)
def analyze_errors(df):
    print("Gold Label Distribution:")
    print(df['Gold_Deprel'].value_counts())
    print("Predicted Label Distribution:")
    print(df['Pred_Deprel'].value_counts())
    
    label_encoder = LabelEncoder()
    df['Gold_Deprel_Code'] = label_encoder.fit_transform(df['Gold_Deprel'].astype(str))
    df['Pred_Deprel_Code'] = df['Pred_Deprel'].apply(
        lambda x: label_encoder.transform([str(x)])[0] if str(x) in label_encoder.classes_ else -1
    )

    X = df[['Gold_Deprel_Code']]
    y = df['Correct_Deprel']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_true_dep = df['Gold_Deprel']
    y_pred_dep = df['Pred_Deprel']

    unique_labels = sorted(set(y_true_dep) | set(y_pred_dep))
    print("Classification Report:")
    print(classification_report(y_true_dep, y_pred_dep, labels=unique_labels, zero_division=0))

    # Support counts for each label
    gold_counts = y_true_dep.value_counts()
    pred_counts = y_pred_dep.value_counts()
    print("\nLabel Support in Gold vs. Prediction:")
    support_table = pd.DataFrame({'Gold_Count': gold_counts, 'Predicted_Count': pred_counts}).fillna(0)
    print(support_table)

    # F1-score per label and LOESS smoothing for trend line and threshold detection
    classification_stats = classification_report(y_true_dep, y_pred_dep, labels=unique_labels, output_dict=True, zero_division=0)
    support = []
    f1_scores = []
    for label in unique_labels:
        support.append(support_table.loc[label, "Gold_Count"] if label in support_table.index else 0)
        f1_scores.append(classification_stats[label]["f1-score"] if label in classification_stats else 0)

    # Compute LOESS smoothed values (using frac=0.3; adjust as needed)
    loess_result = sm.nonparametric.lowess(f1_scores, support, frac=0.3)
    
    # Determine threshold support for a target F1 (e.g., 0.80)
    f1_target = 0.80
    threshold = None
    for s, f in loess_result:
        if f >= f1_target:
            threshold = s
            break

    if threshold is not None:
        print(f"Minimum support for LOESS-smoothed F1 >= {f1_target} is: {threshold:.1f}")
    else:
        print(f"No support level reached an LOESS-smoothed F1 of {f1_target}")

    # Plot support vs F1 with LOESS trend line and threshold line
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(support, f1_scores, alpha=0.7, c=f1_scores, cmap="coolwarm", edgecolors="k", label='Data')
    plt.plot(loess_result[:, 0], loess_result[:, 1], color='black', linestyle='--', label='LOESS Trend')
    if threshold is not None:
        plt.axvline(x=threshold, color='red', linestyle=':', 
                    label=f'Support threshold (F1 >= {f1_target}): {threshold:.1f}')
        plt.text(threshold, f1_target, f' {threshold:.1f}', color='red', fontsize=10)
    plt.colorbar(scatter, label="F1-Score")
    plt.xlabel("Number of Occurrences in Gold File (Support)")
    plt.ylabel("F1-Score")
    plt.title("Relationship Between Data Quantity and Prediction Accuracy (with LOESS Trend)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("support_vs_f1.png", dpi=300, bbox_inches="tight")
    plt.close()

    missing_labels = sorted(set(df['Gold_Deprel']) - set(df['Pred_Deprel']))
    print("Gold labels that were never predicted:", missing_labels)
    missing_counts = [df['Gold_Deprel'].value_counts().get(label, 0) for label in missing_labels]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing_labels, y=missing_counts, hue=missing_labels, palette="Reds_r", legend=False)
    plt.xlabel("Missing Labels")
    plt.ylabel("Occurrences in Gold Data")
    plt.xticks(rotation=45, ha="right")
    plt.title("Gold Labels That Were Never Predicted")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.savefig("missing_labels.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Hyperparameter tuning for the logistic regression using GridSearchCV
    hyperparameter_tuning(X, y)

    return model, label_encoder, y_true_dep, y_pred_dep, unique_labels

# New: PCA clustering on token-level features to visualize error patterns
def plot_pca_clustering(df):
    features = df[['Token_Index', 'Sentence_Length', 'Head_Distance', 'Is_Function']].copy()
    features['Norm_Index'] = features['Token_Index'] / features['Sentence_Length']
    X_features = features[['Norm_Index', 'Head_Distance', 'Is_Function']].fillna(0).values
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_features)
    df['PCA1'] = X_reduced[:, 0]
    df['PCA2'] = X_reduced[:, 1]
    df['Error'] = 1 - df['Correct_Deprel']  # 1 indicates an error
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['PCA1'], df['PCA2'], c=df['Error'], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Error (1 if error, 0 if correct)')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Clustering of Dependency Parsing Errors")
    plt.savefig("pca_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

# New: Correlate error rate with sentence length (binned) and export raw numbers
def plot_error_vs_sentence_length(df):
    df['Error'] = 1 - df['Correct_Deprel']
    max_len = df['Sentence_Length'].max()
    bins = np.arange(0, max_len + 5, 5)
    df['Sentence_Length_Bin'] = pd.cut(df['Sentence_Length'], bins=bins)
    error_by_length = df.groupby('Sentence_Length_Bin', observed=False)['Error'].mean()
    error_by_length_df = error_by_length.reset_index()
    error_by_length_df.columns = ['Sentence_Length_Bin', 'Mean_Error_Rate']
    error_by_length_df.to_csv("error_by_sentence_length.csv", index=False)
    print(error_by_length_df)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(error_by_length)), error_by_length.values, tick_label=[str(x) for x in error_by_length.index])
    plt.xlabel("Sentence Length Bin")
    plt.ylabel("Mean Error Rate")
    plt.title("Error Rate vs. Sentence Length")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("sentence_length_vs_error.png", dpi=300, bbox_inches="tight")
    plt.close()

# New: Correlate error rate with head distance (binned) and export raw numbers
def plot_error_vs_head_distance(df):
    df['Error'] = 1 - df['Correct_Deprel']
    max_distance = df['Head_Distance'].max()
    bins = np.arange(0, max_distance + 2, 2)
    df['Head_Distance_Bin'] = pd.cut(df['Head_Distance'], bins=bins)
    error_by_distance = df.groupby('Head_Distance_Bin', observed=False)['Error'].mean()
    error_by_distance_df = error_by_distance.reset_index()
    error_by_distance_df.columns = ['Head_Distance_Bin', 'Mean_Error_Rate']
    error_by_distance_df.to_csv("error_by_head_distance.csv", index=False)
    print(error_by_distance_df)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(error_by_distance)), error_by_distance.values, tick_label=[str(x) for x in error_by_distance.index])
    plt.xlabel("Head Distance Bin")
    plt.ylabel("Mean Error Rate")
    plt.title("Error Rate vs. Head Distance")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("head_distance_vs_error.png", dpi=300, bbox_inches="tight")
    plt.close()

# New: Cross-lingual confusion analysis if language information is available
def cross_lingual_confusion(df):
    languages = df['Language'].dropna().unique()
    for lang in languages:
        df_lang = df[df['Language'] == lang]
        if df_lang.empty:
            continue
        unique_labels = sorted(set(df_lang['Gold_Deprel']) | set(df_lang['Pred_Deprel']))
        cm = confusion_matrix(df_lang['Gold_Deprel'], df_lang['Pred_Deprel'], labels=unique_labels)
        plt.figure(figsize=(14, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("Gold Label")
        plt.title(f"Confusion Matrix for Language: {lang}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{lang}.png", dpi=300, bbox_inches="tight")
        plt.close()

# New: Hyperparameter tuning for logistic regression using GridSearchCV
def hyperparameter_tuning(X, y):
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    model = LogisticRegression(max_iter=200)
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X, y)
    best_params = grid.best_params_
    best_score = grid.best_score_
    with open("grid_search_results.txt", "w") as f:
        f.write("Best Parameters: {}\n".format(best_params))
        f.write("Best Cross-Validated Accuracy: {:.4f}\n".format(best_score))
    print("Grid search complete. Best parameters:", best_params, "with accuracy:", best_score)

# Visualize overall model performance per dependency type (original)
def plot_error_patterns(df, y_true_dep, y_pred_dep, label_encoder, unique_labels):
    error_rates = df.groupby('Gold_Deprel')['Correct_Deprel'].mean()
    error_rates = 1 - error_rates  # Convert to error rate
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x=error_rates.index, y=error_rates.values, hue=error_rates.index, palette="Blues_r", legend=False)
    plt.axhline(np.mean(error_rates.values), color='red', linestyle='--', label="Avg Error Rate")
    plt.legend()
    plt.xticks(rotation=90, ha="right")
    plt.xlabel("Dependency Label")
    plt.ylabel("Error Rate")
    plt.title("Model Performance Across Dependency Types")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.savefig("error_rates.png", dpi=300, bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(y_true_dep, y_pred_dep, labels=unique_labels)
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xticks(rotation=90, ha="right")
    plt.yticks(rotation=0)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix of Dependency Predictions")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", help="Path to gold standard CoNLL-U file")
    parser.add_argument("pred", help="Path to predicted CoNLL-U file")
    args = parser.parse_args()
    
    # Read files and determine if extra features are available
    gold_sentences, gold_extra = read_conllu(args.gold)
    pred_sentences, pred_extra = read_conllu(args.pred)
    extra_features_available = gold_extra and pred_extra

    results_df = compare_predictions(gold_sentences, pred_sentences, extra_features_available)
    model, encoder, y_true_dep, y_pred_dep, unique_labels = analyze_errors(results_df)
    plot_error_patterns(results_df, y_true_dep, y_pred_dep, encoder, unique_labels)

    # Conditionally run extra analyses if extra features are available
    if extra_features_available:
        # PCA clustering requires token index, sentence length, head distance, and is_function
        if all(col in results_df.columns for col in ["Token_Index", "Sentence_Length", "Head_Distance", "Is_Function"]) and results_df[["Token_Index", "Sentence_Length", "Head_Distance", "Is_Function"]].notnull().all().all():
            plot_pca_clustering(results_df)
        else:
            print("Skipping PCA clustering analysis: extra features missing or incomplete.")

        if "Sentence_Length" in results_df.columns and results_df["Sentence_Length"].notnull().any():
            plot_error_vs_sentence_length(results_df)
        else:
            print("Skipping error vs. sentence length analysis: sentence length information not available.")

        if "Head_Distance" in results_df.columns and results_df["Head_Distance"].notnull().any():
            plot_error_vs_head_distance(results_df)
        else:
            print("Skipping error vs. head distance analysis: head distance information not available.")
    else:
        print("Extra feature analyses (PCA, sentence length, head distance) are skipped due to lack of extra features.")

    # Cross-lingual confusion analysis requires language information
    if "Language" in results_df.columns and results_df["Language"].notnull().any():
        cross_lingual_confusion(results_df)
    else:
        print("Skipping cross-lingual confusion analysis: language information not available.")
