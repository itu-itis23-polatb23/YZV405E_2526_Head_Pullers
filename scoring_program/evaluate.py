#!/usr/bin/env python3
"""
Script to evaluate predictions for PARSEME 2.0 Subtask 2 (B).

Original author: Manon Scholivet
Code improvements & comments: ChatGPT

Description:
    This script compares system predictions to gold references for
    minimal and creative rephrasings, using BERTScore as the metric.
    It also computes the diversity of the prediction and references.

Usage example:
    time python3 evaluate.py gold.json pred_minimal.json
"""

import argparse
import json
import diversutils
import difflib
import spacy
import regex as re
from typing import Dict, List, Tuple
from bert_score import score
from itertools import compress
from collections import Counter


ja_tokenizer = spacy.load("ja_core_news_sm")


################################################################################
def merge_special_case(tokens: List[str]) -> List[str]:
    """
    Merge the French clitic "d'" with the token that follows it.
    Same for the words starting with a '-'
    Example:
        Input:  ["je", "parle", "d'", "amour"]
        Output: ["je", "parle", "d'amour"]

    Behavior:
    - Whenever the token "d'" appears and it is not the last token,
      it is concatenated with the following token.
    - The merged token replaces both original tokens.
    - All other tokens are left unchanged.

    Args:
        tokens: A list of token strings.

    Returns:
        A new list of tokens in which occurrences of "d'" have been merged
        with the next token.
    """
    
    merged: List[str] = []
    i = 0
    
    while i < len(tokens):
        # Check if the current token is "d'" and there's a following token.
        if i + 1 < len(tokens) and (tokens[i] == "d'" or tokens[i+1].startswith('-')) :
            # Merge "d'" with the next token.
            merged.append(tokens[i] + tokens[i + 1])
            # Skip the next token since it's already merged.
            i += 2
        else:
            # Keep the token as-is.
            merged.append(tokens[i])
            i += 1

    return merged

################################################################################
def extract_mwe_tokens(annotated_text: str, is_JA: bool) -> List[str]:
    """Extract all tokens inside [[...]] segments, merging multiple segments."""
    matches = re.findall(r"\[\[(.*?)\]\]", annotated_text)
    tokens = []
    
    for match in matches:
        elements = match.strip().split()
        if is_JA:
            for element in elements:
                current_tokens = [str(tok) for tok in ja_tokenizer(element)]
                tokens.extend(current_tokens)
        else:
            tokens.extend(match.strip().split())
    tokens = merge_special_case(tokens)
    return tokens

################################################################################
def has_mwe_been_deleted(item: Dict[str, str], is_JA: bool) -> bool:
    """
    Determine if the Multiword Expression (MWE) from the original text
    has been partially or fully deleted in the prediction.

    Extraction rules:
    - All MWE tokens are identified inside [[...]] annotations in item["text"].
    - Multiple [[...]] segments are concatenated to form the full MWE token list.
    - A deletion is considered valid if at least one MWE token is missing
      or replaced in the prediction compared to raw_text.

    Args:
        item: Dictionary containing:
            - "text": sentence with MWE annotated as [[ ... ]]
            - "raw_text": original unannotated sentence
            - "prediction": system-generated rephrasing

    Returns:
        True if at least one token from the MWE has been deleted or replaced.
        False otherwise.
    """

    # Extract MWE tokens
    mwe_tokens = extract_mwe_tokens(item["text"], is_JA)
        
    # Tokenize original and prediction
    if is_JA:
        original_tokens = [str(tok) for tok in ja_tokenizer(item["raw_text"])]
        prediction_tokens = [str(tok) for tok in ja_tokenizer(item["prediction"])]
    else:
        original_tokens = item["raw_text"].split()
        prediction_tokens = item["prediction"].split()
            
    # Use SequenceMatcher to find deletions/replacements
    matcher = difflib.SequenceMatcher(None, original_tokens, prediction_tokens)
    removed_tokens = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            for token in original_tokens[i1:i2]:
                removed_tokens.append(
                        re.sub(
                                r"[^\p{L}\p{N}\p{M}\p{Join_Control}'’`\-‐-‒–—―\u05BE]+",  # M = marks (accents, diacritiques)
                                "",
                                token
                        )
                )
    # Check if any removed token belongs to the MWE
    return any(token in mwe_tokens for token in removed_tokens)

################################################################################
def load_json(path: str) -> List[Dict[str, str]]:
    """Load a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

################################################################################
def merge_gold_and_predictions(
    gold_data: List[Dict[str, str]],
    pred_data: List[Dict[str, str]]
) -> Dict[str, Dict[str, str]]:
    """
    Merge gold references and predictions into a single dictionary keyed by source_sent_id.

    Args:
        gold_data: List of gold reference entries.
        pred_data: List of prediction entries.

    Returns:
        Dictionary mapping source_sent_id to combined entry.
    """
    # Transform data to have a dict with source_sent_id as the key
    merged = {item.pop("source_sent_id"): item for item in gold_data}

    for pred_item in pred_data:
        sent_id = pred_item.pop("source_sent_id")
        if sent_id not in merged:
            raise ValueError(f"Sentence ID '{sent_id}' in prediction file not found in gold references.")
        merged[sent_id]["prediction"] = pred_item["prediction"]

    return merged

################################################################################
def extract_references_and_predictions(
    data: Dict[str, Dict[str, str]],
    is_JA: bool
) -> Tuple[List[str], List[str], List[str], List[str], List[bool]]:
    """
    Extract creative and minimal references, predictions, and a mask for valid evaluation.

    Args:
        data: Dictionary mapping sentence IDs to gold + prediction data.

    Returns:
        Tuple of:
            - predictions: list of predicted sentences
            - references_creative: list of creative references
            - references_minimal: list of minimal references
            - original_texts: list of original texts
            - mask: boolean list indicating valid entries (MWE deleted)
    """
    predictions, references_creative, references_minimal, original_texts, mask = [], [], [], [], []

    for _, item in data.items():
        labels = item.get("label", [])
        prediction = item["prediction"]
        original_text = item["raw_text"]
        
        # Ensure no incomplete annotations
        if any("[**TODO**]" in label for label in labels):
            raise ValueError(f"Incomplete annotation detected: {item}")

        reference_creative = None
        reference_minimal = None

        # Extract labeled references
        for label in labels:
            if label.startswith("Creative:"):
                reference_creative = label[len("Creative:"):].strip() or None
            elif label.startswith("Minimal:"):
                reference_minimal = label[len("Minimal:"):].strip() or None
            else:
                raise ValueError(f"Unexpected label format: {labels}")

        # Fallback if one reference is missing
        if not reference_creative or not reference_minimal:
            if not reference_minimal and reference_creative:
                reference_minimal = reference_creative
            elif not reference_creative and reference_minimal:
                reference_creative = reference_minimal
            else:
                raise ValueError(f"Missing both Creative and Minimal references: {item}")

        predictions.append(prediction)
        references_creative.append(reference_creative)
        references_minimal.append(reference_minimal)
        original_texts.append(original_text)
        
        # Penalize predictions that still contain the original MWE
        mask.append(has_mwe_been_deleted(item, is_JA))
        
    return predictions, references_creative, references_minimal, original_texts, mask

################################################################################
def evaluate_performance(predictions: List[str], references_creative: List[str], references_minimal: List[str], mask: List[bool]) -> None:
    """
    Evaluate predictions against both creative and minimal references using BERTScore.

    Args:
        predictions: List of predicted sentences.
        references_creative: List of creative gold references.
        references_minimal: List of minimal gold references.
        mask: Boolean list indicating which sentences to include in the score.
    """
    # Apply mask
    predictions = list(compress(predictions, mask))
    references_creative = list(compress(references_creative, mask))
    references_minimal = list(compress(references_minimal, mask))

    # Compute BERTScore for both creative and minimal
    P_creat, R_creat, F1_creat = score(predictions, references_creative, lang="fr")
    P_min, R_min, F1_min = score(predictions, references_minimal, lang="fr")

    # Take the better score for each sentence
    best_scores = [max(F1_creat[i], F1_min[i]) for i in range(len(F1_min))]

    # Compute final metrics
    avg_score = 100 * sum(best_scores) / float(len(best_scores) + (len(mask) - sum(mask)))
    zero_score_count = len(mask) - sum(mask)

    print("\n\nPERFORMANCE :")
    print(f"Average f-bertscore for the current system: {avg_score:.2f}")
    print(f"Number of evaluated elements: {len(best_scores)}")
    print(f"Number of elements with a 0 score (MWE not deleted): {zero_score_count}")

################################################################################
def compute_diversities(texts: List[str], original_texts: List[str]) -> Tuple[float, float, float]:
    """
    Compute diversity metrics for a given list of texts.

    The function counts the frequency of each token across all texts,
    creates a diversity graph using `diversutils`, and computes:
      - Shannon-Weaver entropy
      - Richness (variety)
      - Shannon evenness (balance)

    Args:
        texts (List[str]): A list of strings, each representing a text.
        original_texts (List[str]): A list of strings, each representing a text.

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - entropy (float): Shannon-Weaver entropy
            - variety (float): Richness index
            - balance (float): Evenness index
    """
    # Count token occurrences across all texts
    token_counter = Counter()

    
    for i in range(len(texts)):
        original_tokens = original_texts[i].split()
        text_tokens = texts[i].split()

        matcher = difflib.SequenceMatcher(None, original_tokens, text_tokens)
        diff_tokens = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':  # Keep only words that changed
                token_counter.update(text_tokens[j1:j2])

    # Create an empty diversity graph (parameters 0, 0: for init)
    graph_index = diversutils.create_empty_graph(0, 0)

    # Add each token as a node with its occurrence count
    for category_name, element_count in token_counter.items():
        diversutils.add_node(graph_index, element_count)

    diversutils.compute_relative_proportion(graph_index)

    entropy, hill_number = diversutils.individual_measure(graph_index, diversutils.DF_ENTROPY_SHANNON_WEAVER)
    variety = diversutils.individual_measure(graph_index, diversutils.DF_INDEX_RICHNESS)[0]
    balance = diversutils.individual_measure(graph_index, diversutils.DF_INDEX_SHANNON_EVENNESS)[0]

    return entropy, variety, balance

################################################################################
def evaluate_diversity(predictions: List[str], references_creative: List[str], references_minimal: List[str], original_texts: List[str]) -> None:
    """
    Evaluate the diversity (entropy, variety and balance) of the predictions and the references.

    Args:
        predictions: List of predicted sentences.
        references_creative: List of creative gold references.
        references_minimal: List of minimal gold references.
        original_texts: List of original texts before rephrasing.
        mask: Boolean list indicating which sentences to include in the score.
    """
    # UNCOMMENT if we want to ignore the sentences where the MWEs have not been removed
    # # Apply mask
    # predictions = list(compress(predictions, mask))
    # references_creative = list(compress(references_creative, mask))
    # references_minimal = list(compress(references_minimal, mask))

    entropy_pred, variety_pred, balance_pred = compute_diversities(predictions, original_texts)
    entropy_mini, variety_mini, balance_mini = compute_diversities(references_minimal, original_texts)
    entropy_creative, variety_creative, balance_creative = compute_diversities(references_creative, original_texts)
    
    print("\n\nDIVERSITY :")
    print(f"Entropy, variety, balance for the current system:\t {entropy_pred:.3f}, {variety_pred:.0f}, {balance_pred:.3f}")
    print(f"Entropy, variety, balance for the minimal reference:\t {entropy_mini:.3f}, {variety_mini:.0f}, {balance_mini:.3f}")
    print(f"Entropy, variety, balance for the creative reference:\t {entropy_creative:.3f}, {variety_creative:.0f}, {balance_creative:.3f}")
    
################################################################################
def main() -> None:
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate system predictions for PARSEME 2.0 Subtask 2 (B)")
    parser.add_argument("gold_path", help="Path to the file containing the gold references")
    parser.add_argument("predictions_path", help="Path to the file containing the predictions")
    parser.add_argument("--is_JA", help="Indicate that the file is in Japanese and must use a JA tokenizer", action='store_true')
    args = parser.parse_args()

    gold_data = load_json(args.gold_path)
    pred_data = load_json(args.predictions_path)

    print("--------------------------------------",args.predictions_path,"--------------------------------------")
    
    merged_data = merge_gold_and_predictions(gold_data, pred_data)
    predictions, references_creative, references_minimal, original_texts, mask = extract_references_and_predictions(merged_data, args.is_JA)
    evaluate_performance(predictions, references_creative, references_minimal, mask)
    evaluate_diversity(predictions, references_creative, references_minimal, original_texts)

################################################################################
if __name__ == "__main__":
    main()
