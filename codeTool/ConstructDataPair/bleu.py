"""
Borrowed from https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py

Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math


def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)

def code_compute_bleu(code1, code2):
  code1_corpus = [[code1]]
  code2_corpus = [code2]
  result = compute_bleu(code1_corpus, code2_corpus, 4)
  return result[0]
if __name__ == "__main__":
  CodeContent = "a=int(input())\nif a==1 or a==8 or a==15 or a==22 or a==29:\n    print(\"mon\")\nif a==2 or a==9 or a==16 or a==23 or a==30:\n    print(\"tue\")\nif a==3 or a==10 or a==17 or a==24:\n    print(\"wed\")\nif a==4 or a==11 or a==18 or a==25:\n    print(\"thu\")\nif a==5 or a==12 or a==19 or a==26:\n    print(\"fri\")\nif a==6 or a==13 or a==20 or a==27:\n    print(\"sat\")\nif a==7 or a==14 or a==21 or a==28:\n    print(\"sun\")\n"
  CodeContent2 = "a=int(input())\nif a==1 or a==8 or a==15 or a==22 or a==29:\n    print(\"fri\")\nif a==2 or a==9 or a==16 or a==23 or a==30:\n    print(\"sat\")\nif a==3 or a==10 or a==17 or a==24:\n    print(\"sun\")\nif a==4 or a==11 or a==18 or a==25:\n    print(\"mon\")\nif a==5 or a==12 or a==19 or a==26:\n    print(\"tue\")\nif a==6 or a==13 or a==20 or a==27:\n    print(\"wed\")\nif a==7 or a==14 or a==21 or a==28:\n    print(\"thu\")\n"
  reference_corpus = [[CodeContent]]
  translation_corpus = [CodeContent2]
  result = compute_bleu(reference_corpus, translation_corpus, 4)
  print(result)
  print("0100")