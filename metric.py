#!/usr/bin/env python3
from typing import Dict, Tuple, List
import torch

# Threshold of 10 seconds to consider match
THRESHOLD = 10

def evaluate_metrics(predicted: torch.Tensor, ts : List[float], gt : Dict) -> Tuple[float, float, float]: 
    """
    Format of the gt and  pred
    {
        "<ts>": "<topic summary>",
        ...
    }

    Returns:
        precition, recall, Fscore
    """

    # Convert the prediced result with its orignal timestamps
    predicted = list(predicted)
    pred_ts_arr = sorted(list([t for t, pred in zip(ts, predicted) if pred > 0]))



    # Calculate number of hits: overlapping starting point
    gt_ts_arr = sorted(map(int, gt.keys()))

    hits = 0
    # Naive matching without repetitive match
    for gt_ts in gt_ts_arr:
        for pred_ts in pred_ts_arr:
            if abs(pred_ts - gt_ts) <= THRESHOLD:
                hits += 1
                break


    precision = 0. if len(pred_ts_arr) == 0 else float(hits/len(pred_ts_arr)) 
    recall = float(hits/len(gt_ts_arr))
    
    if precision > 0 or recall > 0:
	    fscore = 2* float((precision * recall) / (precision + recall))
    else:
	    fscore = 0
    return recall, precision, fscore

