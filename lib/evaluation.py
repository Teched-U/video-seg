#!/usr/bin/env python3
from typing import Dict, Tuple

# Threshold of 10 seconds to consider match
THRESHOLD = 10




def evaluate(gt : Dict, pred : Dict) -> Tuple[float, float, float]: 
    """
    Format of the gt and  pred
    {
        "<ts>": "<topic summary>",
        ...
    }

    Returns:
        precition, recall, Fscore
    """

    print(f"Ground Truth:{gt}")
    print(f"Prediction :{pred}")

    # Calculate number of hits: overlapping starting point
    gt_ts_arr = sorted(map(int, gt.keys()))
    pred_ts_arr = sorted(map(int, pred.keys()))

    hits = 0
    # Naive matching without repetitive match
    for gt_ts in gt_ts_arr:
        for pred_ts in pred_ts_arr:
            if abs(pred_ts - gt_ts) <= THRESHOLD:
                hits += 1
                break

    precision = float(hits/len(pred)) 
    recall = float(hits/len(gt))

	if(precision > 0 or recall > 0):
		fscore = 2* float((precision * recall) / (precision + recall))
	else:
		fscore = 0
	print(precision, recall, fscore)
    
	return precision, recall, fscore


