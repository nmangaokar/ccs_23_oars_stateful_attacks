import os
import sys
import argparse
import parse
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from icecream import ic
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default=None)

    args = parser.parse_args()
    logs = sorted([x for x in os.listdir(args.log_path) if 'log' in x])

    aggregate_log = []
    for log in logs:
        path = os.path.join(args.log_path, log)
        log_lines = open(path).readlines()
        log_lines = [x for x in log_lines if 'FINISHED' not in x]
        log_lines = [x for x in log_lines if 'INFO:root:True Label' in x]
        aggregate_log.extend(log_lines)
    print("Length of log file: {}".format(len(aggregate_log)))

    y_true, y_pred = [], []
    queries = []
    cache_hits = []
    for idx, line in enumerate(aggregate_log):
        format_str = "INFO:root:True Label : {} | Predicted Label : {} | Cache Hits / Total Queries : {} / {}"
        parsed = parse.parse(format_str, line)
        y_true.append(int(parsed[0]))
        y_pred.append(int(parsed[1]))
        cache_hits.append(int(parsed[2]))
        queries.append(int(parsed[3]))
    accuracy = len([i for i in range(len(y_true)) if queries[i] != 0]) * 100 / len(y_true)
    adv_accuracy = accuracy_score(y_true, y_pred) * 100
    adv_macro_f1 = f1_score(y_true, y_pred, average='macro') * 100
    attack_success = len([i for i in range(len(y_true)) if y_true[i] != y_pred[i] and queries[i] > 0]) * 100 / len(
        [i for i in range(len(y_true)) if queries[i] > 0])
    avg_queries = np.mean([queries[i] for i in range(len(queries)) if y_true[i] != y_pred[i] and queries[i] > 0])
    std_queries = np.std([queries[i] for i in range(len(queries)) if y_true[i] != y_pred[i] and queries[i] > 0])
    avg_cache_hits = np.mean(
        [cache_hits[i] for i in range(len(cache_hits)) if y_true[i] != y_pred[i] and queries[i] > 0])
    std_cache_hits = np.std(
        [cache_hits[i] for i in range(len(cache_hits)) if y_true[i] != y_pred[i] and queries[i] > 0])
    avg_queries_if_account_bans = np.mean(
        [queries[i] + cache_hits[i] for i in range(len(cache_hits)) if y_true[i] != y_pred[i] and queries[i] > 0])
    std_queries_if_account_bans = np.std(
        [queries[i] + cache_hits[i] for i in range(len(cache_hits)) if y_true[i] != y_pred[i] and queries[i] > 0])
    attack_success_per_ban_budget = {}

    ic(accuracy, adv_accuracy, adv_macro_f1, attack_success, avg_queries, std_queries, avg_cache_hits, std_cache_hits, avg_queries_if_account_bans)
