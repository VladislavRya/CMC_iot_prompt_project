# Prompt (English) — Network Malware / Intrusion Classification

Use this prompt as-is when asking an LLM (ChatGPT, Claude, Gemini, etc.) to generate the application.

---

## System / role (optional)

You are an experienced Python developer specializing in network security and intrusion detection systems (IDS).

## User prompt

Write a **fully runnable** Python 3 application for **binary classification** of network traffic (normal / attack) using the Kaggle dataset:

**Dataset:** [Network Malware Detection - Connection Analysis](https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis)  
(connection-level features, similar to UNSW-NB15)

**Requirements:**

1. Load CSV from directory `data/raw/` (files such as `UNSW_NB15_training-set.csv` or any single `*.csv` in that folder).
2. Target column: `label` (0 = normal, 1 = attack; 0 may also appear as `"benign"`, `"normal"`, `"0"`, `"false"`, and 1 as `"malicious"`, `"attack"`, `"1"`, `"true"`). Drop `id`, `attack_cat`, and IP fields (`srcip`, `dstip`, etc.) if present.
3. Use the CSV header and the first two data rows as a format example:
```
ts|uid|id.orig_h|id.orig_p|id.resp_h|id.resp_p|proto|service|duration|orig_bytes|resp_bytes|conn_state|local_orig|local_resp|missed_bytes|history|orig_pkts|orig_ip_bytes|resp_pkts|resp_ip_bytes|tunnel_parents|label|detailed-label
1525879831.015811|CUmrqr4svHuSXJy5z7|192.168.100.103|51524|65.127.233.163|23|tcp|-|2.999051|0|0|S0|-|-|0|S|3|180|0|0|-|Malicious|PartOfAHorizontalPortScan
1525879831.025055|CH98aB3s1kJeq6SFOc|192.168.100.103|56305|63.150.16.171|23|tcp|-|-|-|-|S0|-|-|0|S|1|60|0|0|-|Malicious|PartOfAHorizontalPortScan
```

4. Do the data preprocessing.
5. Do train/test split in adequate proportion, `stratify=label`, `random_state=const`.
6. Choose the model and its main parameters.
7. On the test set, print: **accuracy**, **precision**, **recall**, **F1**, and **confusion matrix**.
8. Save the trained pipeline to `results/model.joblib`.
9. Libraries only: `pandas`, `numpy`, `scikit-learn`, `joblib` (no neural networks).
10. Run via `python malware_classifier.py`; CLI flag `--data-dir` defaulting to `../../data/raw`.

Return **one file** `malware_classifier.py` with all imports and logic.
