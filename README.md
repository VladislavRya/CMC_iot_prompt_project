## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Вариант A: Kaggle (нужен ~/.kaggle/kaggle.json)
# Либо скачать датасет с https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis
pip install kaggle
bash scripts/download_kaggle_data.sh

# Сравнение всех моделей
python scripts/run_comparison.py
```

Отдельный запуск:

```bash
python generated/openai_gpt4o/malware_classifier.py --data-dir data/raw
python baseline/hand_coded_classifier.py --data-dir data/raw
```
