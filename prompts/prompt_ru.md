# Промпт (русский) — классификация вредоносного сетевого трафика

Используйте этот текст при обращении к русскоязычным или мультиязычным LLM (YandexGPT, GigaChat, ChatGPT с русским запросом и т.д.).

---

## Роль (по желанию)

Ты — опытный Python-разработчик в области сетевой безопасности и систем обнаружения вторжений (IDS).

## Пользовательский промпт

Напиши **полностью готовое к запуску** приложение на Python 3 для **бинарной классификации** сетевого трафика (нормальный / атака) по датасету Kaggle:

**Датасет:** [Network Malware Detection - Connection Analysis](https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis)  
(признаки на уровне соединений, аналог UNSW-NB15)

**Требования:**

1. Загрузка CSV из каталога `data/raw/` (файлы вида `UNSW_NB15_training-set.csv` или любой один `*.csv` в папке).
2. Целевая переменная: столбец `label` (0 — норма, 1 — атака; также в качестве 0 могут быть значения "benign", "normal", "0", "false", а в качестве 1 могут быть значения "malicious", "attack", "1", "true"). Исключить `id`, `attack_cat` и IP-поля (`srcip`, `dstip` и т.п.), если есть.
3. Используй Заголовок .csv файла и первые две строки в качестве примера:
ts|uid|id.orig_h|id.orig_p|id.resp_h|id.resp_p|proto|service|duration|orig_bytes|resp_bytes|conn_state|local_orig|local_resp|missed_bytes|history|orig_pkts|orig_ip_bytes|resp_pkts|resp_ip_bytes|tunnel_parents|label|detailed-label
1525879831.015811|CUmrqr4svHuSXJy5z7|192.168.100.103|51524|65.127.233.163|23|tcp|-|2.999051|0|0|S0|-|-|0|S|3|180|0|0|-|Malicious|PartOfAHorizontalPortScan
1525879831.025055|CH98aB3s1kJeq6SFOc|192.168.100.103|56305|63.150.16.171|23|tcp|-|-|-|-|S0|-|-|0|S|1|60|0|0|-|Malicious|PartOfAHorizontalPortScan

4. Обязательно выполнить предобработку данных.
5. Сделать разбиение в адекватной пропорции на обучающую и тестовую выборку, `stratify=label`, `random_state=const`.
6. Самостоятельно выбрать модель и её основные параметры.
7. На тесте вывести: **accuracy**, **precision**, **recall**, **F1**, **матрицу ошибок**.
8. Сохранить обученный pipeline в `results/model.joblib`.
9. Только библиотеки: `pandas`, `numpy`, `scikit-learn`, `joblib` (без нейросетей).
10. Запуск: `python malware_classifier.py`, параметр `--data-dir` по умолчанию `../../data/raw`.

Верни **один файл** `malware_classifier.py` со всеми импортами и логикой.
