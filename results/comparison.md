# Comparison table (train)

Dataset: `CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv` · 80/20 stratified, random_state=42 · train n=806998

| Model | Prompt | Accuracy | Precision | Recall | F1 |
|-------|--------|----------|-----------|--------|-----|
| GPT-5.5 (EN prompt) | `prompts/prompt_en.md` | 0.957410 | 0.926310 | 0.999907 | 0.961702 |
| GPT-5.5 (RU prompt) | `prompts/prompt_ru.md` | 0.957410 | 0.926308 | 0.999910 | 0.961703 |
| Anthropic Claude 4.6 Opus (EN prompt) | `prompts/prompt_en.md` | 0.956976 | 0.925594 | 0.999933 | 0.961328 |
| Anthropic Claude 4.6 Opus (RU prompt) | `prompts/prompt_ru.md` | 0.956973 | 0.925586 | 0.999935 | 0.961325 |
| Yandex Alice AI (EN prompt) | `prompts/prompt_en.md` | 0.957410 | 0.926302 | 0.999917 | 0.961703 |
| Yandex Alice AI (RU prompt) | `prompts/prompt_ru.md` | 0.957410 | 0.926302 | 0.999917 | 0.961703 |
| Hand-coded baseline | `N/A (expert implementation)` | 0.957030 | 0.925681 | 0.999930 | 0.961374 |
