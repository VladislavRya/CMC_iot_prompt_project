# Comparison table (test)

Dataset: `CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv` · 80/20 stratified, random_state=42 · **test holdout** n=201750

| Model | Prompt | Accuracy | Precision | Recall | F1 |
|-------|--------|----------|-----------|--------|-----|
| GPT-5.5 (EN prompt) | `prompts/prompt_en.md` | 0.956198 | 0.925025 | 0.999073 | 0.960624 |
| GPT-5.5 (RU prompt) | `prompts/prompt_ru.md` | 0.956203 | 0.925018 | 0.999092 | 0.960629 |
| Anthropic Claude 4.6 Opus (EN prompt) | `prompts/prompt_en.md` | 0.956565 | 0.924970 | 0.999889 | 0.960972 |
| Anthropic Claude 4.6 Opus (RU prompt) | `prompts/prompt_ru.md` | 0.956570 | 0.924978 | 0.999889 | 0.960976 |
| Yandex Alice AI (EN prompt) | `prompts/prompt_en.md` | 0.956213 | 0.925012 | 0.999120 | 0.960639 |
| Yandex Alice AI (RU prompt) | `prompts/prompt_ru.md` | 0.956213 | 0.925012 | 0.999120 | 0.960639 |
| Hand-coded baseline | `N/A (expert implementation)` | 0.956550 | 0.924990 | 0.999833 | 0.960957 |
