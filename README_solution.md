# Решение по ДЗ — SDXL LoRA

Что подготовлено:
- заполнен `lora_code/model/lora.py`
- заполнен `lora_code/trainer_sdxl.py`
- заполнен `lora_code/inferencer_sdxl.py`
- добавлен `inference.sh`
- исправлен `train_lora.sh`
- добавлен шаблон `metrics_template.py`

## Что осталось сделать вам
1. Положить 3–5 фото в `my_dataset/`.
2. При желании заменить `placeholder_token` и `class_name` в `train_lora.sh`.
3. Установить зависимости: `pip install -r requirements.txt`.
4. Запустить обучение: `bash train_lora.sh`.
5. По валидации выбрать лучший чекпойнт и подставить его в `inference.sh`.
6. Запустить инференс: `bash inference.sh`.
7. Подсчитать метрики шаблоном `metrics_template.py`.

## Набор из 10 промптов для отчёта
1. `a photo of {0}`
2. `portrait photo of {0}, studio lighting, 85mm`
3. `{0} in a black business suit in a modern office`
4. `{0} on a tropical beach at sunset`
5. `{0} in a snowy forest, winter clothes`
6. `{0} riding a bicycle in Amsterdam`
7. `{0} as a fantasy wizard, cinematic light`
8. `{0} in anime style`
9. `oil painting portrait of {0}`
10. `{0} in a cyberpunk city at night, neon lights`

## Краткое обоснование
- 1–2: проверка сохранения идентичности без сильной стилизации;
- 3–5: смена аутфита и фона;
- 6: новая поза и композиция;
- 7–10: сильная стилизация и перенос в новые домены.

Для каждого промпта удобно генерировать 4 изображения:
- 1 мало для оценки устойчивости;
- 4 уже показывают разнообразие;
- 8+ заметно дольше и дороже по GPU.
