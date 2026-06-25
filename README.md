# Photo Filter for Database

Фильтр фотографий для проверки качества и соответствия перед добавлением в базу данных. Двухэтапная система: быстрая проверка через OpenCV + финальная валидация через мультимодальную нейросеть.

## Что проверяется

### Критические проблемы (автоматический отказ):
- **Фото документа** — паспорт, удостоверение, медицинская карта с вклеенным фото
- **Неправильная ориентация** — поворот на 90° или 180°
- **Маленькое лицо** — лицо занимает меньше 15% кадра
- **Захламленный фон** — полки, коробки, техника, документы, беспорядок

### Хорошее фото:
- Лицо четко видно, занимает значительную часть кадра
- Нормальная вертикальная ориентация
- Простой фон (стена, размытие, улица без мусора)
- Это обычный портрет человека, а не фото бумаги/документа
- На фото сбалансирован свет и нет пересветов, мешающих увидеть лицо

## Установка

### 1. Зависимости

```bash
pip install openai opencv-python numpy
```

### 2. Конфигурация

Пропишите ключ OPEN_AI_API b BASE_API_URL в настройках клиента

```python
client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="YOUR_API_URL",
    max_retries=1,
    timeout=300.0,
)
```

### 3. Требования к серверу

Проект использует OpenAI-совместимый API с мультимодальной моделью. Поддерживаются:
- vLLM
- Ollama
- TGI (Text Generation Inference)
- Любой сервер с поддержкой vision-моделей

Рекомендуемая модель: `gemma-4-e4b-it` или аналог с поддержкой изображений.

## Использование

### Базовое использование

```python
from filter import check_photo_for_database

result = check_photo_for_database("path_to_photo.jpg")
print(result)
```

### Пример ответа

**Хорошее фото:**
```json
{
    "is_valid": true,
    "score": 92,
    "errors": [],
    "recommendations": "",
    "opencv_debug": {
        "is_rotated": false,
        "is_document": false,
        "face_ratio": 0.35,
        "background_clutter": 0.12
    }
}
```

**Плохое фото:**
```json
{
    "is_valid": false,
    "score": 20,
    "errors": ["document_photo", "cluttered_background"],
    "recommendations": "Сделайте обычный портрет: лицо крупно, вертикально, на простом фоне без документов и вещей.",
    "opencv_debug": {
        "is_rotated": false,
        "is_document": true,
        "face_ratio": 0.08,
        "background_clutter": 0.67
    }
}
```


## Архитектура

### Этап 1: OpenCV Prefilter (быстрая проверка)

Проверяет без вызова LLM:
1. **Ориентация** — соотношение ширины и высоты
2. **Признаки документа** — поиск прямоугольных контуров с резкими краями
3. **Размер лица** — через Haar Cascade Classifier
4. **Шум фона** — плотность рёбер вне области лица

Если найдены критические проблемы → сразу отказ, LLM не вызывается.

### Этап 2: LLM Validation (финальная проверка)

Вызывается только если OpenCV не нашёл явных проблем. Анализирует:
- Контекст фото (документ или портрет)
- Качество фона
- Общее соответствие критериям

## Настройка порогов

В функции `opencv_prefilter()` можно настроить чувствительность:

```python
# Поворот: ширина > высоты * коэффициент
is_rotated = w > h * 1.3  # Увеличить для более строгой проверки

# Документ: площадь контура
if len(approx) == 4 and area > (w * h * 0.3) and area < (w * h * 0.95):
    is_document = True

# Маленькое лицо: порог face_ratio
if prefilter["face_ratio"] < 0.15:  # Уменьшить для более мягкой проверки
    hard_errors.append("small_face")

# Шум фона: порог background_clutter
if prefilter["background_clutter"] > 0.5:  # Увеличить для более мягкой проверки
    hard_errors.append("cluttered_background")
```

Поле `opencv_debug` в ответе помогает калибровать пороги под ваши данные.

## Интеграция с FastAPI

```python
from fastapi import FastAPI, UploadFile, HTTPException
from filter import check_photo_for_database
import tempfile
import os

app = FastAPI()

@app.post("/validate-photo")
async def validate_photo(file: UploadFile):
    # Сохранение файла во временную директорию
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        result = check_photo_for_database(tmp_path)
        return result
    finally:
        os.unlink(tmp_path)  # Удаление временного файла
```

## Требования к изображениям

- **Форматы**: JPEG, PNG, WEBP
- **Размер**: рекомендуется не более 10 MB
- **Разрешение**: минимум 640x480 для корректного детектирования лица

## Возможные проблемы

### Модель возвращает не JSON
Парсер `extract_json_from_text()` автоматически удаляет markdown-разметку (```json ... ```). Если проблема сохраняется, проверьте промпт.

### Ошибка `response_format.type must be 'json_schema' or 'text'`
Используйте `response_format={"type": "text"}` вместо `json_object`. Парсер JSON справится с форматированием.

### OpenCV не детектирует лицо
Haar Cascade работает не идеально. Для боковых профилей или закрытых лиц может не сработать. В таких случаях LLM всё равно проверит фото на втором этапе.

### Ложные срабатывания на документы
Если фильтр слишком строго отсекает нормальные фото, уменьшите порог `background_clutter` или отключите проверку `is_document`.
