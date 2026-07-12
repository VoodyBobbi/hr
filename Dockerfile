FROM python:3.11-slim
WORKDIR /app

# КРИТИЧНО: сначала ставим CPU-only torch с официального индекса PyTorch.
# Без этой строки sentence-transformers потянет обычный torch и вместе с ним
# ~2.5-3 ГБ пакетов nvidia-cuda-*/cudnn/triton, которые на CPU бесполезны
# и упрутся в лимиты сборки на бесплатном хостинге.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x start.sh

# Render сам передаёт порт через переменную окружения PORT — не хардкодим порт здесь.
CMD ["./start.sh"]
