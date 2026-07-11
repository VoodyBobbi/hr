FROM python:3.11-slim

# HF Spaces запускает контейнер под UID 1000 — создаём такого пользователя заранее,
# иначе os.makedirs() в assistant.py/candidates.py/logger.py упадёт по правам доступа.
RUN useradd -m -u 1000 user
WORKDIR /app

# КРИТИЧНО: сначала ставим CPU-only torch с официального индекса PyTorch.
# Без этой строки sentence-transformers потянет обычный torch и вместе с ним
# ~2.5-3 ГБ пакетов nvidia-cuda-*/cudnn/triton, которые на CPU бесполезны
# и легко упрутся в лимиты сборки на бесплатном хостинге.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
RUN chmod +x start.sh

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Hugging Face Docker Spaces по умолчанию ждут приложение на порту 7860.
EXPOSE 7860

CMD ["./start.sh"]
