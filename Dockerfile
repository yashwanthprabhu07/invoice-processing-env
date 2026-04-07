FROM python:3.11-slim-bullseye

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn pydantic openai groq openenv-core

COPY --chown=user . .

EXPOSE 7860

CMD ["python", "server/app.py"]