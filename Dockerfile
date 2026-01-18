FROM python:3.11-slim
WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lib/ lib/
COPY model/ model/
COPY main.py .
COPY run.sh .

RUN sed -i 's/\r$//' run.sh && chmod +x run.sh
CMD ["sh", "run.sh"]
