FROM python:3.12-slim-bookworm

RUN mkdir -p /usr/src/

WORKDIR /usr/src/

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME /data

# ensure the code is working before the simulation
RUN ./scripts/run_tests.sh 

CMD ["./scripts/run_simulation.sh"]
