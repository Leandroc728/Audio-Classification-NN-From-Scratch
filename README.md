# Audio Classification Neural Network From Scratch

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Deep_Learning-013243.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791.svg?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED.svg?logo=docker&logoColor=white)

Um projeto completo que visou construir uma Rede Neural Multi-layer Perceptron do zero e aplicá-la para classificação de áudios. Em vez de utilizar frameworks de alto nível como PyTorch ou TensorFlow, a **Rede Neural foi desenvolvida puramente em NumPy**, implementando algoritmos como *Forward Propagation* (com Dropout), *Backpropagation* e métodos de otimização matemáticos.

A inferência do modelo é exposta via uma **API REST com FastAPI**, que extrai as features do áudio (MFCCs) em tempo real, faz a predição e **salva os logs (com latência e nível de confiança) em um banco de dados PostgreSQL**, tudo orquestrado via **Docker e Docker Compose**.

---

## Funcionalidades

- **Engine Customizada (NumPy):** Construção da rede neural profunda, cálculo de perdas (*loss*), otimizadores e camadas do zero.
- **Processamento de Áudio:** Extração de 40 coeficientes MFCC usando `librosa`.
- **API Rápida e Documentada:** FastAPI gerando documentação automática via Swagger UI.
- **Integração de Banco de Dados:** Logging em tempo real de inferências e métricas usando SQLAlchemy e PostgreSQL.
- **Ambiente Containerizado:** Arquitetura pronta para produção com Docker e multi-stage build.

---

## Estrutura do Projeto

```text
audio-ml-from-scratch/
├── api/                  # Código do FastAPI e integração com DB
│   ├── main.py           # Endpoints da API (/predict, /)
│   ├── database.py       # Conexão com o PostgreSQL
│   ├── models_db.py      # Schemas do SQLAlchemy (PredictionLog)
│   └── schemas.py        # Pydantic schemas para validação
├── artifacts/            # Artefatos salvos (.npz, scalers)
├── data/                 # Datasets (ex: esc50.csv) e áudios brutos
├── docs/                 # Documentação teórica (Backprop, Forwardprop, MFCCs)
├── engine/               # Módulos matemáticos da Rede Neural (NumPy)
│   ├── layers.py         # Forward, Backward, Dropout
│   ├── loss.py           # Funções de custo e ativação
│   ├── model.py          # Arquitetura do Modelo
│   └── optimizers.py     # Atualização de pesos (Adam, SGD, etc)
├── pipeline/             # Processamento de dados
│   └── audio_processor.py# Extração de MFCCs e mapeamento de classes
├── requirements/         # Pasta que contém os arquivos de dependências
│   ├── dev.txt           # Dependências para desenvolvimento
│   ├── prod.txt          # Dependências para produção
│   └── test.txt          # Dependências para teste
├── tests/                # Testes automatizados (Pytest)
├── train.py              # Script principal para treinar o modelo
├── Dockerfile            # Configuração de build da API
└── docker-compose.yml    # Orquestração do PostgreSQL + API
```

---

## Pré-requisitos

Para rodar este projeto na sua máquina, você precisa ter:

- [Docker](https://www.docker.com/) e [Docker Compose](https://docs.docker.com/compose/) instalados.
- (Opcional para Dev) Python 3.11+ e `virtualenv`.

---

## Como Executar

O `docker-compose.yml` subirá simultaneamente o banco de dados PostgreSQL e a API do FastAPI.

**1. Configurar Variáveis de Ambiente**

Crie um arquivo `.env` na raiz do projeto (use o `.env.example` como base) com as credenciais do banco:

```env
POSTGRES_USER=audio_admin
POSTGRES_PASSWORD=secure_password_98765
POSTGRES_DB=audio_classifier_db
DATABASE_URL=postgresql://audio_admin:secure_password_98765@postgres-db:5432/audio_classifier_db
```

**2. Levantar a Aplicação**

No terminal, execute:

```bash
docker compose up -d --build
```

**3. Acessar a API**

- **Documentação Swagger UI:** [http://localhost:8080/docs](http://localhost:8080/docs)
- A API tentará carregar automaticamente os pesos do modelo (`trained_model_params.npz`). Se o arquivo não existir, será necessário rodar o script de treinamento primeiro.

---

## Treinando o Modelo

Se você clonou o repositório e precisa gerar os arquivos `.npz` da rede neural localmente:

**1. Instale as dependências locais:**

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements/dev.txt
```

**2. Baixe e posicione os dados brutos ⚠️**

Antes de rodar o treinamento, é **necessário baixar os áudios do dataset** (ESC-50) e colocá-los dentro da pasta `raw` (ex: `data/raw/audio`, conforme mapeado no seu `.env`). O script falhará caso não encontre os arquivos de som para extrair as características.

**3. Execute o pipeline de treino:**

```bash
python train.py
```

*Este comando lerá os áudios na pasta `raw`, extrairá as MFCCs, rodará as epochs em NumPy, e salvará os pesos em `trained_model_params.npz` na raiz e os metadados de normalização em `scaler.npz` na pasta `artifacts/`.*

---

## Endpoints da API

| Método | Rota | Descrição |
| :--- | :--- | :--- |
| `GET` | `/` | Retorna o status de saúde (Health Check) e as dimensões (layers) carregadas pelo modelo. |
| `POST` | `/predict` | Recebe um arquivo `.wav` ou `.mp3`, extrai MFCCs, passa pela rede neural (Forward Propagation), salva a métrica no **Postgres** e retorna a predição. |

**Exemplo de Resposta de `/predict`:**

```json
{
  "filename": "cachorro_latindo.wav",
  "prediction": "Dog",
  "confidence": "96.40%",
  "latency_ms": 12.35,
  "probabilities_pct": {
    "Dog": 96.40,
    "Cat": 2.15,
    "Siren": 1.45
  }
}
```

---

## Licença

Desenvolvido por **Leandro Carvalho**. Este projeto é de código aberto e foi criado para fins educacionais de aprofundamento na matemática do Deep Learning e ML Engineering.
