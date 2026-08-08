# Audio Classification Neural Network From Scratch

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Deep_Learning-013243.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791.svg?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED.svg?logo=docker&logoColor=white)

Este projeto visou construir uma Rede Neural Multi-Layer Perceptron (MLP) do zero e aplicá-la para classificação de áudios. Ao invés de se utilizar frameworks de alto nível como PyTorch ou TensorFlow, a Rede Neural foi desenvolvida puramente em NumPy, implementando algoritmos como Forward Propagation (com e sem Dropout), Backpropagation (padrão e com algoritmos de otimização) e algoritmos de otimização de aprendizagem como Adam e learning rate decay.
 
O modelo foi treinado, inicialmente, para reconhecer 5 classes: vento, corvo, passos, máquina de lavar e motor. Porém, pode ser adaptado para reconhecer mais classes ou mudar as que tem, refazendo seu treinamento e gerando novos parâmetros. O número de neurônios e layers também pode ser definido de forma dinâmica, alterando os valores correspondentes no script `train.py`.
 
A inferência do modelo é exposta via uma **API REST com FastAPI**, que extrai as features do áudio (MFCCs) em tempo real, faz a predição e **salva os logs (com latência e nível de confiança) em um banco de dados PostgreSQL**, tudo orquestrado via **Docker e Docker Compose**.

---

## Funcionalidades

- **Engine Customizada (NumPy):** Construção da rede neural profunda, cálculo de perdas (*loss*), otimizadores e camadas do zero.
- **Modelo Re-treinável e Adaptável:** A arquitetura da biblioteca foi pensada de forma que o modelo possa ser treinado novamente ou, com adaptação do código em `train.py`, ser treinado para outras tarefas.
- **Processamento de Áudio:** Extração de 40 coeficientes MFCC por áudio (média e desvio padrão dos MFCCs originais para redução de dimensionalidade) usando a biblioteca `librosa`.
- **API Rápida e Documentada:** FastAPI gerando documentação automática via Swagger UI.
- **Integração de Banco de Dados:** Logging em tempo real de inferências e métricas usando SQLAlchemy e PostgreSQL.
- **Ambiente Containerizado:** Arquitetura pronta para produção com Docker e multi-stage build.

---
 
## Estrutura do Projeto
 
```text
audio-ml-from-scratch/
├── api/                    # Código do FastAPI e integração com DB
│   ├── main.py              # Endpoints da API (/predict, /)
│   ├── database.py          # Conexão com o PostgreSQL
│   ├── models_db.py         # Schemas do SQLAlchemy (PredictionLog)
│   └── schemas.py           # Pydantic schemas para validação
├── artifacts/               # Artefatos salvos (.npz, scalers)
├── data/                    # Datasets (ex: esc50.csv) e áudios brutos
│   ├── raw/                 # Onde os áudios devem ser guardados
│   └── esc50.csv            # Mapeamento dos áudios para seus labels
├── docs/                    # Documentação teórica (Backprop, Forwardprop, MFCCs)
├── engine/                  # Módulos matemáticos da Rede Neural (NumPy)
│   ├── layers.py            # Forward, Backward, Dropout
│   ├── loss.py               # Funções de custo e ativação
│   ├── model.py              # Arquitetura do Modelo
│   └── optimizers.py         # Atualização de pesos (Adam, SGD, etc)
├── pipeline/                 # Processamento de dados
│   └── audio_processor.py    # Extração de MFCCs e mapeamento de classes
├── requirements/              # Pasta que contém os arquivos de dependências
│   ├── dev.txt                # Dependências para desenvolvimento
│   ├── prod.txt               # Dependências para produção
│   └── test.txt               # Dependências para teste
├── tests/
│   ├── conftest.py            # Arquivo que centraliza a configuração dos testes
│   ├── test_api.py            # Testes para os endpoints e funções da API
│   └── test_engine.py         # Testes para os elementos do modelo
├── .dockerignore               # Arquivo que descreve tudo a ser ignorado pelo Docker
├── .env.example                 # Exemplo das variáveis de ambiente que precisam ser criadas
├── .gitignore                   # Arquivo que descreve tudo a ser ignorado pelo git
├── train.py                     # Script principal para treinar o modelo
├── Dockerfile                   # Configuração de build da API
└── docker-compose.yml           # Orquestração do PostgreSQL + API
```
 
---

## Pré-requisitos
 
Para rodar este projeto na sua máquina, você precisa ter:
 
- [Docker](https://www.docker.com/) e [Docker Compose](https://docs.docker.com/compose/) instalados.
- (Opcional para Dev) Python 3.11+ e `virtualenv`.
---
 
## Como Executar

### Via Docker (Recomendado)

O `docker-compose.yml` subirá simultaneamente o banco de dados PostgreSQL e a API FastAPI.

> **Nota:** Certifique-se de que o **Docker Desktop** esteja aberto e rodando antes de executar os comandos.

**1. Configurar Variáveis de Ambiente**

Crie um arquivo `.env` na raiz do projeto (utilize o `.env.example` como base) com as credenciais do banco:

```env
POSTGRES_USER=audio_admin
POSTGRES_PASSWORD=secure_password_98765
POSTGRES_DB=audio_classifier_db
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}
```

**2. Levantar a Aplicação**

No terminal, execute:

```bash
docker-compose up -d --build
```

**3. Acessar a API**

- **Documentação Swagger UI:** [http://localhost:8080/docs](http://localhost:8080/docs)
- A API tentará carregar automaticamente os pesos salvos do modelo localizados em `artifacts/trained_model_params.npz`. Se o arquivo não existir, execute o script de treinamento primeiro.

---

### Execução Local (Ambiente de Desenvolvimento)

Caso queira rodar o projeto diretamente na sua máquina local sem o Docker:

**1. Criar e ativar o ambiente virtual:**

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate
```

**2. Instalar as dependências de desenvolvimento:**

```bash
pip install -r requirements/dev.txt
```

**3. Iniciar o servidor de desenvolvimento com Uvicorn:**

```bash
uvicorn api.main:app --reload --port 8080
```

---

**4. Acessar a API:**

- **Documentação Swagger UI:** [http://localhost:8080/docs](http://localhost:8080/docs)
- A API tentará carregar automaticamente os pesos salvos do modelo localizados em `artifacts/trained_model_params.npz`. Se o arquivo não existir, execute o script de treinamento primeiro.

---

## Treinando o Modelo
 
Caso queira treinar o modelo para outras classes ou fazer modificações, siga os seguintes passos:
 
**1. Instale as dependências locais:**
 
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements/dev.txt
```
 
**2. Baixe e posicione os dados brutos**
 
Antes de rodar o treinamento, é **necessário baixar os áudios do dataset** (ESC-50) e colocá-los dentro da pasta `raw` (ex: `data/raw/audio`, conforme mapeado no seu `.env`). O script falhará caso não encontre os arquivos de som para extrair as características.
 
**3. Personalize as classes (Opcional)**
 
Caso queira treinar o modelo para reconhecer classes diferentes das originais, defina as classes alvo em `audio_processor.py` antes de rodar o treinamento.
 
**4. Execute o pipeline de treino:**
 
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
  "filename": "vento_forte.wav",
  "prediction": "wind",
  "confidence": "96.00%",
  "latency_ms": 12.35,
  "probabilities_pct": {
    "wind": 96.00,
    "crow": 1.50,
    "engine": 1.50,
    "footsteps": 0.50,
    "washing_machine": 0.50
  }
}
```
 
---
 
## Autor
 
Desenvolvido por **Leandro Carvalho**. Este projeto é de código aberto e foi criado para fins educacionais de aprofundamento na matemática do Deep Learning e ML Engineering.
