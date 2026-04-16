<img width="200" height="100" alt="spyder" src="https://github.com/user-attachments/assets/e789750b-2c86-4222-a4e2-6591898adf59" />
<img width="100" height="200" alt="docker" src="https://github.com/user-attachments/assets/a2219122-397b-4898-a45d-e162d9eb0e2a" />
<img width="50" height="100" alt="excel" src="https://github.com/user-attachments/assets/5211df27-b8ed-4962-914f-c02e78a313b8" />
<img width="150" height="412" alt="amazon" src="https://github.com/user-attachments/assets/88238889-61d2-4195-81d1-31bf921642cc" />


# QuanticXLSX API

> Serviço automatizado de análise de risco de crédito com **Machine Learning** e **Explicabilidade (XAI)**. Processa planilhas financeiras, calcula indicadores, gera scores de crédito e retorna relatórios detalhados em Excel.

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## Funcionalidades Principais

- 📈 Cálculo automático de **indicadores financeiros** (Liquidez, ROIC, Dívida/EBITDA, Cobertura de Juros, etc.)
- 🤖 **Machine Learning** para previsão de probabilidade de default e geração de score (escala 300–850)
- 🔍 **Explicabilidade (XAI)** integrada via SHAP: transparência total nas decisões do modelo
- 🏢 **Arquitetura Multi-tenant**: isolamento lógico por API Key, pronto para SaaS
- ⚡ Endpoints otimizados para integração com dashboards, ERPs e sistemas externos
- 📦 Saída padronizada em Excel com abas organizadas para análise executiva e auditoria

---

## 🛠️ Stack Tecnológica

| Categoria | Tecnologia |
|---|---|
| **Backend** | FastAPI + Uvicorn |
| **Data/ML** | Pandas, NumPy, Scikit-learn, SHAP |
| **Containerização** | Docker (imagem otimizada `slim`) |
| **Cloud (Sugestão)** | AWS ECS Fargate + ECR + ALB |
| **Auth/Config** | Pydantic Settings + Middleware de API Key |

---

## Como Executar

### 1️⃣ Via Docker (Recomendado)
```bash
docker build -t quanticxlsx-api .
docker run -p 8000:8000 -e API_KEYS='{"demo-key":"demo-tenant"}' quanticxlsx-api
```

### 2️⃣ Localmente (Python 3.10+)
```bash
# 1. Ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Dependências
pip install -r requirements.txt

# 3. Iniciar servidor
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
🌐 Acesse a documentação interativa: `http://localhost:8000/docs`

---

## 🔌 Uso da API

### 🔐 Autenticação
Todas as requisições exigem uma API Key válida, passada via:
- **Header:** `x-api-key: sua_chave_aqui`
- **Query:** `?api_key=sua_chave_aqui`

Configure as chaves criando um arquivo `.env`:
```env
API_KEYS={"cliente_a":"tenant_01","cliente_b":"tenant_02"}
```

### 📡 Endpoints

#### `POST /api/v1/analisar`
Processa a planilha e retorna um Excel com os resultados.
```bash
curl -X POST "http://localhost:8000/api/v1/analisar?xai=true" \
  -H "x-api-key: demo-key" \
  -F "file=@dados_empresa.xlsx" \
  --output resultado_quantic.xlsx
```
**Query Parameters:**
| Nome | Tipo | Padrão | Descrição |
|---|---|---|---|
| `xai` | `bool` | `true` | Incluir abas de explicabilidade SHAP no output |
| `api_key` | `string` | - | Autenticação alternativa via query string |

#### `GET /api/v1/explicar/{empresa}`
Retorna a explicação detalhada do score para uma empresa específica.
```json
{
  "empresa": "TechCorp LTDA",
  "score": 720.5,
  "factors": [
    {"feature": "Liquidez_Corrente", "value": 2.1, "impact": 0.15},
    {"feature": "Divida_EBITDA", "value": 3.2, "impact": -0.12}
  ]
}
```

#### `GET /api/v1/health`
Verifica status da API e carregamento do modelo.
```json
{"status":"ok","version":"1.0.0","model":true}
```

---

## 📋 Formato da Planilha de Entrada

A API espera um arquivo `.xlsx` com cabeçalho na primeira linha:

| Coluna | Obrigatória | Tipo | Exemplo |
|---|---|---|---|
| `Empresa` | ✅ | String | `Construtora Alfa SA` |
| `Setor` | ✅ | String | `Construção Civil` |
| `Divida_Liquida` | ✅ | Float | `1500000.00` |
| `EBITDA` | ✅ | Float | `850000.00` |
| `Despesas_Financeiras` | ✅ | Float | `120000.00` |
| `Fluxo_Caixa_Livre` | ✅ | Float | `320000.00` |
| `Divida_Total` | ✅ | Float | `2100000.00` |
| `Ativo_Circulante` | ✅ | Float | `1800000.00` |
| `Passivo_Circulante` | ✅ | Float | `950000.00` |
| `Receita` | ✅ | Float | `5400000.00` |
| `NOPAT` | ✅ | Float | `410000.00` |
| `Capital_Investido` | ✅ | Float | `3200000.00` |
| `Default` | ❌ | `0` ou `1` | `0` *(usado apenas para calibrar o modelo)* |

> ⚠️ Valores não numéricos nas colunas financeiras serão convertidos automaticamente ou tratados como `0`.

### 📤 Estrutura do Arquivo Gerado
A resposta contém um `.xlsx` com as seguintes abas:
1. `Dados_Completos` → Dados originais + indicadores calculados + `credit_score` + `default_probability`
2. `KPIs` → Médias e métricas agregadas do portfólio
3. `Ranking` → Empresas ordenadas do maior para o menor score
4. `Resumo` → Contagem de Alto Risco (`<400`) vs Baixo Risco (`≥400`)
5. `XAI_Importancia` *(opcional)* → Ranking global de variáveis mais impactantes
6. `XAI_Detalhes` *(opcional)* → Fatores individuais que elevaram/reduziram o score de cada empresa

---

## 🏗️ Arquitetura & Design

### 🔐 Multi-tenant
- Identificação por `x-api-key` mapeada para `tenant_id`
- Middleware de validação executa **antes** de qualquer processamento
- Isolamento lógico pronto para expansão (ex: prefixo S3, schemas separados, filas dedicadas)
- Logs estruturados por tenant para auditoria, cobrança e monitoramento

### 🧠 Machine Learning & XAI
- **Modelo:** `GradientBoostingClassifier` (Scikit-learn)
- **Treino:** Automático na primeira requisição ou via coluna `Default` se fornecida
- **Score:** Escala 300–850 (padrão FICO-like). **Maior = menor risco**
- **Explicabilidade:** `shap.TreeExplainer` gera contribuições aditivas por feature
  - Transparência regulatória (LGPD, Bacen, auditorias internas/externas)
  - Endpoint `/explicar/{empresa}` retorna fatores positivos/negativos ranqueados
  - Abas `XAI_*` no Excel facilitam análise executiva sem necessidade de dashboard

---

## ☁️ Deploy em Produção

### Docker
O `Dockerfile` incluído segue boas práticas: imagem `slim`, sem cache pip, variáveis não hardcoded, e expõe apenas a porta `8000`.

### AWS ECS Fargate (Serverless)
1. Crie um repositório no **ECR**
2. Build & Push: `docker push <account>.dkr.ecr.<region>.amazonaws.com/quanticxlsx:latest`
3. Crie uma **Task Definition** com a imagem e mapeie a porta `8000`
4. Deploy em um **Cluster Fargate** com **Application Load Balancer**
5. Configure HTTPS via ACM e Security Groups restritos
6. Injete `API_KEYS` via **AWS Secrets Manager** ou **Parameter Store**

---

## ⚙️ Variáveis de Ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `API_KEYS` | `{"demo-key":"demo-tenant"}` | JSON com mapeamento chave → tenant |
| `MODEL_PATH` | `credit_model.pkl` | Caminho para salvar/carregar o modelo treinado |
| `PYTHONUNBUFFERED` | `1` | Garantir logs em tempo real no container |

---

## 🗺️ Roadmap & Melhorias Futuras
- [ ] Autenticação JWT + painel de gestão de tenants
- [ ] Processamento assíncrono com Celery + Redis para arquivos `>10MB`
- [ ] Versionamento de modelos com MLflow + rollback automático
- [ ] Dashboard Streamlit para visualização interativa de portfólio
- [ ] Exportação de relatórios em PDF com laudo técnico assinado
- [ ] Integração com bureaus de crédito via API externa

---

## 📄 Licença
Distribuído sob a licença **BSD License**, a fins de pesquisa e desenvolvimento.

---
> 💡 **Documentação interativa:**

> `http://localhost:8000/docs`

> `FastAPI • Pandas • SHAP • Docker • AWS`

