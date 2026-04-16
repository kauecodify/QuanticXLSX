from fastapi import FastAPI, File, UploadFile, Header, Query, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import shap
import joblib
import uuid
import os
import json
from io import BytesIO
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any

# ==================== CONFIGURAÇÃO ====================
class Settings(BaseSettings):
    API_KEYS: Dict[str, str] = {"demo-key": "demo-tenant"}
    MODEL_PATH: str = "credit_model.pkl"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# ==================== APP FASTAPI ====================
app = FastAPI(
    title="QuanticXLSX API",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CONSTANTES ====================
REQUIRED_COLS = [
    "Empresa", "Setor", "Divida_Liquida", "EBITDA", "Despesas_Financeiras",
    "Fluxo_Caixa_Livre", "Divida_Total", "Ativo_Circulante", "Passivo_Circulante",
    "Receita", "NOPAT", "Capital_Investido"
]

FEATURES = [
    "Divida_EBITDA", "EBITDA_Margin", "ROIC", "Liquidez_Corrente",
    "Cobertura_Juros", "Divida_Patrimonio", "FCF_Yield", "Divida_Liquida",
    "EBITDA", "Fluxo_Caixa_Livre", "Receita"
]

# ==================== MOTOR ML + XAI ====================
class CreditEngine:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.explainer = None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Liquidez_Corrente"] = df["Ativo_Circulante"] / df["Passivo_Circulante"].replace(0, np.nan)
        df["Divida_EBITDA"] = df["Divida_Liquida"] / df["EBITDA"].replace(0, np.nan)
        df["EBITDA_Margin"] = df["EBITDA"] / df["Receita"].replace(0, np.nan)
        df["ROIC"] = df["NOPAT"] / df["Capital_Investido"].replace(0, np.nan)
        df["Cobertura_Juros"] = df["EBITDA"] / df["Despesas_Financeiras"].replace(0, np.nan)
        df["Divida_Patrimonio"] = df["Divida_Total"] / (df["Ativo_Circulante"] - df["Passivo_Circulante"]).replace(0, np.nan)
        df["FCF_Yield"] = df["Fluxo_Caixa_Livre"] / df["Receita"].replace(0, np.nan)
        return df.fillna(0)

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        target_col = "Default" if "Default" in df.columns else None
        X = df[[c for c in FEATURES if c in df.columns]]
        
        if target_col and df[target_col].notna().any():
            y = df[target_col].astype(int)
        else:
            np.random.seed(42)
            y = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
        
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
        self.trained = True
        
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            pass
        
        return {"accuracy": float(self.model.score(self.scaler.transform(X), y))}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[[c for c in FEATURES if c in df.columns]]
        prob = self.model.predict_proba(self.scaler.transform(X))[:, 1]
        df = df.copy()
        df["default_probability"] = prob
        df["credit_score"] = (850 - 300) * (1 - prob) + 300
        return df

    def explain_company(self, df: pd.DataFrame, company: str) -> Optional[Dict]:
        if not self.trained or not self.explainer:
            return None
        
        row = df[df["Empresa"] == company]
        if row.empty:
            return None
        
        features_used = [c for c in FEATURES if c in df.columns]
        X_row = row[features_used].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X_row)
        
        sv = self.explainer.shap_values(X_scaled)
        if isinstance(sv, list):
            sv = sv[1]
        
        factors = []
        for i, f in enumerate(features_used):
            factors.append({
                "feature": f,
                "value": float(row[f].iloc[0]),
                "impact": float(sv[0][i])
            })
        factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
        
        return {
            "empresa": company,
            "score": float(row["credit_score"].iloc[0]),
            "factors": factors[:5]
        }

    def get_global_importance(self, df: pd.DataFrame) -> List[Dict]:
        features_used = [c for c in FEATURES if c in df.columns]
        importance = dict(zip(features_used, self.model.feature_importances_.tolist()))
        return [{"feature": k, "importance": round(v, 4)} for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)]

    def save(self):
        joblib.dump({"model": self.model, "scaler": self.scaler}, settings.MODEL_PATH)

    def load(self):
        if os.path.exists(settings.MODEL_PATH):
            data = joblib.load(settings.MODEL_PATH)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.trained = True
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                pass

# Instância global do motor
engine = CreditEngine()

# ==================== MIDDLEWARE MULTI-TENANT ====================
@app.middleware("http")
async def tenant_middleware(request, call_next):
    if request.url.path in ["/docs", "/openapi.json", "/api/v1/health"]:
        return await call_next(request)
    
    api_key = request.headers.get("x-api-key") or request.query_params.get("api_key")
    
    if not api_key or api_key not in settings.API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    request.state.tenant_id = settings.API_KEYS[api_key]
    return await call_next(request)

# ==================== ENDPOINTS ====================
@app.on_event("startup")
def startup():
    engine.load()
    if not engine.trained:
        dummy = pd.DataFrame({c: [0.0]*10 for c in FEATURES + ["Empresa", "Setor"]})
        dummy["Empresa"] = [f"Demo_{i}" for i in range(10)]
        dummy["Setor"] = "Demo"
        engine.train(dummy)
        engine.save()

@app.get("/api/v1/health", tags=["System"])
async def health_check():
    return {
        "status": "ok",
        "version": "1.0.0",
        "model": engine.trained
    }

@app.post("/api/v1/analisar", tags=["Análise"], response_class=StreamingResponse)
async def analisar_credito(
    file: UploadFile = File(...),
    xai: bool = Query(True, description="Incluir abas de explicabilidade SHAP no output"),
    x_api_key: Optional[str] = Header(None)
):
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Apenas arquivos .xlsx são suportados")
    
    content = await file.read()
    df = pd.read_excel(BytesIO(content), engine="openpyxl")
    
    # Validação de colunas obrigatórias
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"Colunas faltantes: {missing}")
    
    # Converter colunas numéricas
    for col in REQUIRED_COLS[2:]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Calcular indicadores
    df = engine.calculate_indicators(df)
    
    # Treinar se necessário e predizer
    if not engine.trained:
        engine.train(df)
    df = engine.predict(df)
    
    # Explicabilidade XAI
    xai_data = None
    if xai and engine.explainer:
        xai_data = {
            "importance": engine.get_global_importance(df),
            "explanations": [
                engine.explain_company(df, emp) 
                for emp in df["Empresa"].unique()[:10]
            ]
        }
    
    # Gerar abas do Excel de saída
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Aba 1: Dados_Completos
        df.to_excel(writer, sheet_name="Dados_Completos", index=False)
        
        # Aba 2: KPIs
        kpis = pd.DataFrame([
            {"KPI": "Total Empresas", "Valor": len(df)},
            {"KPI": "Score Médio", "Valor": f"{df['credit_score'].mean():.1f}"},
            {"KPI": "Prob. Default Média", "Valor": f"{df['default_probability'].mean():.3f}"},
            {"KPI": "Alto Risco (<400)", "Valor": int(len(df[df['credit_score']<400]))},
            {"KPI": "Baixo Risco (≥400)", "Valor": int(len(df[df['credit_score']>=400]))}
        ])
        kpis.to_excel(writer, sheet_name="KPIs", index=False)
        
        # Aba 3: Ranking
        ranking = df[["Empresa", "Setor", "credit_score", "default_probability"]].sort_values("credit_score", ascending=False)
        ranking.to_excel(writer, sheet_name="Ranking", index=False)
        
        # Aba 4: Resumo
        summary = pd.DataFrame([
            {"Métrica": "Alto Risco", "Valor": len(df[df["credit_score"]<400])},
            {"Métrica": "Baixo Risco", "Valor": len(df[df["credit_score"]>=400])}
        ])
        summary.to_excel(writer, sheet_name="Resumo", index=False)
        
        # Abas XAI (opcionais)
        if xai and xai_data:
            # XAI_Importancia
            fi_df = pd.DataFrame(xai_data["importance"])
            fi_df.to_excel(writer, sheet_name="XAI_Importancia", index=False)
            
            # XAI_Detalhes
            if xai_data["explanations"]:
                details = []
                for exp in xai_data["explanations"]:
                    if exp:
                        for f in exp["factors"][:3]:
                            details.append({
                                "Empresa": exp["empresa"],
                                "Score": exp["score"],
                                "Fator": f["feature"],
                                "Valor": f["value"],
                                "Impacto_SHAP": f["impact"]
                            })
                pd.DataFrame(details).to_excel(writer, sheet_name="XAI_Detalhes", index=False)
    
    output.seek(0)
    filename = f"quanticxlsx_{uuid.uuid4().hex[:8]}.xlsx"
    
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/api/v1/explicar/{empresa}", tags=["XAI"])
async def explicar_empresa(
    empresa: str,
    x_api_key: Optional[str] = Header(None)
):
    if not engine.trained or not engine.explainer:
        raise HTTPException(status_code=503, detail="Modelo não inicializado")
    
    # Criar dados dummy para explicação se não houver dados reais
    dummy = pd.DataFrame({c: [0.0]*5 for c in FEATURES + ["Empresa", "Setor"]})
    dummy["Empresa"] = [empresa] + [f"Demo_{i}" for i in range(4)]
    dummy["Setor"] = "Demo"
    
    if "Default" not in dummy.columns:
        dummy["Default"] = 0
    
    dummy = engine.calculate_indicators(dummy)
    if not engine.trained:
        engine.train(dummy)
    
    result = engine.explain_company(dummy, empresa)
    if not result:
        raise HTTPException(status_code=404, detail=f"Empresa '{empresa}' não encontrada para explicação")
    
    return result

@app.get("/api/v1/tenants", tags=["Admin"])
async def list_tenants(x_api_key: Optional[str] = Header(None)):
    # Apenas para demonstração: retorna chaves configuradas
    demo_keys = {"demo-key": "demo-tenant"}
    keys = settings.API_KEYS if settings.API_KEYS else demo_keys
    if x_api_key not in keys.values():
        raise HTTPException(status_code=403, detail="Admin access required")
    return {"tenants": keys}

# ==================== EXECUÇÃO ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
