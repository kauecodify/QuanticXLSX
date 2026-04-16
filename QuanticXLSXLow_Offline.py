# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:26:14 2026

QuanticXLSX Low - Offline

@author: kauec

pip install pandas scikit-learn shap openpyxl

"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import os, time, threading
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    import shapS
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    MISSING_DEP = str(e)

REQUIRED_COLS = ["Empresa", "Setor", "Divida_Liquida", "EBITDA", "Despesas_Financeiras", 
                 "Fluxo_Caixa_Livre", "Divida_Total", "Ativo_Circulante", "Passivo_Circulante", 
                 "Receita", "NOPAT", "Capital_Investido"]
FEATURES = ["Divida_EBITDA","EBITDA_Margin","ROIC","Liquidez_Corrente","Cobertura_Juros",
            "Divida_Patrimonio","FCF_Yield","Divida_Liquida","EBITDA","Fluxo_Caixa_Livre","Receita"]

class QuanticProcessor:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        self.explainer = None

    def calculate_indicators(self, df):
        df = df.copy()
        df["Liquidez_Corrente"] = df["Ativo_Circulante"] / df["Passivo_Circulante"].replace(0, np.nan)
        df["Divida_EBITDA"] = df["Divida_Liquida"] / df["EBITDA"].replace(0, np.nan)
        df["EBITDA_Margin"] = df["EBITDA"] / df["Receita"].replace(0, np.nan)
        df["ROIC"] = df["NOPAT"] / df["Capital_Investido"].replace(0, np.nan)
        df["Cobertura_Juros"] = df["EBITDA"] / df["Despesas_Financeiras"].replace(0, np.nan)
        df["Divida_Patrimonio"] = df["Divida_Total"] / (df["Ativo_Circulante"] - df["Passivo_Circulante"]).replace(0, np.nan)
        df["FCF_Yield"] = df["Fluxo_Caixa_Livre"] / df["Receita"].replace(0, np.nan)
        return df.fillna(0)

    def train_or_load(self, df):
        target = "Default" if "Default" in df.columns else None
        X = df[[c for c in FEATURES if c in df.columns]]
        if target:
            y = df[target].astype(int)
        else:
            # Modo offline/demonstração: simula targets para treinar o modelo
            np.random.seed(42)
            y = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
        self.trained = True
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            pass

    def predict_and_explain(self, df):
        X = df[[c for c in FEATURES if c in df.columns]]
        prob = self.model.predict_proba(self.scaler.transform(X))[:, 1]
        df = df.copy()
        df["default_probability"] = prob
        df["credit_score"] = (850 - 300) * (1 - prob) + 300

        explanations = []
        if self.explainer:
            for idx in range(len(df)):
                row_vals = df.iloc[idx][[c for c in FEATURES if c in df.columns]].values.reshape(1, -1)
                sv = self.explainer.shap_values(self.scaler.transform(row_vals))
                if isinstance(sv, list): sv = sv[1]
                factors = []
                for i, f in enumerate([c for c in FEATURES if c in df.columns]):
                    factors.append({"feature": f, "value": float(row_vals[0][i]), "impact": float(sv[0][i])})
                factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
                explanations.append(factors[:3])
        return df, explanations

    def generate_output(self, df, explanations, input_path):
        base = os.path.splitext(input_path)[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"{base}_QuanticXLSX_{timestamp}.xlsx"

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Dados_Completos', index=False)

            kpis = pd.DataFrame([
                {"KPI": "Total Empresas", "Valor": len(df)},
                {"KPI": "Score Médio", "Valor": f"{df['credit_score'].mean():.1f}"},
                {"KPI": "Prob. Default Média", "Valor": f"{df['default_probability'].mean():.3f}"},
                {"KPI": "Alto Risco (<400)", "Valor": len(df[df['credit_score']<400])},
                {"KPI": "Baixo Risco (≥400)", "Valor": len(df[df['credit_score']>=400])}
            ])
            kpis.to_excel(writer, sheet_name='KPIs', index=False)

            ranking = df[["Empresa", "Setor", "credit_score", "default_probability"]].sort_values("credit_score", ascending=False)
            ranking.to_excel(writer, sheet_name='Ranking', index=False)

            if explanations:
                xai_rows = []
                for i in range(len(df)):
                    for f in explanations[i][:2]:
                        xai_rows.append({
                            "Empresa": df.iloc[i]["Empresa"],
                            "Score": df.iloc[i]["credit_score"],
                            "Fator": f["feature"],
                            "Valor": f["value"],
                            "Impacto_SHAP": f["impact"]
                        })
                pd.DataFrame(xai_rows).to_excel(writer, sheet_name='XAI_Detalhes', index=False)

                imp = dict(zip([c for c in FEATURES if c in df.columns], self.model.feature_importances_))
                fi = pd.DataFrame({"Variável": list(imp.keys()), "Importância": list(imp.values())})
                fi.sort_values("Importância", ascending=False).to_excel(writer, sheet_name='XAI_Importancia_Global', index=False)

        return output_path

class QuanticApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QuanticXLSX Low - Offline")
        self.root.geometry("520x320")
        self.root.resizable(False, False)
        
        self.processor = QuanticProcessor()
        self.file_path = tk.StringVar()
        self._build_ui()

    def _build_ui(self):
        tk.Label(self.root, text="QuanticXLSX Low", font=("Segoe UI", 18, "bold")).pack(pady=12)
        tk.Label(self.root, text="Análise de crédito 100% offline na sua máquina", fg="#555").pack()
        
        frm = tk.Frame(self.root)
        frm.pack(pady=20, fill="x", padx=30)
        
        tk.Entry(frm, textvariable=self.file_path, state="readonly", font=("Segoe UI", 10)).pack(side="left", fill="x", expand=True, padx=(0, 5))
        tk.Button(frm, text="📂 Abrir", command=self._select, bg="#0078D4", fg="white", font=("Segoe UI", 9, "bold")).pack(side="right")
        
        self.progress = ttk.Progressbar(self.root, length=450, mode='determinate')
        self.progress.pack(pady=12)
        
        self.status = tk.Label(self.root, text="Selecione um arquivo .xlsx para começar", fg="#666", font=("Segoe UI", 9))
        self.status.pack()
        
        tk.Button(self.root, text="▶️ PROCESSAR ANÁLISE", command=self._start, bg="#107C10", fg="white", 
                  font=("Segoe UI", 12, "bold"), width=22).pack(pady=15)
        
        tk.Label(self.root, text="Requer: pandas • scikit-learn • shap • openpyxl", font=("Segoe UI", 8), fg="#888").pack(side="bottom", pady=5)

    def _select(self):
        path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if path: self.file_path.set(path)

    def _start(self):
        if not self.file_path.get():
            messagebox.showwarning("Atenção", "Selecione um arquivo .xlsx primeiro.")
            return
        self.progress['value'] = 0
        self.status.config(text="Iniciando...", fg="#0078D4")
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        try:
            self._ui(10, "Lendo planilha...")
            df = pd.read_excel(self.file_path.get(), engine='openpyxl')
            for c in ["Empresa", "Setor"]:
                if c not in df.columns: raise ValueError(f"Coluna obrigatória faltando: '{c}'")
            
            self._ui(30, "Calculando indicadores financeiros...")
            df = self.processor.calculate_indicators(df)
            
            self._ui(50, "Treinando modelo ML + explicabilidade SHAP...")
            self.processor.train_or_load(df)
            df, explanations = self.processor.predict_and_explain(df)
            
            self._ui(80, "Gerando relatório Excel...")
            out = self.processor.generate_output(df, explanations, self.file_path.get())
            
            self._ui(100, f"✅ Concluído! Salvo em:\n{os.path.basename(out)}")
            messagebox.showinfo("Sucesso", f"Análise finalizada!\nArquivo gerado:\n{out}")
        except Exception as e:
            self._ui(0, f"❌ Erro: {str(e)}")
            messagebox.showerror("Erro", str(e))

    def _ui(self, val, txt, color="blue"):
        self.root.after(0, self.progress.__setitem__, 'value', val)
        self.root.after(0, lambda: self.status.config(text=txt, fg=color))

if __name__ == "__main__":
    if not DEPS_OK:
        r = tk.Tk(); r.withdraw()
        messagebox.showerror("Dependências Faltando", f"Instale uma vez:\npip install pandas scikit-learn shap openpyxl\n\nDetalhe: {MISSING_DEP}")
        exit(1)
    root = tk.Tk()
    QuanticApp(root)
    root.mainloop()
