# Sprint 3 — MLP aplicada à Medicina do Estilo de Vida

## Integrantes:
- **Edson Leonardo Pacheco Navia**
- **Eduardo Mazelli**
- **Lucas Masaki Nagahama**
- **Joseh Gabriel Trimboli Agra**
- **Pedro Henrique de Assumção Lima** 
## Repositório: 
https://github.com/GiantMazelliWasHere/IA_MLP_Sprint-3

---

## Objetivo
Utilizar uma **MLP (Multilayer Perceptron)** para resolver um problema dentro do projeto
de Medicina do Estilo de Vida — classificar o **nível de obesidade/saúde** de um
indivíduo a partir de dados de hábitos diários (alimentação, atividade física,
hidratação, sono, uso de tecnologia, álcool, fumo e meio de transporte).

A Medicina do Estilo de Vida atua sobre **6 pilares**:
1. **Alimentação** saudável (predominantemente vegetal)
2. **Atividade física** regular
3. **Sono** restaurador
4. **Manejo do estresse**
5. **Conexões sociais** saudáveis
6. **Evitar substâncias de risco** (álcool, fumo, drogas)

Cada dataset fornecido cobre um ou mais pilares; este notebook centraliza o
treino da MLP no **ObesityDataSet** (que reúne 4 dos 6 pilares) e referencia
os demais como extensões futuras.

## Problema

**O que a IA faz no projeto?**

A IA aprende, a partir de variáveis de estilo de vida coletadas via
questionário, a **predizer em qual das 7 categorias de status de peso**
o indivíduo se encontra:
`Insufficient_Weight`, `Normal_Weight`, `Overweight_Level_I/II`,
`Obesity_Type_I/II/III`.

Esse classificador serve como **ferramenta de triagem** num app de
Medicina do Estilo de Vida: ao receber as respostas do usuário sobre
hábitos, sugere o pilar mais crítico a ser trabalhado e identifica
perfis de risco antes mesmo de medir IMC.

## Dados

### De onde vieram
- **Dataset principal:** `ObesityDataSet_raw_and_data_sinthetic.csv`  
  Originário de Palechor & Manotas (2019) — coleta real (México, Peru,
  Colômbia) **balanceada por sobre-amostragem sintética via SMOTE**.
  Disponível no UCI ML Repository.

- **Datasets de apoio (mapeados aos pilares):**
  | Pilar | Dataset | Link 
  |---|---|
  | Sono | `Sleep_health_and_lifestyle_dataset.csv` |
  | Estresse | `StressLevelDataset.csv`, `Stress_Dataset.csv` |
  | Conexões sociais | `self-reported-loneliness-older-adults.csv` |
  | Substâncias de risco | `deaths-illicit-drugs.csv` |
  | Nutrição + Atividade | `meal_metadata.csv`, `Final_data.csv` |

### Variáveis principais (ObesityDataSet)
| Coluna | Significado | Pilar |
|---|---|---|
| FAVC | Consumo frequente de alimentos calóricos | Nutrição |
| FCVC | Frequência de vegetais nas refeições | Nutrição |
| NCP | Nº de refeições principais | Nutrição |
| CAEC | Lanches entre refeições | Nutrição |
| CH2O | Consumo diário de água (L) | Nutrição |
| SCC | Monitora calorias? | Nutrição |
| FAF | Frequência de atividade física | Atividade |
| TUE | Tempo de tela / sedentarismo | Atividade |
| MTRANS | Meio de transporte | Atividade |
| SMOKE | Fumante? | Substâncias |
| CALC | Consumo de álcool | Substâncias |
| Gender, Age, Height, Weight | Demografia | Contexto |
| family_history_with_overweight | Histórico familiar | Contexto |
| **NObeyesdad** | **Alvo** (7 classes) | — |

## Pré-processamento

- **Categóricas → one-hot** (`pd.get_dummies`)
- **Numéricas → z-score** (média 0, desvio 1)
- **Split estratificado** 70 / 10 / 20 (treino / val / teste)

## Modelo — MLP construída do zero (NumPy)

Optamos por implementar a MLP do zero (sem `sklearn` ou `tensorflow`)
para deixar **explícitos** todos os mecanismos didáticos:

### Arquitetura
```
Entrada (31)  →  Linear(31, 64) + ReLU  →  Linear(64, 7) + Softmax
```

### Hiperparâmetros
| Item | Valor |
|---|---|
| Camada oculta | 64 neurônios |
| Ativação oculta | ReLU |
| Saída | Softmax (7 classes) |
| Função de perda | Cross-Entropy categórica |
| Otimizador | SGD com momentum (0.9) |
| Taxa de aprendizado | 0.05 |
| Tamanho do mini-batch | 64 |
| Épocas | 200 |
| Inicialização | He (√(2/n_in)) |

### Ferramenta utilizada
**Python 3 + NumPy + pandas + matplotlib** (pacotes científicos padrão).  
O código pode ser facilmente substituído por `MLPClassifier` do
`sklearn` ou por uma rede equivalente em Keras/PyTorch.

## Discussão

### Pontos fortes
- **Acurácia de teste: 95.67%** em 7 classes — excelente
  para um modelo simples com apenas 1 camada oculta.
- A classe **Obesity_Type_III** atinge 100% de F1 (todos os atributos do
  questionário se correlacionam fortemente).
- O treinamento converge cedo (~20 épocas) sem overfitting severo: a
  acurácia de validação se mantém acima de 94%.

### Limitações
- **Normal_Weight × Overweight_Level_I** apresentam o maior número de
  confusões (fronteira biológica naturalmente fluida).
- O dataset foi balanceado por SMOTE — em produção é prudente reavaliar
  com dados reais não-sintéticos.

### Conexão com a Medicina do Estilo de Vida
Os atributos mais informativos para o modelo são justamente os pilares
da Medicina do Estilo de Vida: **alimentação (FAVC, FCVC, CAEC),
atividade física (FAF, MTRANS), hidratação (CH2O) e substâncias (CALC,
SMOKE)** — confirmando que estilo de vida é um preditor robusto do
status de peso.