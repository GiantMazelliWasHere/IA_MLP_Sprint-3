Fazer tbm: Pilares essenciais incluem alimentação equilibrada, atividade física regular, sono adequado e relações sociais positivas.
# Sprint 3 - MLP - H-bit
## Medicina Preventiva por Hábitos

## Integrantes:
- Edson Leonardo Pacheco Navia
- Eduardo Mazelli
- Lucas Masaki Nagahama
- Joseh Gabriel Trimboli Agra
- Pedro Henrique de Assumção Lima

---

### **Problema**
Os humanos possuem hábitos que aumentam o risco de distúrbios do sono (Insônia e Apneia), mas poucos recebem orientação preventiva personalizada.

### **O que a IA faz no projeto?**
A MLP **classifica o risco** (`Nenhum`, `Insônia`, `Apneia do Sono`) com base em hábitos diários, permitindo ao app recomendar **mudanças preventivas personalizadas**.

### **Dados**
- **Dataset**: Health and Lifestyle Metrics (público)
- **Amostras**: 374 indivíduos
- **Features**: 13 variáveis (idade, sono, atividade física, estresse, IMC, etc.)
- **Target**: Distúrbio do sono (3 classes)
- **Link do daraset**: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

### **Ferramenta**
- **Python** (3.10)
- **scikit-learn** (MLPClassifier)
- **Pandas** (Pre-process)
- **MATPLOTLIB** (Vizualizações)
- **Numpy** (Numéricos)

---

## **Arquitetura da Rede Neural**

Entrada (11 features de hábitos)
|
[Dense: 64 neurônios, ReLU]
|
[Dense: 32 neurônios, ReLU]
|
[Saída: 3 neurônios, Softmax]
|
Nenhum | Insônia | Apneia


**Configurações:**
- Otimizador: Adam
- Learning Rate: 0.001 (adaptativo)
- Regularização L2: 0.001
- Early Stopping: ativado
- Batch Size: 32

---

## **Resultados**

### Métricas de Performance
- **Acurácia**: ~89%
- **F1-Score Macro**: ~0.88
- **Cross-Validation (5-fold)**: 87.2% ± 3.1%

### Features Mais Importantes
1. **Sleep Duration** - Duração do sono
2. **Quality of Sleep** - Qualidade do sono
3. **Stress Level** - Nível de estresse
4. **Age** - Idade
5. **Physical Activity Level** - Atividade física

---

## **Aplicação Prática**

### Exemplo de Predição
**Usuário:** Homem, 35 anos, engenheiro
- Sono: 6h/dia, qualidade 5/10
- Estresse: 8/10
- Atividade física: 30 min/dia
- 4.000 passos/dia

**Resultado:**
- **Risco de Insônia**: 73.2%
- **Sem distúrbio**: 21.5%
- **Risco de Apneia**: 5.3%

### Recomendações do App
1. **Melhore a higiene do sono** (7-8h/noite)
2. **Técnicas de relaxamento** (reduzir estresse)
3. **Aumentar atividade física** (45-60 min/dia)
4. **Monitoramento contínuo** via app

---

## **Impacto para o Negócio**

### Para os Usuários
- **Prevenção personalizada** antes dos sintomas
- **Mudanças de hábitos orientadas por IA**
- **Monitoramento contínuo** do progresso

### Para a Care Plus
- **Engajamento** aumentado no app
- **Valor agregado** com medicina preventiva
- **Diferencial competitivo** no mercado
