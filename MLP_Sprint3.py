from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
DATA = Path('.\IA_Datasets_Sprint_3\ObesityDataSet_raw_and_data_sinthetic.csv')
print('NumPy:', np.__version__, '| Pandas:', pd.__version__)

df = pd.read_csv(DATA).drop_duplicates().reset_index(drop=True)
print('Linhas:', len(df), '| Colunas:', df.shape[1])
print('\nDistribuição do alvo:')
print(df['NObeyesdad'].value_counts())
df.head()

ax = df['NObeyesdad'].value_counts().plot(kind='bar', figsize=(9,4),
                                          color='#3b82f6', rot=30)
ax.set_title('Distribuição das classes (NObeyesdad)')
ax.set_ylabel('Quantidade')
plt.tight_layout(); plt.show()

y_raw = df['NObeyesdad'].values
classes = sorted(np.unique(y_raw).tolist())
class_to_idx = {c:i for i,c in enumerate(classes)}
y = np.array([class_to_idx[c] for c in y_raw])

X_df = df.drop(columns=['NObeyesdad'])
cat = X_df.select_dtypes(include='object').columns.tolist()
num = [c for c in X_df.columns if c not in cat]

X_cat = pd.get_dummies(X_df[cat], drop_first=False).astype(float)
X_num = (X_df[num].astype(float) - X_df[num].mean()) / X_df[num].std()
X = pd.concat([X_num, X_cat], axis=1).values.astype(np.float32)

print('Features após one-hot:', X.shape[1], '| Classes:', len(classes))

def stratified_split(X, y, test=0.2, val=0.1, seed=42):
    rng = np.random.RandomState(seed)
    tr, vl, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y==c)[0]; rng.shuffle(idx)
        n = len(idx); nt = int(round(n*test)); nv = int(round(n*val))
        te.extend(idx[:nt]); vl.extend(idx[nt:nt+nv]); tr.extend(idx[nt+nv:])
    rng.shuffle(tr); rng.shuffle(vl); rng.shuffle(te)
    return X[tr], y[tr], X[vl], y[vl], X[te], y[te]

Xtr, ytr, Xv, yv, Xt, yt = stratified_split(X, y)
print(f'train={len(Xtr)}  val={len(Xv)}  test={len(Xt)}')

def relu(x):       return np.maximum(0, x)
def relu_grad(x):  return (x > 0).astype(x.dtype)
def softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
def one_hot(y, k):
    o = np.zeros((y.size, k), dtype=np.float32); o[np.arange(y.size), y] = 1; return o

class MLP:
    def __init__(self, n_in, n_h, n_out, lr=0.05, mom=0.9, seed=42):
        r = np.random.RandomState(seed)
        self.W1 = r.randn(n_in, n_h).astype(np.float32) * np.sqrt(2/n_in)
        self.b1 = np.zeros((1, n_h), np.float32)
        self.W2 = r.randn(n_h, n_out).astype(np.float32) * np.sqrt(2/n_h)
        self.b2 = np.zeros((1, n_out), np.float32)
        self.lr, self.mom = lr, mom
        self.vW1 = np.zeros_like(self.W1); self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2); self.vb2 = np.zeros_like(self.b2)

    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1; self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2; self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, Y):
        m = self.X.shape[0]
        dZ2 = (self.A2 - Y) / m
        dW2 = self.A1.T @ dZ2; db2 = dZ2.sum(0, keepdims=True)
        dZ1 = (dZ2 @ self.W2.T) * relu_grad(self.Z1)
        dW1 = self.X.T @ dZ1; db1 = dZ1.sum(0, keepdims=True)
        self.vW1 = self.mom*self.vW1 - self.lr*dW1
        self.vb1 = self.mom*self.vb1 - self.lr*db1
        self.vW2 = self.mom*self.vW2 - self.lr*dW2
        self.vb2 = self.mom*self.vb2 - self.lr*db2
        self.W1 += self.vW1; self.b1 += self.vb1
        self.W2 += self.vW2; self.b2 += self.vb2

    def loss(self, Y, eps=1e-9):
        return float(-np.mean(np.sum(Y*np.log(self.A2+eps), axis=1)))

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

model = MLP(n_in=X.shape[1], n_h=64, n_out=len(classes), lr=0.05, mom=0.9)
hist = {'loss':[], 'val_loss':[], 'acc':[], 'val_acc':[]}
Y_tr = one_hot(ytr, len(classes)); Y_v = one_hot(yv, len(classes))

epochs, bs = 200, 64
for ep in range(1, epochs+1):
    perm = np.random.permutation(len(Xtr))
    for i in range(0, len(Xtr), bs):
        b = perm[i:i+bs]; model.forward(Xtr[b]); model.backward(Y_tr[b])
    model.forward(Xtr); tl = model.loss(Y_tr); ta = (model.predict(Xtr)==ytr).mean()
    model.forward(Xv);  vl = model.loss(Y_v);  va = (model.predict(Xv)==yv).mean()
    hist['loss'].append(tl); hist['val_loss'].append(vl)
    hist['acc'].append(float(ta)); hist['val_acc'].append(float(va))
    if ep==1 or ep%20==0 or ep==epochs:
        print(f'Ep {ep:3d}/{epochs} | loss={tl:.4f} val_loss={vl:.4f} | acc={ta:.4f} val_acc={va:.4f}')

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist['loss'], label='treino'); ax[0].plot(hist['val_loss'], label='validação')
ax[0].set_title('Perda'); ax[0].set_xlabel('Época'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(hist['acc'], label='treino'); ax[1].plot(hist['val_acc'], label='validação')
ax[1].set_title('Acurácia'); ax[1].set_xlabel('Época'); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()

y_pred = model.predict(Xt)
acc = (y_pred == yt).mean()
print(f'Acurácia no teste: {acc*100:.2f}%')

# matriz de confusão
cm = np.zeros((len(classes), len(classes)), int)
for a, b in zip(yt, y_pred): cm[a, b] += 1

# precisão / recall / f1 por classe
print(f"\n{'Classe':<25}{'Prec.':>8}{'Recall':>8}{'F1':>8}{'Sup.':>8}")
for i, c in enumerate(classes):
    tp = cm[i,i]; fn = cm[i,:].sum()-tp; fp = cm[:,i].sum()-tp
    p = tp/(tp+fp) if tp+fp>0 else 0; r = tp/(tp+fn) if tp+fn>0 else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    print(f'{c:<25}{p:>8.3f}{r:>8.3f}{f1:>8.3f}{cm[i,:].sum():>8d}')

fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha='right'); ax.set_yticklabels(classes)
ax.set_xlabel('Predito'); ax.set_ylabel('Real'); ax.set_title('Matriz de Confusão (Teste)')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                color='white' if cm[i,j]>cm.max()/2 else 'black', fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.show()
