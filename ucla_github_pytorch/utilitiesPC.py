
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import time
import math
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from sklearn.metrics import accuracy_score
import torch

def knn(data_train, data_test, label_train, label_test, nn=1):
    label_train = np.asarray(label_train)
    label_test = np.asarray(label_test)

    Xtr_Norm = preprocessing.normalize(data_train)
    Xte_Norm = preprocessing.normalize(data_test)

    knn = KNeighborsClassifier(n_neighbors=nn,
                               metric='cosine')  # , metric='cosine'#'mahalanobis', metric_params={'V': np.cov(data_train)})
    knn.fit(Xtr_Norm, label_train)
    pred = knn.predict(Xte_Norm)
    acc = accuracy_score(pred, label_test)
    return acc

def save_checkpoint(model, epoch, optimizer, loss, PATH):
    torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss,
    }, PATH)

def load_checkpoint(model, optimizer, PATH):
    data = torch.load(PATH)
    model.load_state_dict(data['model_state_dict'])
    optimizer.load_state_dict(data['optimizer_state_dict'])
    return data['epoch'], data['loss']

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))