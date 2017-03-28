
# coding: utf-8

# In[1]:

from kitt_net import FeedForwardNet
from shelve import open as open_shelve
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib qt')
from numpy import newaxis, delete, zeros, concatenate, unique


# In[2]:

net = FeedForwardNet(hidden=[], tf_name='Sigmoid')
net_sz = FeedForwardNet(hidden=[], tf_name='Sigmoid')


# In[3]:

net.load('../examples/speech/net_speech_pruned.net')
net_sz.load('../examples/speech/net_speech_sz_pruned.net')


# In[4]:

dataset = open_shelve('../examples/speech/dataset_speech_bs2_cs5_ds811_nr500.ds', 'c')
dataset_sz = open_shelve('../examples/speech/dataset_speech_sz.ds', 'c')


# In[7]:

net.t_data = net.prepare_data(x=dataset['x'], y=dataset['y'])
net.v_data = net.prepare_data(x=dataset['x_val'], y=dataset['y_val'])
test_data = net.prepare_data(x=dataset['x_test'], y=dataset['y_test'])
print 'full:', len(net.t_data), len(net.v_data), len(test_data)

net_sz.t_data = net_sz.prepare_data(x=dataset_sz['x'], y=dataset_sz['y'])
net_sz.v_data = net_sz.prepare_data(x=dataset_sz['x_val'], y=dataset_sz['y_val'])
test_data_sz = net_sz.prepare_data(x=dataset_sz['x_test'], y=dataset_sz['y_test'])
print 'sz:', len(net_sz.t_data), len(net_sz.v_data), len(test_data_sz)


# In[8]:

print test_data[0][0].shape, net.w[0].shape, net.b[0].shape
labels = [label for label in net.labels if label in dataset['y_test']]
y_pred = [net.predict(x)[0][0] for x, y in test_data]
print 'Acc:', accuracy_score(y_true=dataset['y_test'], y_pred=y_pred)
cm = confusion_matrix(y_true=dataset['y_test'], y_pred=y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, newaxis]
plt.imshow(cm, aspect='auto', interpolation='none', vmin=0, vmax=1)
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.grid()
plt.colorbar()
plt.show()


# In[37]:

print net.structure, net_sz.structure
print net.labels.index('s'), net.labels.index('z'), net.labels


# In[5]:

# Delete connections to s and z
net.w_is[1][net.labels.index('s'),:] = 0
net.w_is[1][net.labels.index('z'),:] = 0
net.w[1][net.labels.index('s'),:] = 0
net.w[1][net.labels.index('z'),:] = 0
net.b_is[1][net.labels.index('s')] = 0
net.b_is[1][net.labels.index('z')] = 0
net.b[1][net.labels.index('s')] = 0
net.b[1][net.labels.index('z')] = 0


# In[6]:

# Merge nets
tmp_1 = net.w[1].copy()
z_1 = zeros(shape=(net.w[1].shape[0], net_sz.structure[1]))
z_1[net.labels.index('s'),:] = net_sz.w[1][1]
z_1[net.labels.index('z'),:] = net_sz.w[1][2]
net.w[1] = concatenate((tmp_1, z_1), axis=1)
net.b[1][net.labels.index('s')] = net_sz.b[1][1]
net.b[1][net.labels.index('z')] = net_sz.b[1][2]

used_features_new = [(i, ff) for i, ff in enumerate(sorted(unique([f[1] for f in net.used_features]+[f[1] for f in net_sz.used_features])))]
features = [f[1] for f in used_features_new]
w_0 = zeros(shape=(net.structure[1]+net_sz.structure[1], len(used_features_new)))
b_0 = zeros(shape=(net.structure[1]+net_sz.structure[1], 1))
for row in range(net.w[0].shape[0]):
    for col in range(net.w[0].shape[1]):
        w_0[row,features.index(net.used_features[col][1])] = net.w[0][row,col]
    b_0[row] = net.b[0][row]
        
for row in range(net_sz.w[0].shape[0]):
    for col in range(net_sz.w[0].shape[1]):
        w_0[net.w[0].shape[0]+row,features.index(net_sz.used_features[col][1])] = net_sz.w[0][row,col]
    b_0[net.w[0].shape[0]+row] = net_sz.b[0][row]
    
print w_0.shape, len(features), net.structure, net_sz.structure

net.used_features = used_features_new
net.w[0] = w_0
net.b[0] = b_0


# In[9]:

net.dump('../examples/speech/net_speech_tailored.net')

