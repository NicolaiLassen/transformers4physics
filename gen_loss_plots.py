import matplotlib.pyplot as plt
import h5py
import numpy as np

def showplot(x, y, title, xlabel, ylabel, log = False):
    plt.plot(l,losses)
    plt.title(title, fontsize=48)
    if log:
        plt.yscale('log')
    plt.ylabel(ylabel, fontsize=32)
    plt.xlabel(xlabel, fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    date = '00'
    time = 'all at once'
    val_every_n_epoch = 50


    # f = h5py.File('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\{}\\{}\\losses.h5'.format(date,time), 'r')
    f = h5py.File('C:\\Users\\s174270\\Documents\\transformers4physics\\transformer_output\\{}\\{}\\transformer_losses.h5'.format(date,time), 'r')
    # losses = np.array(f['train'])
    # l = np.arange(len(losses))
    # showplot(l,losses,'Training loss','Epoch','Loss',log=True)
    # losses = np.array(f['val'])
    # l = np.arange(len(losses))
    # l = np.arange(val_every_n_epoch, (len(losses)+1)*val_every_n_epoch, val_every_n_epoch)
    # showplot(l,losses,'Validation loss','Epoch','Loss',log=True)
    # path = './transformer_output/{}/{}/'.format('00', time)
    # f = h5py.File(path + 'transformer_losses.h5', 'r')
    losses = np.array(f['train'])
    l = np.arange(len(losses))
    plt.figure(figsize=(16,9),dpi=140)
    plt.plot(l,losses)
    plt.title('Training Loss', fontsize=48)
    plt.yscale('log')
    plt.ylabel('Loss', fontsize=32)
    plt.xlabel('Epoch', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid()
    plt.show()
    losses_val = np.array(f['val'])
    l = np.arange(val_every_n_epoch, (len(losses_val)+1)*val_every_n_epoch, val_every_n_epoch)
    f.close()
    plt.figure(figsize=(16,9),dpi=140)
    plt.plot(l,losses_val)
    plt.title('Validation Loss', fontsize=48)
    plt.yscale('log')
    plt.ylabel('Loss', fontsize=32)
    plt.xlabel('Epoch', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid()
    plt.show()
