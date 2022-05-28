import pandas as pd
import numpy as np
import pylab as plt

plot_loss = True
create_log_file = False

epoch = []
train_loss = []
train_error = []
val_error = []
val_3px_error = []

if __name__ == "__main__":

    if(create_log_file):
        f = open('./run/Kitti15/log.txt', 'r')
        lines = f.readlines()
        f.close()

        for l in lines:
            if('Complete' in l):
                errors = l.split(',')
                avg_loss_str = errors[0]
                avg_error_str = errors[1]
                epoch.append(int(avg_loss_str[avg_loss_str.find("Epoch")+6:avg_loss_str.find("Epoch")+8]))
                avg_loss = avg_loss_str[avg_loss_str.find("(")+1:avg_loss_str.find(")")]
                avg_error = avg_error_str[avg_error_str.find("(")+1:avg_error_str.find(")")]
                train_loss.append(float(avg_loss))
                train_error.append(float(avg_error))
            elif('Test:' in l):
                errors = l[l.find("(")+1:l.find(")")]
                errors = errors.split(' ')
                val_error.append(float(errors[0]))
                val_3px_error.append(float(errors[1]))
                print(errors)

        results = list(zip(epoch, train_loss, train_error, val_error, val_3px_error))

        results_df = pd.DataFrame(results, columns=['epoch', 'train_loss', 'train_error', 'val_error', 'val_3px_error'])
        results_df.to_csv('./run/Kitti15/log.csv', index=False)

    if(plot_loss):
        results = pd.read_csv('./run/Kitti15/log.csv')
        epoch = results['epoch'].tolist()
        train_loss = results['train_loss'].tolist()
        train_error = results['train_error'].tolist()
        val_error = results['val_error'].tolist()
        val_3px_error = results['val_3px_error'].tolist()

        font = {'family' : 'normal',
        'size'   : 14}

        fig = plt.figure(figsize=(10,10))
        plt.subplot(2, 1, 1)
        plt.plot(epoch, train_loss, label='train loss')
        plt.xlabel('Epochs')
        plt.ylabel('Smooth L1 Loss')
        plt.legend(loc='upper right')

        plt.subplot(2, 1, 2)
        plt.plot(epoch, train_error, label='train_error')
        plt.plot(epoch, val_error, label='val_error')
        plt.plot(epoch, val_3px_error, label='val_3px_error')
        plt.xlabel('Epochs')
        plt.ylabel('End Point Error (EPE)')
        plt.legend(loc='lower right')

        plt.show()

