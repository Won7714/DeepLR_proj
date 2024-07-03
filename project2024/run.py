# Wonsang Yun, Dept. of mathmetics, Yonsei Univ.
# DEEP LEARNING (STA3140.01-00) Final project 2024/05/07~

import numpy as np
import pandas as pd
from train import train_test
import time
import argparse



if __name__ == '__main__':
    c = time.time()
    c = str(c)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', required=False, default=100, type=int)
    parser.add_argument('--batch_size', required=False, default=32, type=int)
    parser.add_argument('--lr', required=False, default=1e-5, type=float)
    parser.add_argument('--method', required=False, default='Adam', type=str)
    parser.add_argument('--rho', required=False, default=0, type=float)                 #for SAM
    parser.add_argument('--init', required=False, default=0, type=float)                #for adversarial attack train, blur
    parser.add_argument('--end', required=False, default=0, type=float)                 #for adversarial attack train, blur

    args = parser.parse_args()

    val_history_loss, val_history_acc, history_loss, history_acc = train_test(args)
    f = open(f"result/{args.method}/result{args.epochs}" + c[-7:-1] + ".txt", 'w')
    data = f'epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}, method: {args.method}, rho: {args.rho}\n'
    f.write(data)
    data = 'VALIDATION\n'
    f.write(data)
    for i in range(0, len(val_history_loss)):
        data = f'{val_history_loss[i]:.4f}, {val_history_acc[i]:.4f}\n'
        f.write(data)

    data = 'TEST\n'
    f.write(data)
    for i in range(0, len(history_acc)):
        data = ''
        for j in range(0, 6):
            if j == 5:
                data += f'{history_loss[i][j]:.4f}, {history_acc[i][j]:.4f}'
            else:
                data += f'{history_loss[i][j]:.4f}, {history_acc[i][j]:.4f}, '
        data += '\n'
        f.write(data)


    best_model = np.min(val_history_loss)
    f.write(f'{best_model:.4f}')
    f.close()
