for RHO in 0.001 0.005 0.01 0.05 0.1
do
for LR in 1e-5 5e-5 1e-4 5e-4
do
    python run.py --epochs 100 --batch_size 32 --lr $LR --method 'SAM' --rho $RHO
done
done