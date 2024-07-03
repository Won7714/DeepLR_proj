for LR in 1e-5 5e-5 1e-4 5e-4
do
    python run.py --epochs 100 --batch_size 32 --lr $LR --method 'Blur' --init 0.5
done