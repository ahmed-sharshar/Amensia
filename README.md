This is the official implementation for "Amnesia: A Stealthy Replay Attack on Continual Learning Dreams" paper

## First Step

Create a folder named "data", download the necessary datasets, and place them inside it.

## How to run (Sample)

- CIFAR-10
```bash
python utils/main.py   --dataset seq-cifar10 --forward_dataset seq-cifar10   --model er_ace --buffer_size 500 --validation 1   --aux_trim 1 --aux_keep_frac 0.1   --scdt 1 --scdt_divergence KL --scdt_budget 0.1   --scdt_window 5 --minibatch_size 64 --batch_size 32
```

Change the model, data, and other hyperparameters as needed.
