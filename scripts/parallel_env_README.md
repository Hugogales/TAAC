# True Parallel Environment Execution in TAAC

This document explains how the parallel environment execution works in TAAC and how the new `parallel_env_example.py` script demonstrates true parallelism using multiprocessing.

## Types of Parallelism in Reinforcement Learning

There are two main approaches to parallelism in reinforcement learning:

1. **Simulated Parallelism** - Running multiple environments sequentially in the same process
2. **True Parallelism** - Running multiple environments simultaneously across different CPU cores/processes

## Current TAAC Implementation

The current TAAC implementation in `train_taac.py` uses **simulated parallelism**:

- The `ParallelEnvironmentManager` class creates multiple environment instances
- These environments are run sequentially within the same process
- This approach is simple and avoids multiprocessing complexities
- However, it doesn't utilize multiple CPU cores for environment stepping

```python
# From train_taac.py - Simulated parallelism
parallel_manager = ParallelEnvironmentManager(env_name, env_kwargs, num_parallel)
# ...
for env_idx in range(num_parallel):
    # Run episode for this environment (sequentially)
    # ...
```

## True Parallelism with `parallel_env_example.py`

The new `parallel_env_example.py` script demonstrates **true parallelism**:

- Uses Python's `multiprocessing` module to run environments in separate processes
- Each process has its own environment and agent instance
- Experiences are collected in parallel and then combined for updates
- This approach fully utilizes multiple CPU cores

```python
# From parallel_env_example.py - True parallelism
with mp.Pool(processes=num_processes) as pool:
    # Run episodes in parallel across multiple processes
    results = pool.map(run_single_episode, args_list)
```

## Key Differences

| Feature | Simulated Parallelism | True Parallelism |
|---------|----------------------|------------------|
| CPU Usage | Single core | Multiple cores |
| Memory Usage | Lower | Higher (separate process overhead) |
| Speed | Limited by single core | Scales with CPU cores |
| Complexity | Simpler | More complex (process communication) |
| GPU Usage | Single GPU | Can use multiple GPUs |
| Rendering | Possible | Limited (only one process can render) |

## When to Use Each Approach

**Use Simulated Parallelism (default TAAC) when:**
- You have a simple environment with fast step times
- Memory usage is a concern
- You're using rendering during training
- You want simpler code with fewer dependencies

**Use True Parallelism (new example) when:**
- You have computationally expensive environments
- Environment stepping is the bottleneck (not network updates)
- You have multiple CPU cores available
- You want maximum training throughput

## Usage Example

```bash
# Run with 8 parallel processes (8 CPU cores)
python scripts/parallel_env_example.py --config configs/boxjump.yaml --num_processes 8

# Run with rendering (automatically uses single process)
python scripts/parallel_env_example.py --config configs/boxjump.yaml --render

# Continue training from existing model
python scripts/parallel_env_example.py --config configs/boxjump.yaml --model_path files/Models/boxjump/best_model.pth
```

## Implementation Notes

1. **Process Initialization**:
   - Each process gets its own environment and agent
   - Random seeds are set differently for each process
   - GPU devices are assigned in round-robin fashion

2. **Memory Management**:
   - Experiences are collected in each process
   - After episodes complete, memories are combined
   - The master agent performs updates with combined experiences

3. **Rendering**:
   - When rendering is enabled, only one process is used
   - This avoids conflicts with multiple processes trying to render

4. **GPU Usage**:
   - Each process can be assigned to a different GPU
   - This enables multi-GPU training without framework-specific parallelism

## Comparison to Your Previous Code

The implementation is similar to your previous code that used:
```python
with mp.Pool(processes=num_games) as pool:
    results = pool.map(run_single_game, args_list)
```

But adapted for the TAAC architecture and PettingZoo environments. 