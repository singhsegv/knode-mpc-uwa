# All about MPC performance

## Logs and Performance Experiments

### 1. Fatrop Solver 
- Not realtime (currently ~15hz with FATROP solver)

```bash
# Non knode performance for 10 steps horizon of dt=0.01 each
# Solve time is for single iteration of those 1000 ones -> So pretty bad
Starting closed-loop MPC for 00_circle (10.0s)...
  Step 0/1000 | Time: 0.01s | Solve: 164.1ms
  Step 100/1000 | Time: 1.01s | Solve: 65.8ms
  Step 200/1000 | Time: 2.01s | Solve: 68.5ms
  Step 300/1000 | Time: 3.01s | Solve: 65.4ms
  Step 400/1000 | Time: 4.01s | Solve: 66.8ms
  Step 500/1000 | Time: 5.01s | Solve: 67.7ms
  Step 600/1000 | Time: 6.01s | Solve: 66.4ms
  Step 700/1000 | Time: 7.01s | Solve: 65.6ms
  Step 800/1000 | Time: 8.01s | Solve: 65.5ms
  Step 900/1000 | Time: 9.01s | Solve: 66.3ms

========================================================================================================
idx  type            RMSE [m]     DTW [m]  DTW mean [m]     max [m]  solve avg [ms]
--------------------------------------------------------------------------------------------------------
  0  circle            0.0022      1.5325        0.0015      0.0041            68.9
  1  line              0.0032      1.8945        0.0024      0.0039            67.5
========================================================================================================
```

### 2. Generic IPOPT Solver
- Not converging with the current setting of 100 (q and qdot weights in Q) and 0.01 in R
- Q * 25 as the final state cost
- "max_iter": 500 and "hessian_approximation": "limited-memory"

### 3. Well tuned IPOPT Solver with CasADi (near realtime 100hz for simulation)
```bash
  Step 0/1000 | Time: 0.01s | Solve: 263.8ms per single step
  Step 100/1000 | Time: 1.01s | Solve: 8.9ms per single step
  Step 200/1000 | Time: 2.01s | Solve: 9.4ms per single step
  Step 300/1000 | Time: 3.01s | Solve: 9.3ms per single step
  Step 400/1000 | Time: 4.01s | Solve: 9.7ms per single step
  Step 500/1000 | Time: 5.01s | Solve: 11.9ms per single step
  Step 600/1000 | Time: 6.01s | Solve: 8.5ms per single step
  Step 700/1000 | Time: 7.01s | Solve: 8.6ms per single step
  Step 800/1000 | Time: 8.01s | Solve: 8.4ms per single step
  Step 900/1000 | Time: 9.01s | Solve: 8.5ms per single step

========================================================================================================
idx  type            RMSE [m]     DTW [m]  DTW mean [m]     max [m]  solve avg [ms]
--------------------------------------------------------------------------------------------------------
  0  circle            0.0019      1.2735        0.0013      0.0037             9.4
========================================================================================================
```