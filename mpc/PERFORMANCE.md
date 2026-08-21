# All about MPC performance

## Not realtime worthy
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