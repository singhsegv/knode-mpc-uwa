# About

Cherry picked stuff from the codebase to be run on real hardware. Everything in this folder is self sufficient.

```
ros2 topic pub /dummy_xy geometry_msgs/msg/Point "{x: 0.4, y: 0.2, z: 0.0}"
```

## Todo

Add a timeout to that hold. Holding the last torque forever is the dangerous part. If the MPC node crashes or stalls while commanding a large torque, the arm keeps applying it. forward_command_controller has no timeout, so you need a separate guard. There are two layers:

The ros2_control process dies. The X-series servos have a Bus Watchdog register (address 98, in units of 20 ms). Set it, for example, to 5 (100 ms). If the servo receives no packets for that long, it stops on its own. This does nothing if only the MPC node dies, because the hardware interface keeps talking to the bus.
The MPC node stalls. The most reliable fix is a small C++ controller copied from forward_command_controller. It stores the timestamp of the last command and writes zero (or g(q)) if that timestamp is older than about 3 control periods. Putting the check inside the realtime loop means it doesn't depend on any Python process staying alive. The simpler alternative is a separate watchdog node that deactivates the effort controller through controller_manager/switch_controller when commands stop. That works, but it is slower and still adds another process that can fail.
