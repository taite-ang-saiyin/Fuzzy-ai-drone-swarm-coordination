Fuzzy Drone Swarm – Setup Guide (Windows)

1) Prerequisites
- Install Webots R2025a or newer.
- Install Anaconda/Miniconda (recommended).
- GPU is NOT required. PPO will run on CPU.

2) Clone/Copy the Project
- Place the folder at a path without spaces if possible, e.g. C:\Workspace\Fuzzy

3) Create the Python environment
- Open Anaconda Prompt and run:
  conda create -n drone_rl_env python=3.10 -y
  conda activate drone_rl_env

4) Install Python dependencies
- Base (required):
  pip install numpy scikit-fuzzy
- RL and model loading:
  pip install gymnasium stable-baselines3[extra] torch
- Optional (visualization, logging):
  pip install tensorboard opencv-python rich tqdm

Verify:
  python -c "import torch, gymnasium, stable_baselines3 as sb3; print('OK')"

5) Point Webots to your Conda Python
- Open Webots → Tools → Preferences… → General → Python command
- Set to your env’s python.exe, e.g.:
  C:\Users\<you>\anaconda3\envs\drone_rl_env\python.exe
- Click OK and restart Webots.

6) Configure model path (PPO)
- The supervisor loads: controllers/my_rl_supervisor/ppo_final_model.zip
- Ensure the file exists at that exact path. If you have a different model, place it there or update ai_supervisor.py accordingly.
- The supervisor forces CPU by default. To silence GPU warnings, we already set device="cpu" in ai_supervisor.py.

7) Running the simulation
- Open the world:
  worlds/mavic_2_pro.wbt
- Press Play. You should see:
  - AI supervisor starts (my_rl_supervisor)
  - Drones start Python controller (mavic_controller)
  - PPO model loads (message: "AI model loaded successfully")

8) What the controllers do
- mavic_controller:
  - Reads IMU/GPS/gyro/compass if available on the drone PROTO
  - Attempts to get front distance via physical sensors; if not present, falls back to the built‑in camera and estimates proximity from image edges
  - Applies fuzzy policy outputs (thrust/yaw) and boids steering
  - Adds a forward motion bias so drones translate even without PPO commands

- my_rl_supervisor (ai_supervisor.py):
  - Finds drones by node type (Mavic2Pro)
  - Loads PPO model, computes a 4‑dim observation per drone, predicts a discrete action (0=obstacle_avoidance, 1=cohesion, 2=alignment), and broadcasts actions via the supervisor’s emitter

9) Optional: Add radio Receiver and RangeFinder to drones
- Stock Webots Mavic2Pro PROTO does not expose a Receiver or a RangeFinder. The project works without them (camera‑based fallback), but for true radio control + distance sensing, create a custom PROTO:
  a) Copy the official PROTO to the project (Command Prompt):
     copy "C:\Program Files\Webots\projects\robots\dji\mavic\protos\Mavic2Pro.proto" "C:\Workspace\Fuzzy\protos\Mavic2Pro_Custom.proto"
  b) Edit C:\Workspace\Fuzzy\protos\Mavic2Pro_Custom.proto
     - Ensure first line is: #VRML_SIM R2025a utf8
     - Find the top‑level Robot { ... } → children [ ... ] block
       Add:
         Receiver { name "receiver" channel 1 type "radio" }
         RangeFinder { name "range finder" }
  c) In worlds/mavic_2_pro.wbt add:
     EXTERNPROTO "../protos/Mavic2Pro_Custom.proto"
     Replace each drone node with Mavic2Pro_Custom { ... }
  d) The Python controller already looks for devices named "receiver" and "range finder".

10) Supervisor Emitter (already configured)
- The world defines a supervisor Robot named AI_Supervisor with an Emitter named "emitter" on channel 1.
- The drone controllers do not require a Receiver when running with the built‑in forward/fuzzy logic; it is only needed for strict radio‑based PPO control.

11) Tuning motion speed and avoidance
- In controllers/mavic_controller/mavic_controller.py:
  - SPEED_MULTIPLIER: scales forward push (default 2.0)
  - Emergency avoidance thresholds are defined in the reactive logic (look for front_distance < ...). Lower thresholds or increase yaw gain to turn harder.

12) Troubleshooting
- PPO warning about GPU: harmless (we force CPU).
- "Device 'receiver'/'range finder' was not found": expected on stock Mavic2Pro; the controller falls back to camera‑based estimation.
- No movement / very slow:
  - Increase SPEED_MULTIPLIER (e.g., 2.5–3.0)
  - Slightly raise K_VERTICAL_THRUST (e.g., 72–74) if takeoff is weak
- Supervisor crashes with observation shape error:
  - Ensure ai_supervisor.py observation space is shape (4,) and the obs vector length is 4
- UDP errors (Windows firewall): allow Python in firewall or run as Administrator

13) Training (optional, not required to run)
- Training scripts/templates exist under controllers/my_rl_supervisor/ (e.g., train.py). If you retrain:
  - Save the model as controllers/my_rl_supervisor/ppo_final_model.zip
  - Keep the observation/action spaces consistent with ai_supervisor.py

14) File map (key entries)
- worlds/mavic_2_pro.wbt – entry world
- controllers/mavic_controller/mavic_controller.py – drone low‑level controller
- controllers/my_rl_supervisor/ai_supervisor.py – PPO action broadcaster
- controllers/my_rl_supervisor/fuzzy_policies_advanced.py – modular fuzzy policies
- protos/ (optional) – place Mavic2Pro_Custom.proto here if you add Receiver/RangeFinder

You are ready. Open the world, press Play, and observe the drones moving forward, coordinating, and avoiding obstacles using fuzzy logic with a camera‑based fallback.


