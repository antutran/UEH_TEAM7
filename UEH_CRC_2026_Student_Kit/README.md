# UEH CRC 2026 – Student Kit

Everything you need to start on the final round today, before you receive your car.

| Item | What it is |
|---|---|
| `crc_team/` | Race template: a complete ROS 2 package (lane following, START light, signs, obstacles, overtaking) that runs **unchanged in the simulator and on the car** |
| `crc_team/tools/` | Laptop tools: camera bird's-eye calibration, lens calibration, lane detector on saved images, colour tuning |
| `view_car.sh` | Laptop viewer for the real car: camera, RViz, rqt, ROS shell (Linux or WSL2 with the simulator image) |
| `UEH_CRC_2026_Car_Handbook.pdf` | The car handbook: read chapters 1 to 7 before the finals |
| `chessboard_9x6_25mm_A4.pdf` | Board for the lens calibration (print at 100 %) |

## 1. Run the template in the simulator (now)

You need the Student Pack from the qualifying round (Docker image `crc_sim:humble`).

1. Copy the package into the Student Pack: `cp -r crc_team UEH_CRC_2026_Student_Pack/src/`
2. Build: `bash scripts/run_docker.sh compile`
3. Start the simulator: `bash scripts/run_docker.sh up` (or `up-headless`)
4. In a second terminal: `bash scripts/run_docker.sh sh`, then
   `ros2 launch crc_team race.launch.py config:=sim`

The status line every second shows what the car does, e.g.
`DRIVE lines=LR offset=+0.012 m frames=20/s lane=2.4 ms`.

What the template already does in the simulator: lane following on straights and bends, junction
crossings, the zebra crossing, the ramp, the dark tunnel and its bend. What it does not do (your
job): choose a route at junctions, read the signs (it reacts to `/signs` on the car, chapter 10),
collect diamonds, handle the other car. Right after START it can get lost in the merge and the first
junction; to watch a longer run, start the robot on the bottom straight (Student Pack README, launch arguments
`x:=-2.5 y:=-1.68 yaw:=0.0`).

## 2. Make it yours

- Rename the package if you like, then improve `crc_team/race_node.py` (state machine) and
  `crc_team/lane.py` (lane detector). Every number is in `config/race_sim.yaml` and
  `config/race_car.yaml`: tune there, not in the code.
- Keep everything in your own package: on the car there is no `crc_sim` package and no ground
  truth (`/sky_cam`, `/model_states`, ...).
- The final track has **400 mm lanes and an 800 mm road** (the simulator has 350 mm):
  `lane_width` is already 0.425 (line centre to line centre) in `race_car.yaml`.
- Test the lane detector on saved images without driving:
  `python3 crc_team/tools/lane_debug.py 'frames/*.jpg' --config crc_team/config/race_sim.yaml`
  (needs `pip install opencv-python numpy pyyaml`).

## 3. On the car (finals)

Copy the same folder to the car and follow the handbook: chapter 1 (first 30 minutes),
chapter 6 (build and run: `car compile`, `car start "ros2 launch crc_team race.launch.py"`),
chapter 8 (camera calibration, required) and chapter 12 (the template). The car also has this
package in `~/crc_car/examples/crc_team`.
