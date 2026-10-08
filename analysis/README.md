# UEH CRC 2026 Simulation Run Analysis & Measurement Suite

This package provides an automated, reproducible, and strictly compliant measurement framework for evaluating the autonomous robot controller in the UEH Creative Robot Contest (CRC) 2026 Simulation Round.

---

## 1. Package Structure

```
analysis/
??? README.md               # Complete documentation of measurement methodology
??? record_run.py           # ROS 2 measurement and data logging node
??? summarize_runs.py       # Computes summary statistics from recorded CSVs
??? plot_runs.py            # Generates publication-ready figures
??? run_experiments.py      # Automated lifecycle runner for measurement runs
??? data/                   # Raw measured CSV data and run metadata JSONs
?   ??? run_01.csv
?   ??? run_01_metadata.json
?   ??? ...
?   ??? run_summary.csv
?   ??? summary_statistics.csv
??? figures/                # High-resolution report figures
    ??? completion_time_by_run.png
    ??? speed_profile.png
    ??? angular_velocity_profile.png
    ??? cumulative_distance.png
    ??? trajectory_odometry.png
    ??? detections/         # Screenshots of detected valid objects
```

---

## 2. Permitted ROS Topics & Data Integrity Rules

The analysis framework operates exclusively through participant-allowed interfaces and does **NOT** access ground-truth simulator state:

### Allowed Topics Used for Measurement:
- `/odom` (`nav_msgs/msg/Odometry`): Odometry position, orientation quaternion, and linear/angular velocity.
- `/cmd_vel` (`geometry_msgs/msg/Twist`): Control commands issued by the autonomous controller.
- `/imu` (`sensor_msgs/msg/Imu`): Inertial measurement unit data.
- `/detection_debug/image` (`sensor_msgs/msg/Image`): Visual perception debug overlay for logging detection screenshots (max 5 per class).

### Strictly Prohibited Topics & Services (NOT USED):
- **NO** `/model_states`, `/link_states`, `/sky_cam/*`, `/traffic_lights`, `/traffic_light/*`, or `/automobile/semaphores`.
- **NO** Gazebo service calls: `/get_entity_state`, `/set_entity_state`, `/spawn_entity`, `/delete_entity`.
- **NO** Gazebo introspection or ground-truth world/model file coordinate lookups at runtime.
- **NO** code modification or parameter tuning in `starter_node.py`. The controller is evaluated strictly in its final submitted form.

---

## 3. Measurement Methodology

### A. Run Start-Time Detection
- Simulator startup and initial loading times are excluded.
- Measurement start ($t = 0.0\text{ s}$, `elapsed_sec = 0.0`) is automatically triggered upon detecting sustained robot motion:
  $$\left|\text{cmd\_vel.linear.x}\right| > 0.005\text{ m/s} \quad \text{OR} \quad \left|\text{cmd\_vel.angular.z}\right| > 0.01\text{ rad/s}$$
  confirmed across $\ge 3$ consecutive samples.
- The robot's initial odometry pose $(x_0, y_0, \theta_0)$ is recorded at this instant as the reference origin.

### B. Cumulative Travelled Distance
- Calculated strictly by integrating Euclidean steps between consecutive `/odom` positions:
  $$d_{\text{cum}} = \sum_{k=1}^N \sqrt{(x_k - x_{k-1})^2 + (y_k - y_{k-1})^2}$$

### C. Legal Loop Completion Detection
- Because the UEH CRC 2026 course is a closed loop starting and ending at the bottom-left start lane, finish detection is determined by returning to the initial odometry pose:
  1. **Lockout window:** Completion detection is disabled for the first $300.0\text{ s}$ (5 minutes) to prevent false triggering near the start line.
  2. **Minimum distance:** Requires $d_{\text{cum}} \ge 30.0\text{ m}$ of track progression.
  3. **Spatial & Heading thresholds:**
     $$\sqrt{(x_k - x_0)^2 + (y_k - y_0)^2} \le 0.60\text{ m}$$
     $$\left|\text{wrap\_angle}(\theta_k - \theta_0)\right| \le 0.60\text{ rad} \approx 34^\circ$$
  4. **Persistence:** The arrival condition must persist for at least $12$ consecutive odometry samples ($\approx 0.6\text{ s}$) to filter transient noise.

### D. Safety Timeout Rule
- A strict hard timeout is enforced at $540.0\text{ s}$ (9 minutes) from motion start.
- If the robot has not met the completion criterion by this deadline, the run is terminated, saved, and marked:
  $$\text{completed} = \text{False}, \quad \text{timeout} = \text{True}$$
  Timeouts are **never** reported as successful completion times.

### E. Sanity Check vs Ground Truth
- Prior recorded run observations (~6:30 to 7:00) serve **strictly** as contextual sanity checks for expected time ranges. No values are estimated, fabricated, or manually inserted. All reported metrics come directly from live simulation measurements.

---

## 4. How to Reproduce Measurements

### Step 1: Run Automated Experiment Suite
To execute all measurement runs (1 pilot run + 4 subsequent valid runs, up to max 7 attempts), calculate SHA256 before and after, and generate all outputs:
```bash
python3 analysis/record_run.py --run-id my_run_01
```

### Step 2: Regenerate Summary Statistics
To recalculate summary statistics from existing CSV data in `analysis/data/`:
```bash
python3 analysis/summarize_runs.py --data-dir analysis/data
```
Outputs:
- `analysis/data/run_summary.csv`
- `analysis/data/summary_statistics.csv`

### Step 3: Regenerate Report Figures
To regenerate all publication-ready figures in `analysis/figures/`:
```bash
python3 analysis/plot_runs.py --data-dir analysis/data --figures-dir analysis/figures
```
Outputs:
- `analysis/figures/completion_time_by_run.png`
- `analysis/figures/speed_profile.png`
- `analysis/figures/angular_velocity_profile.png`
- `analysis/figures/cumulative_distance.png`
- `analysis/figures/trajectory_odometry.png`
- `analysis/figures/detections/*.png`

---

## 5. Metrics Definition

| Metric | Definition | Unit |
| :--- | :--- | :--- |
| `duration_sec` | Total wall-clock time from motion start to loop completion | seconds ($\text{s}$) |
| `duration_min` | Formatted completion time (`mm:ss.ss`) | minutes:seconds |
| `cumulative_distance_m` | Total integrated path distance from odometry | meters ($\text{m}$) |
| `mean_linear_speed_mps` | Mean linear velocity command during the run | $\text{m/s}$ |
| `max_linear_speed_mps` | Maximum linear velocity command | $\text{m/s}$ |
| `mean_abs_angular_speed_rps` | Mean absolute steering angular velocity | $\text{rad/s}$ |
| `max_abs_angular_speed_rps` | Maximum steering angular velocity | $\text{rad/s}$ |
| `success_rate` | Ratio of valid completed runs to attempted runs | percentage ($\%$) |
