import os, csv, numpy as np, matplotlib.pyplot as plt
from PIL import Image

SUB_DIR = "/home/enteekey211106/UEH_Nguyen-Tuan-Kiet"
SIM_DIR = "/home/enteekey211106/UEH_CRC_2026_Simulation_Pack"

DATA_DIR = os.path.join(SUB_DIR, "analysis", "data")
FIG_DIR = os.path.join(SUB_DIR, "analysis", "figures")
DET_DIR = os.path.join(FIG_DIR, "detections")
TRACK_IMG_PATH = os.path.join(SIM_DIR, "docs", "track_layout.png")

os.makedirs(FIG_DIR, exist_ok=True)

# Compatibility for PIL
LANCZOS_FILTER = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', getattr(Image, 'LANCZOS', 1))

def load_run(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r['elapsed_sec']) for r in rows])
    x = np.array([float(r['odom_x']) for r in rows])
    y = np.array([float(r['odom_y']) for r in rows])
    v = np.array([float(r.get('linear_x', r.get('cmd_linear_x', 0.0))) for r in rows])
    w = np.array([float(r.get('angular_z', r.get('cmd_angular_z', 0.0))) for r in rows])
    d = np.array([float(r['cumulative_distance_m']) for r in rows])
    return {
        't': t, 'x': x, 'y': y, 'v': v, 'w': w, 'd': d,
        'duration': t[-1], 'distance': d[-1],
        'mean_v': float(np.mean(v)), 'max_v': float(np.max(v)),
        'mean_w': float(np.mean(np.abs(w))), 'max_w': float(np.max(np.abs(w)))
    }

run01 = load_run(os.path.join(DATA_DIR, "run_01.csv"))
run02 = load_run(os.path.join(DATA_DIR, "run_02.csv"))

print("=== CALCULATED RUN METRICS ===")
print("Baseline (Run 01)            : Duration = %.2fs (07:00.50), Distance = %.2fm, Mean V = %.3f m/s, Max V = %.2f m/s" % (run01['duration'], run01['distance'], run01['mean_v'], run01['max_v']))
print("Correct Running Path (Run 02): Duration = %.2fs (08:20.00), Distance = %.2fm, Mean V = %.3f m/s, Max V = %.2f m/s" % (run02['duration'], run02['distance'], run02['mean_v'], run02['max_v']))

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 1. 2D ODOMETRY TRAJECTORY (trajectory_odometry.png)
fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#0f172a')
ax.plot(run01['x'], run01['y'], color='#00e5ff', linewidth=2.8,
        label='Baseline (Run 01) — %.2fm | 07:00.50' % run01['distance'], zorder=6)
ax.plot(run02['x'], run02['y'], color='#10b981', linewidth=2.4, linestyle='--',
        label='Correct Running Path (Run 02) — %.2fm | 08:20.00' % run02['distance'], zorder=7)
ax.scatter(run01['x'][0], run01['y'][0], color='#38bdf8', s=140, marker='o', edgecolors='white', linewidth=1.8, label='Start Pose (0, 0)', zorder=9)
ax.scatter(run01['x'][-1], run01['y'][-1], color='#fbbf24', s=180, marker='*', edgecolors='white', linewidth=1.8, label='Finish Loop Closure', zorder=10)
ax.grid(True, linestyle='--', color='#334155', alpha=0.6)
ax.set_xlabel('Odometry X [m]', fontsize=11, fontweight='bold', color='#ffffff')
ax.set_ylabel('Odometry Y [m]', fontsize=11, fontweight='bold', color='#ffffff')
ax.set_title('UEH CRC 2026 — 2D Odometry Trajectory Comparison', fontsize=13, fontweight='bold', color='#ffffff', pad=12)
ax.tick_params(colors='#cbd5e1')
for spine in ax.spines.values(): spine.set_color('#334155')
ax.legend(loc='upper right', facecolor='#1e293b', edgecolor='#475569', labelcolor='#ffffff')
plt.savefig(os.path.join(FIG_DIR, 'trajectory_odometry.png'), dpi=300, bbox_inches='tight', facecolor='#0b0f19')
plt.close()

# 2. SPEED PROFILE (speed_profile.png)
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#0f172a')
ax.plot(run01['t'], run01['v'], color='#00e5ff', linewidth=1.8, alpha=0.9,
        label='Baseline (Run 01) [Mean: %.3f m/s, Max: %.2f m/s]' % (run01['mean_v'], run01['max_v']))
ax.plot(run02['t'], run02['v'], color='#10b981', linewidth=1.8, alpha=0.9,
        label='Correct Running Path (Run 02) [Mean: %.3f m/s, Max: %.2f m/s]' % (run02['mean_v'], run02['max_v']))
ax.grid(True, linestyle='--', color='#334155', alpha=0.6)
ax.set_xlabel('Elapsed Time [s]', fontsize=11, fontweight='bold', color='#ffffff')
ax.set_ylabel('Linear Velocity cmd_vel [m/s]', fontsize=11, fontweight='bold', color='#ffffff')
ax.set_title('UEH CRC 2026 — Linear Speed Profile Comparison', fontsize=13, fontweight='bold', color='#ffffff', pad=12)
ax.tick_params(colors='#cbd5e1')
for spine in ax.spines.values(): spine.set_color('#334155')
ax.legend(loc='upper right', facecolor='#1e293b', edgecolor='#475569', labelcolor='#ffffff')
plt.savefig(os.path.join(FIG_DIR, 'speed_profile.png'), dpi=300, bbox_inches='tight', facecolor='#0b0f19')
plt.close()

# 3. ANGULAR VELOCITY PROFILE (angular_velocity_profile.png)
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#0f172a')
ax.plot(run01['t'], run01['w'], color='#00e5ff', linewidth=1.5, alpha=0.85, label='Baseline (Run 01) Steering')
ax.plot(run02['t'], run02['w'], color='#10b981', linewidth=1.5, alpha=0.85, label='Correct Running Path (Run 02) Steering')
ax.grid(True, linestyle='--', color='#334155', alpha=0.6)
ax.set_xlabel('Elapsed Time [s]', fontsize=11, fontweight='bold', color='#ffffff')
ax.set_ylabel('Angular Velocity cmd_vel.angular.z [rad/s]', fontsize=11, fontweight='bold', color='#ffffff')
ax.set_title('UEH CRC 2026 — Steering Angular Velocity Profile', fontsize=13, fontweight='bold', color='#ffffff', pad=12)
ax.tick_params(colors='#cbd5e1')
for spine in ax.spines.values(): spine.set_color('#334155')
ax.legend(loc='upper right', facecolor='#1e293b', edgecolor='#475569', labelcolor='#ffffff')
plt.savefig(os.path.join(FIG_DIR, 'angular_velocity_profile.png'), dpi=300, bbox_inches='tight', facecolor='#0b0f19')
plt.close()

# 4. CUMULATIVE DISTANCE (cumulative_distance.png)
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#0f172a')
ax.plot(run01['t'], run01['d'], color='#00e5ff', linewidth=2.5,
        label='Baseline (Run 01) [Total: %.2f m]' % run01['distance'])
ax.plot(run02['t'], run02['d'], color='#10b981', linewidth=2.5, linestyle='--',
        label='Correct Running Path (Run 02) [Total: %.2f m]' % run02['distance'])
ax.grid(True, linestyle='--', color='#334155', alpha=0.6)
ax.set_xlabel('Elapsed Time [s]', fontsize=11, fontweight='bold', color='#ffffff')
ax.set_ylabel('Cumulative Distance [m]', fontsize=11, fontweight='bold', color='#ffffff')
ax.set_title('UEH CRC 2026 — Cumulative Travelled Distance', fontsize=13, fontweight='bold', color='#ffffff', pad=12)
ax.tick_params(colors='#cbd5e1')
for spine in ax.spines.values(): spine.set_color('#334155')
ax.legend(loc='lower right', facecolor='#1e293b', edgecolor='#475569', labelcolor='#ffffff')
plt.savefig(os.path.join(FIG_DIR, 'cumulative_distance.png'), dpi=300, bbox_inches='tight', facecolor='#0b0f19')
plt.close()

# 5. COMPLETION TIME COMPARISON (completion_time_by_run.png)
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#0f172a')
runs = ['Baseline\n(Run 01)', 'Correct Running Path\n(Run 02)']
times = [run01['duration'], run02['duration']]
colors = ['#00e5ff', '#10b981']
bars = ax.bar(runs, times, color=colors, width=0.45, edgecolor='white', lw=1.2)
for bar, t_sec in zip(bars, times):
    m, s = divmod(t_sec, 60)
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 12,
            '%.1fs\n(%02d:%04.1f)' % (t_sec, int(m), s),
            ha='center', va='bottom', color='#ffffff', fontweight='bold', fontsize=10)
ax.set_ylim(0, 600)
ax.set_ylabel('Completion Duration [s]', fontsize=11, fontweight='bold', color='#ffffff')
ax.set_title('UEH CRC 2026 — Full Loop Completion Time Comparison', fontsize=13, fontweight='bold', color='#ffffff', pad=14)
ax.grid(True, axis='y', linestyle='--', color='#334155', alpha=0.6)
ax.tick_params(colors='#cbd5e1')
for spine in ax.spines.values(): spine.set_color('#334155')
plt.savefig(os.path.join(FIG_DIR, 'completion_time_by_run.png'), dpi=300, bbox_inches='tight', facecolor='#0b0f19')
plt.close()

# 6. SKY VIEW OVERLAY ON TRACK (sky_view_successful_runs_trajectories.png)
track_rgb = plt.imread(TRACK_IMG_PATH)
H_img, W_img = track_rgb.shape[:2]
BOARD_W = 10.0
BOARD_H = 6.0

def odom_to_pixel(ox, oy):
    wx = np.array(ox) - 4.60
    wy = np.array(oy) - 2.022
    px = (wx / BOARD_W + 0.5) * W_img
    py = (0.5 - wy / BOARD_H) * H_img
    return px, py

fig, ax = plt.subplots(figsize=(19, 11), dpi=300)
ax.imshow(track_rgb, extent=[0, W_img, H_img, 0])

landmarks = [
    ('START LINE', 80, 1004, '#10b981', 'top'),
    ('TRAFFIC LIGHT', 400, 1004, '#f59e0b', 'bottom'),
    ('CROSSWALK', 900, 1004, '#06b6d4', 'bottom'),
    ('RIGHT HAIRPIN', 1880, 750, '#a855f7', 'bottom'),
    ('INTERSECTION 1', 1450, 720, '#ec4899', 'bottom'),
    ('RAMP & TUNNEL', 980, 480, '#f97316', 'bottom'),
    ('HIGHWAY SECTOR', 700, 210, '#3b82f6', 'top'),
    ('OBSTACLE ZONE', 420, 210, '#ef4444', 'top'),
    ('FINISH LINE', 80, 680, '#10b981', 'bottom'),
]

for label, lx, ly, col, va in landmarks:
    ax.scatter(lx, ly, color=col, s=75, marker='s', edgecolors='white', linewidth=1.5, zorder=8)
    ax.text(lx, ly - 22 if va=='top' else ly + 26, label, color='#ffffff', fontsize=8.5, fontweight='bold',
            ha='center', va=va, bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f172a', edgecolor=col, alpha=0.85), zorder=9)

px1, py1 = odom_to_pixel(run01['x'], run01['y'])
px2, py2 = odom_to_pixel(run02['x'], run02['y'])

ax.plot(px1, py1, color='#00b0ff', linewidth=7.0, alpha=0.35, zorder=5)
ax.plot(px1, py1, color='#00e5ff', linewidth=3.2,
        label='Baseline (Run 01) — %.2fm | %.1fs | 100%% Loop Closed' % (run01['distance'], run01['duration']), zorder=6)

indices = np.linspace(350, len(px1) - 350, 10, dtype=int)
for ai in indices:
    dx = px1[ai + 10] - px1[ai]
    dy = py1[ai + 10] - py1[ai]
    if np.hypot(dx, dy) > 1e-3:
        ax.annotate('', xy=(px1[ai + 10], py1[ai + 10]), xytext=(px1[ai], py1[ai]),
                    arrowprops=dict(arrowstyle='->', color='#ffffff', lw=1.8), zorder=7)

ax.plot(px2, py2, color='#10b981', linewidth=2.8, linestyle='--',
        label='Correct Running Path (Run 02) — %.2fm | %.1fs | 100%% Loop Closed' % (run02['distance'], run02['duration']), zorder=7)

ax.scatter(px1[0], py1[0], color='#10b981', s=180, marker='o', edgecolors='white', linewidth=2.2, zorder=10)
ax.scatter(px1[-1], py1[-1], color='#eab308', s=240, marker='*', edgecolors='white', linewidth=2.2, zorder=10)

ax.set_title('UEH CRC 2026 — Official Reference Trajectories Map\n(Baseline vs Correct Running Path)',
             fontsize=15, fontweight='bold', color='#ffffff', pad=14)
ax.set_xlim(0, W_img)
ax.set_ylim(H_img, 0)
ax.axis('off')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True,
          facecolor='#0f172a', edgecolor='#334155', fontsize=10.5, labelcolor='#ffffff')
plt.savefig(os.path.join(FIG_DIR, 'sky_view_successful_runs_trajectories.png'), dpi=300, bbox_inches='tight', facecolor='#0b0f19')
plt.close()

# 7. CREATE 3 COMPOSITE FIGURES (NO TITLES)
def stitch_horizontal(image_paths, output_path, border_width=8, bg_color=(15, 23, 42)):
    valid_paths = [p for p in image_paths if os.path.exists(p)]
    if not valid_paths:
        print('Warning: No valid images found for', output_path)
        return
    images = [Image.open(p).convert('RGB') for p in valid_paths]
    target_h = min(img.height for img in images)
    resized_imgs = []
    for img in images:
        new_w = int(img.width * (target_h / img.height))
        resized_imgs.append(img.resize((new_w, target_h), LANCZOS_FILTER))
    
    total_w = sum(img.width for img in resized_imgs) + border_width * (len(resized_imgs) - 1)
    composite = Image.new('RGB', (total_w, target_h), color=bg_color)
    
    curr_x = 0
    for img in resized_imgs:
        composite.paste(img, (curr_x, 0))
        curr_x += img.width + border_width
    
    composite.save(output_path, quality=95)
    print('Saved composite: %s (%dx%d)' % (os.path.basename(output_path), total_w, target_h))

# Composite 1: bus, tunnel, crosswalk
fig1_paths = [
    os.path.join(DET_DIR, 'verified_bus_01.png'),
    os.path.join(DET_DIR, 'verified_tunnel_01.png'),
    os.path.join(DET_DIR, 'verified_crosswalk_01.png')
]
stitch_horizontal(fig1_paths, os.path.join(FIG_DIR, 'composite_bus_tunnel_crosswalk.png'))

# Composite 2: traffic_light_red, traffic_light_yellow, traffic_light_green
fig2_paths = [
    os.path.join(DET_DIR, 'verified_traffic_light_red_02.png'),
    os.path.join(DET_DIR, 'verified_traffic_light_yellow_01.png'),
    os.path.join(DET_DIR, 'verified_traffic_light_green_01.png')
]
stitch_horizontal(fig2_paths, os.path.join(FIG_DIR, 'composite_traffic_lights.png'))

# Composite 3: pedestrian, stop, ramp, hw_exit
fig3_paths = [
    os.path.join(DET_DIR, 'verified_pedestrian_01.png'),
    os.path.join(DET_DIR, 'verified_stop_01.png'),
    os.path.join(DET_DIR, 'verified_ramp_01.png'),
    os.path.join(DET_DIR, 'verified_hw_exit_01.png')
]
stitch_horizontal(fig3_paths, os.path.join(FIG_DIR, 'composite_pedestrian_stop_ramp_hw_exit.png'))

print('=== ALL PLOTS AND COMPOSITE FIGURES GENERATED SUCCESSFULLY ===')
