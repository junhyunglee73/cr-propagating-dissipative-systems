import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ========================================
# System Configuration & Geometry
# ========================================
HEIGHT, WIDTH = 800, 800
dt=1
T_steps = 400
Total_steps = int(T_steps / dt)
cy, cx = HEIGHT // 2, WIDTH // 2
y, x = np.ogrid[:HEIGHT, :WIDTH]
dist_map = np.sqrt((x - cx)**2 + (y - cy)**2)

# ========================================
# Spatial Structural Field (beta) & Input Force magnitude (A)
# ========================================
beta_field = np.ones((HEIGHT, WIDTH))
input_force_magnitude = 5

A = input_force_magnitude * beta_field 

# ========================================
# Physical Parameters (The Causes)
# ========================================
stiffness_k = 0.2
mass_m = 50
damping_c = 0.6
V_record = 1.0
    

gamma = damping_c / (2 * mass_m)
w_squared = (stiffness_k / mass_m) - gamma**2
natural_freq = np.sqrt(w_squared) if w_squared > 0 else 0.0

# ========================================
# Causal Operators & Hookean State Operator
# ========================================
def H_time(tau):
    return (tau >= 0).astype(float)

def hookean_state_operator(tau):
    return np.cos(natural_freq * tau)

# ========================================
# Governing Equation (Causal Recording Kernel)
# ========================================
def causal_record_kernel(t_current, r_dist):
    tau = t_current - (r_dist / V_record)
    return np.exp(-gamma * tau) * hookean_state_operator(tau) * H_time(tau)


# ========================================
# Front finder 
# ========================================
x_idx = np.arange(WIDTH)
dist_line = dist_map[cy, :]

def find_front_x(current_time):
    ct_line = current_time[cy, :] if np.ndim(current_time) == 2 else current_time
    tau_line = ct_line - dist_line / V_record
    reached = (tau_line >= 0) & (x_idx >= cx)     
    return int(x_idx[reached].max()) if reached.any() else cx

# ========================================
# Time Evolution & Snapshot Recording
# ========================================
snap_surface_2d    = []
snap_cross_1d      = []
snap_lagrangian_1d = []
capture_times      = []
capture_front_x    = []      # front at each snapshot
capture_front_val  = []      # force at front for each snapshot
capture_steps = [50, 250, 400]

front_x_hist   = []          # front position history over time
front_val_hist = []          # force at front history over time

# Source-position time histories: x = cx, y = cy
source_causal_hist = []
source_lagrangian_hist = []
source_conservative_hist = []

for step in range(Total_steps + 1):
    t_eff = step * dt 
    current_time = t_eff * beta_field
    theta = current_time - (dist_map / V_record)

    surface_record = A * causal_record_kernel(current_time, dist_map)
    lagrangian = A * np.exp(-gamma * theta) * hookean_state_operator(theta)
    
    # Record the response at the source position
    source_causal_hist.append(surface_record[cy, cx])
    source_lagrangian_hist.append(lagrangian[cy, cx])

    # Undamped conservative reference (c = 0)
    omega_0 = np.sqrt(stiffness_k / mass_m)
    source_conservative_hist.append(
        A[cy, cx] * np.cos(omega_0 * t_eff)
    )

    # Find front position and force
    fx = find_front_x(current_time)
    fv = surface_record[cy, fx]
    front_x_hist.append(fx)
    front_val_hist.append(fv)

    if step in capture_steps:
        snap_surface_2d.append(surface_record.copy())
        snap_cross_1d.append(surface_record[cy, :].copy())
        snap_lagrangian_1d.append(lagrangian[cy, :].copy())
        capture_times.append(step)
        capture_front_x.append(fx)
        capture_front_val.append(fv)
        print(f">> Snapshot step {step}: t_eff={t_eff:.1f}, front_x={fx}, force={fv:.2f}")

# ========================================
# Visualization
# ========================================
num_snaps = len(snap_surface_2d)
fig, axes = plt.subplots(2, num_snaps, figsize=(16, 8), constrained_layout=True)
VISUAL_LIMIT = input_force_magnitude * 1.3
norm = TwoSlopeNorm(vmin=-VISUAL_LIMIT, vcenter=0, vmax=VISUAL_LIMIT)

for i in range(num_snaps):
    im1 = axes[0, i].imshow(snap_surface_2d[i], cmap="seismic", norm=norm, origin="lower")
    axes[0, i].set_title(f"Surface Record (t={capture_times[i]})")
    axes[0, i].axis("off")
    axes[0, i].axhline(cy, color="black", linestyle="--", alpha=0.3)

    axes[1, i].plot(np.arange(WIDTH), snap_cross_1d[i], color="black", linewidth=1.5, label="Causal Record")
    axes[1, i].plot(np.arange(WIDTH), snap_lagrangian_1d[i], color="red", linestyle="--", linewidth=1.5, label="Lagrangian Path")

    # front position and force annotation
    front_x = capture_front_x[i]
    force_value = capture_front_val[i]

    axes[1, i].scatter(front_x, force_value, color="blue", s=40, zorder=5, label="Front force")
    axes[1, i].axvline(front_x, color="blue", linestyle=":", alpha=0.4)
    axes[1, i].text(min(front_x + 10, WIDTH - 150), force_value + 1.5,
                    f"Force={force_value:.2f}", color="blue", fontsize=10)

    axes[1, i].set_xlim(0, WIDTH)
    axes[1, i].set_ylim(-VISUAL_LIMIT * 1.3, VISUAL_LIMIT * 1.3)
    axes[1, i].grid(True, linestyle=":", alpha=0.5)
    axes[1, i].legend()

cbar = fig.colorbar(im1, ax=axes[0, :], location="right", fraction=0.015, pad=0.02)

# ========================================
# Track front position and force over time
# ========================================
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
steps_arr = np.arange(Total_steps + 1)
ax2[0].plot(steps_arr, front_x_hist, color="blue")
ax2[0].set_xlabel("step"); ax2[0].set_ylabel("front x position")
ax2[0].set_title("Front position"); ax2[0].grid(True, linestyle=":")

ax2[1].plot(steps_arr, front_val_hist, color="blue")
ax2[1].set_xlabel("step"); ax2[1].set_ylabel("Force at front")
ax2[1].set_title("Force at propagating front"); ax2[1].grid(True, linestyle=":")

print("damping_c =", damping_c, "| gamma =", gamma, "| lambda1 term:", A[cy, cx + 50])
print("center @ last snapshot:", snap_cross_1d[-1][cx])
fig.savefig("check_run.png", dpi=100)

# ========================================
# Source-Position Comparison
# ========================================
time_axis = np.arange(Total_steps + 1) * dt

fig3, ax3 = plt.subplots(figsize=(11, 4.5), constrained_layout=True)

# Undamped conservative reference
ax3.plot(
    time_axis,
    source_conservative_hist,
    color="green",
    linewidth=1.5,
    label="Conservative (c=0.0)"
)

# Lagrangian path
ax3.plot(
    time_axis,
    source_lagrangian_hist,
    color="red",
    linestyle="--",
    linewidth=1.8,
    label=f"Lagrangian Path (c={damping_c})",
    zorder=3
)

# Causal record
ax3.plot(
    time_axis,
    source_causal_hist,
    color="blue",
    linestyle=":",
    linewidth=2,
    label=f"Causal Record (c={damping_c})",
    zorder=4
)

ax3.set_xlabel("Time")
ax3.set_ylabel("Response at source position")
ax3.set_title(f"Source-Position Dynamics at x={cx}, y={cy}")
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.legend()

fig3.savefig("source_position_comparison.png", dpi=150)

plt.show()