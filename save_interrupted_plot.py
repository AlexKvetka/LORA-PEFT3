import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === Paste your losses and steps here ===
# Example:
# losses = [0.9, 0.8, 0.7, ...]
# steps = [0, 100, 200, ...]
losses = [0.9, 0.8, 0.7]  # <-- replace with your collected losses
steps = [0, 100, 200]     # <-- replace with your collected steps

os.makedirs("plots", exist_ok=True)
plt.figure()
plt.plot(steps, losses, label="Train Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve (Interrupted)")
plt.legend()
loss_plot_path = "plots/loss_interrupted.png"
plt.savefig(loss_plot_path)
plt.close()
print(f"Plot saved to {loss_plot_path}")
