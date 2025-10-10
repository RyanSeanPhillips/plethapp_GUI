from dataclasses import dataclass

@dataclass
class Nav:
    sweep_idx: int = 0
    n_sweeps: int = 1
    window_start_s: float = 0.0
    window_dur_s: float = 10.0

    def next_sweep(self):
        self.sweep_idx = min(self.sweep_idx + 1, self.n_sweeps - 1)

    def prev_sweep(self):
        self.sweep_idx = max(self.sweep_idx - 1, 0)

    def shift_window(self, dt: float):
        self.window_start_s = max(0.0, self.window_start_s + dt)
