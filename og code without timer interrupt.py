import csv
import threading
import time
from collections import deque
from datetime import datetime

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector

try:
    import serial

    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("pyserial not found – running in demo mode (no real sensor data).")

# ── Constants ────────────────────────────────────────────────────────────────
BUFFER_LEN = 1000  # samples kept in each deque
ANIM_INTERVAL_MS = 50  # animation redraw period (20 fps)
DEFAULT_PORT = "COM4"
DEFAULT_BAUD = 115200

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ─────────────────────────────────────────────────────────────────────────────
class SensorDashboard(ctk.CTk):

    # ── Init ──────────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.title("ESP32 Sensor Real-Time Dashboard")
        self.geometry("1400x850")
        self.minsize(900, 550)

        # ── Serial state ────────────────────────────────────────────────────
        self.port = DEFAULT_PORT
        self.ser = None
        self.program_running = True

        # ── Recording state ─────────────────────────────────────────────────
        self.is_recording = False
        self.recording_number = 0
        self.recording_start_time = 0.0
        self.samples_saved = 0

        # ── Settings (kept in sync with dropdowns) ──────────────────────────
        self.target_hz = 120
        self.record_duration = 5.0

        # ── Single-session CSV (created once at startup) ─────────────────────
        self.csv_filename = (
            f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        self.csv_file = open(self.csv_filename, mode="w", newline="", buffering=1)
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["Recording#", "Timestamp",
             "AccX", "AccY", "AccZ",
             "GyroX", "GyroY", "GyroZ",
             "FSR", "Piezo"]
        )

        # ── Ring-buffer data (written only by serial thread) ────────────────
        self.acc_x = deque(maxlen=BUFFER_LEN)
        self.acc_y = deque(maxlen=BUFFER_LEN)
        self.acc_z = deque(maxlen=BUFFER_LEN)
        self.gyro_x = deque(maxlen=BUFFER_LEN)
        self.gyro_y = deque(maxlen=BUFFER_LEN)
        self.gyro_z = deque(maxlen=BUFFER_LEN)
        self.fsr = deque(maxlen=BUFFER_LEN)
        self.piezo = deque(maxlen=BUFFER_LEN)
        self.latest_data = [0.0] * 8
        self.data_lock = threading.Lock()

        # ── Zoom state ───────────────────────────────────────────────────────
        self.zoom_mode = False
        self._selectors = []
        self._axis_zoomed = {}

        # ── Build UI then plots, then connect ────────────────────────────────
        self._setup_ui()
        self._setup_plots()
        self._connect_serial()

        # Background serial reader
        self.thread = threading.Thread(
            target=self._serial_reader_thread, daemon=True
        )
        self.thread.start()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _setup_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left – plots
        self.plot_frame = ctk.CTkFrame(self, corner_radius=10)
        self.plot_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Right – scrollable controls
        self.ctrl = ctk.CTkScrollableFrame(self, corner_radius=10, width=230)
        self.ctrl.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")

        # ── Connection status ────────────────────────────────────────────────
        self.lbl_status = ctk.CTkLabel(
            self.ctrl,
            text=f"🟡 Not Connected | {self.port}",
            font=("Arial", 13, "bold"),
            text_color="#fbc531",
            wraplength=210,
        )
        self.lbl_status.pack(pady=(18, 8))

        # ── Recording ────────────────────────────────────────────────────────
        ctk.CTkLabel(self.ctrl, text="Recording", font=("Arial", 13, "bold")).pack(pady=(10, 2))
        self.btn_start = ctk.CTkButton(
            self.ctrl,
            text="▶  Start (5 s)",
            font=("Arial", 15, "bold"),
            fg_color="#2FA572", hover_color="#217852",
            command=self._toggle_recording,
        )
        self.btn_start.pack(pady=6, padx=18, fill="x")

        # ── Live values ───────────────────────────────────────────────────────
        ctk.CTkLabel(self.ctrl, text="Live Values", font=("Arial", 13, "bold")).pack(pady=(14, 4))
        vf = ctk.CTkFrame(self.ctrl, fg_color="transparent")
        vf.pack(padx=18, fill="x")
        vf.columnconfigure(1, weight=1)

        self.live_labels = {}
        sensor_names = ["Acc X", "Acc Y", "Acc Z",
                        "Gyro X", "Gyro Y", "Gyro Z",
                        "FSR", "Piezo"]
        for i, name in enumerate(sensor_names):
            ctk.CTkLabel(vf, text=name, anchor="w").grid(
                row=i, column=0, sticky="w", padx=4, pady=2
            )
            lbl = ctk.CTkLabel(vf, text="0.00", anchor="e")
            lbl.grid(row=i, column=1, sticky="e", padx=4, pady=2)
            self.live_labels[name] = lbl

        # ── Settings ─────────────────────────────────────────────────────────
        ctk.CTkLabel(self.ctrl, text="⚙  Settings", font=("Arial", 13, "bold")).pack(pady=(14, 4))

        ctk.CTkLabel(self.ctrl, text="Baud Rate").pack(anchor="w", padx=18)
        self.combo_baud = ctk.CTkComboBox(
            self.ctrl, values=["9600", "115200", "921600"]
        )
        self.combo_baud.set("115200")
        self.combo_baud.pack(pady=(0, 8), padx=18, fill="x")

        ctk.CTkLabel(self.ctrl, text="Target Rate (Hz)").pack(anchor="w", padx=18)
        self.combo_rate = ctk.CTkComboBox(
            self.ctrl, values=["60", "100", "120"],
            command=self._update_settings,
        )
        self.combo_rate.set("120")
        self.combo_rate.pack(pady=(0, 8), padx=18, fill="x")

        ctk.CTkLabel(self.ctrl, text="Recording Time (s)").pack(anchor="w", padx=18)
        self.combo_time = ctk.CTkComboBox(
            self.ctrl, values=["5", "10", "30", "60"],
            command=self._update_settings,
        )
        self.combo_time.set("5")
        self.combo_time.pack(pady=(0, 8), padx=18, fill="x")

        # ── CSV file info ─────────────────────────────────────────────────────
        ctk.CTkLabel(self.ctrl, text="Output File", font=("Arial", 12, "bold")).pack(pady=(14, 2))
        ctk.CTkLabel(
            self.ctrl,
            text=self.csv_filename,
            font=("Arial", 10),
            text_color="#aaaaaa",
            wraplength=210,
        ).pack(padx=10)

        # ── Status footer ─────────────────────────────────────────────────────
        self.lbl_footer = ctk.CTkLabel(
            self.ctrl, text="Ready", font=("Arial", 11),
            text_color="#aaaaaa", wraplength=210,
        )
        self.lbl_footer.pack(pady=(20, 10))

    # ── Plots ─────────────────────────────────────────────────────────────────
    def _setup_plots(self):
        self.chart_toolbar = ctk.CTkFrame(
            self.plot_frame, fg_color="#1a1a1a", height=38, corner_radius=0
        )
        self.chart_toolbar.pack(side="top", fill="x")
        self.chart_toolbar.pack_propagate(False)

        self.btn_zoom = ctk.CTkButton(
            self.chart_toolbar,
            text="🔍  Zoom",
            font=("Arial", 12),
            width=100, height=28,
            fg_color="#3a3a3a", hover_color="#4a4a4a",
            corner_radius=6,
            command=self._toggle_zoom,
        )
        self.btn_zoom.pack(side="left", padx=(8, 4), pady=5)

        self.btn_reset_zoom = ctk.CTkButton(
            self.chart_toolbar,
            text="↩  Reset",
            font=("Arial", 12),
            width=90, height=28,
            fg_color="#3a3a3a", hover_color="#4a4a4a",
            corner_radius=6,
            command=self._reset_zoom,
        )
        self.btn_reset_zoom.pack(side="left", padx=(0, 8), pady=5)

        ctk.CTkLabel(
            self.chart_toolbar, text="|", font=("Arial", 14), text_color="#555555"
        ).pack(side="left")

        self.lbl_zoom_hint = ctk.CTkLabel(
            self.chart_toolbar,
            text="  Drag on a chart to zoom  ·  double-click to reset",
            font=("Arial", 11),
            text_color="#666666",
        )
        self.lbl_zoom_hint.pack(side="left", padx=8)

        # 4 Subplots (Acc, Gyro, FSR, Piezo)
        plt.style.use("dark_background")
        self.fig, axes = plt.subplots(4, 1, figsize=(8, 8), dpi=100)
        self.ax_acc, self.ax_gyro, self.ax_fsr, self.ax_piezo = axes
        self.fig.patch.set_facecolor("#2b2b2b")
        for ax in axes:
            ax.set_facecolor("#1e1e1e")

        self._default_ylims = {
            self.ax_acc: (-20, 20),
            self.ax_gyro: (-5, 5),
            self.ax_fsr: (-10, 4200),
            self.ax_piezo: (-10, 4200),
        }
        self._axis_zoomed = {ax: False for ax in axes}

        # 1. Accelerometer
        (self.l_ax,) = self.ax_acc.plot([], [], "#ff5252", lw=0.9, label="Acc X")
        (self.l_ay,) = self.ax_acc.plot([], [], "#4cd137", lw=0.9, label="Acc Y")
        (self.l_az,) = self.ax_acc.plot([], [], "#00a8ff", lw=0.9, label="Acc Z")
        self.ax_acc.set_title("Accelerometer (m/s²)", color="white", fontsize=10)
        self.ax_acc.set_ylim(-20, 20)
        self.ax_acc.legend(loc="upper right", fontsize=8, framealpha=0.3)
        self.ax_acc.grid(True, alpha=0.15)

        # 2. Gyroscope
        (self.l_gx,) = self.ax_gyro.plot([], [], "#e84118", lw=0.9, label="Gyro X")
        (self.l_gy,) = self.ax_gyro.plot([], [], "#fbc531", lw=0.9, label="Gyro Y")
        (self.l_gz,) = self.ax_gyro.plot([], [], "#0097e6", lw=0.9, label="Gyro Z")
        self.ax_gyro.set_title("Gyroscope (rad/s)", color="white", fontsize=10)
        self.ax_gyro.set_ylim(-5, 5)
        self.ax_gyro.legend(loc="upper right", fontsize=8, framealpha=0.3)
        self.ax_gyro.grid(True, alpha=0.15)

        # 3. FSR
        (self.l_fsr,) = self.ax_fsr.plot([], [], "#e1b12c", lw=0.9, label="FSR")
        self.ax_fsr.set_title("FSR Sensor", color="white", fontsize=10)
        self.ax_fsr.set_ylim(-10, 4200)
        self.ax_fsr.legend(loc="upper right", fontsize=8, framealpha=0.3)
        self.ax_fsr.grid(True, alpha=0.15)

        # 4. Piezo
        (self.l_piezo,) = self.ax_piezo.plot([], [], "#8c7ae6", lw=0.9, label="Piezo")
        self.ax_piezo.set_title("Piezo Sensor", color="white", fontsize=10)
        self.ax_piezo.set_ylim(-10, 4200)
        self.ax_piezo.legend(loc="upper right", fontsize=8, framealpha=0.3)
        self.ax_piezo.grid(True, alpha=0.15)

        self.fig.tight_layout(pad=1.2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        self.ani = FuncAnimation(
            self.fig, self._update_plots,
            interval=ANIM_INTERVAL_MS,
            cache_frame_data=False,
        )

    # ── Zoom ──────────────────────────────────────────────────────────────────
    def _toggle_zoom(self):
        self.zoom_mode = not self.zoom_mode
        if self.zoom_mode:
            self.btn_zoom.configure(
                text="🔍  Zoom  ✓", fg_color="#1a6fa8", hover_color="#155989"
            )
            self.lbl_zoom_hint.configure(
                text="  Click & drag to select area  ·  double-click to reset",
                text_color="#4da8e8",
            )
            self._create_selectors()
        else:
            self.btn_zoom.configure(
                text="🔍  Zoom", fg_color="#3a3a3a", hover_color="#4a4a4a"
            )
            self.lbl_zoom_hint.configure(
                text="  Drag on a chart to zoom  ·  double-click to reset",
                text_color="#666666",
            )
            self._destroy_selectors()

    def _create_selectors(self):
        self._destroy_selectors()
        for ax in (self.ax_acc, self.ax_gyro, self.ax_fsr, self.ax_piezo):
            def _cb(eclick, erelease, _ax=ax):
                self._on_rect_select(eclick, erelease, _ax)

            sel = RectangleSelector(
                ax, _cb, useblit=False, button=[1],
                minspanx=5, minspany=5, spancoords="pixels",
                interactive=False,
                props=dict(facecolor="cyan", edgecolor="white", alpha=0.15, fill=True, linewidth=1),
            )
            self._selectors.append(sel)

    def _destroy_selectors(self):
        for sel in self._selectors:
            try:
                sel.set_active(False)
                sel.disconnect_events()
            except Exception:
                pass
        self._selectors.clear()

    def _on_rect_select(self, eclick, erelease, ax):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        if abs(x2 - x1) < 1e-6 or abs(y2 - y1) < 1e-6:
            return
        ax.set_xlim(min(x1, x2), max(x1, x2))
        ax.set_ylim(min(y1, y2), max(y1, y2))
        self._axis_zoomed[ax] = True
        self.canvas.draw_idle()

    def _reset_zoom(self):
        with self.data_lock:
            n = len(self.acc_x)
        for ax in (self.ax_acc, self.ax_gyro, self.ax_fsr, self.ax_piezo):
            ax.set_xlim(0, max(n, 10))
            ax.set_ylim(*self._default_ylims[ax])
            self._axis_zoomed[ax] = False
        self.canvas.draw_idle()

    def _on_canvas_click(self, event):
        if event.dblclick:
            self._reset_zoom()

    # ── Serial & Commands ─────────────────────────────────────────────────────
    def _connect_serial(self):
        if not SERIAL_AVAILABLE:
            self.lbl_status.configure(
                text="⚠ Demo mode (no pyserial)", text_color="#fbc531"
            )
            return
        baud = int(self.combo_baud.get())
        try:
            self.ser = serial.Serial(self.port, baud, timeout=0.1)
            # Reset ESP32 on startup
            self.ser.dtr = False;
            self.ser.rts = False
            time.sleep(0.3)
            self.ser.dtr = True;
            self.ser.rts = True
            time.sleep(0.5)
            self.ser.flushInput()

            # Update UI state
            self.lbl_status.configure(
                text=f"🟢 Connected | {self.port} | {baud}",
                text_color="#2FA572",
            )

            # Send initial configured rate to ESP32 upon connection
            self._send_rate_command(self.target_hz)

        except Exception as exc:
            self.lbl_status.configure(
                text=f"🔴 Disconnected | {self.port}",
                text_color="#ff5252",
            )
            print(f"Serial error: {exc}")

    def _send_rate_command(self, target_hz: int):
        """Transmit sampling rate command string to ESP32 (e.g. 'RATE:120\\n')."""
        if self.ser and self.ser.is_open:
            try:
                cmd = f"RATE:{target_hz}\n"
                self.ser.write(cmd.encode("utf-8"))
                print(f"--> Transmitted command to ESP32: {cmd.strip()}")
            except Exception as exc:
                print(f"Failed to write serial command: {exc}")

    def _serial_reader_thread(self):
        while self.program_running:
            try:
                if not (self.ser and self.ser.is_open):
                    time.sleep(0.05)
                    continue

                if self.ser.in_waiting == 0:
                    time.sleep(0.001)
                    continue

                raw = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue

                parts = raw.split(",")
                if len(parts) != 8:
                    continue

                try:
                    data = [float(p) for p in parts]
                except ValueError:
                    continue

                # Update shared buffers
                with self.data_lock:
                    self.acc_x.append(data[0]);
                    self.acc_y.append(data[1]);
                    self.acc_z.append(data[2])
                    self.gyro_x.append(data[3]);
                    self.gyro_y.append(data[4]);
                    self.gyro_z.append(data[5])
                    self.fsr.append(data[6]);
                    self.piezo.append(data[7])
                    self.latest_data = data

                # Write to CSV if recording (time-based check)
                if self.is_recording:
                    elapsed = time.time() - self.recording_start_time
                    if elapsed >= self.record_duration:
                        self.after(0, self._stop_recording)
                        continue
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    self.csv_writer.writerow([self.recording_number, ts] + data)
                    self.samples_saved += 1

            except Exception as exc:
                if self.program_running:
                    self.after(
                        0,
                        lambda: self.lbl_status.configure(
                            text="🔴 Disconnected", text_color="#ff5252"
                        ),
                    )
                    print(f"Serial thread error: {exc}")
                time.sleep(1)

    # ── Settings / Recording ──────────────────────────────────────────────────
    def _update_settings(self, _=None):
        new_hz = int(self.combo_rate.get())

        # If rate changed, write the new configuration over serial
        if new_hz != self.target_hz:
            self.target_hz = new_hz
            self._send_rate_command(self.target_hz)

        self.record_duration = float(self.combo_time.get())

        if not self.is_recording:
            self.btn_start.configure(text=f"▶  Start ({int(self.record_duration)} s)")

    def _toggle_recording(self):
        if not self.is_recording:
            self.recording_number += 1
            self.samples_saved = 0
            self._update_settings()
            self.recording_start_time = time.time()
            self.is_recording = True
            self.btn_start.configure(
                text="⏹  Stop Recording",
                fg_color="#e84118", hover_color="#c23616",
            )
        else:
            self._stop_recording()

    def _stop_recording(self):
        self.is_recording = False
        try:
            self.csv_file.flush()
        except Exception:
            pass
        self.btn_start.configure(
            text=f"▶  Start ({int(self.record_duration)} s)",
            fg_color="#2FA572", hover_color="#217852",
        )
        self.lbl_footer.configure(
            text=(
                f"✅ Rec #{self.recording_number} done\n"
                f"{self.samples_saved} samples → {self.csv_filename}"
            )
        )

    # ── Animation callback ────────────────────────────────────────────────────
    def _update_plots(self, _frame):
        with self.data_lock:
            if not self.acc_x:
                return
            ax_d = list(self.acc_x);
            ay_d = list(self.acc_y);
            az_d = list(self.acc_z)
            gx_d = list(self.gyro_x);
            gy_d = list(self.gyro_y);
            gz_d = list(self.gyro_z)
            fsr_d = list(self.fsr);
            pz_d = list(self.piezo)
            latest = list(self.latest_data)

        n = len(ax_d)
        xs = range(n)

        # Update line data
        self.l_ax.set_data(xs, ax_d);
        self.l_ay.set_data(xs, ay_d);
        self.l_az.set_data(xs, az_d)
        self.l_gx.set_data(xs, gx_d);
        self.l_gy.set_data(xs, gy_d);
        self.l_gz.set_data(xs, gz_d)
        self.l_fsr.set_data(xs, fsr_d)
        self.l_piezo.set_data(xs, pz_d)

        xlim_max = max(n, 10)
        for ax in (self.ax_acc, self.ax_gyro, self.ax_fsr, self.ax_piezo):
            if not self._axis_zoomed.get(ax):
                ax.set_xlim(0, xlim_max)

        names = ["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z", "FSR", "Piezo"]
        for i, name in enumerate(names):
            self.live_labels[name].configure(text=f"{latest[i]:.2f}")

        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            pct = min(100.0, elapsed / self.record_duration * 100.0)
            self.lbl_footer.configure(
                text=(
                    f"🔴 Rec #{self.recording_number} "
                    f"{elapsed:.1f}/{self.record_duration:.0f} s  "
                    f"({pct:.0f}%)\n"
                    f"{self.samples_saved} samples collected"
                )
            )

    # ── Clean shutdown ────────────────────────────────────────────────────────
    def on_closing(self):
        print("Shutting down …")
        self.program_running = False

        if hasattr(self, "ani") and self.ani.event_source is not None:
            try:
                self.ani.event_source.stop()
            except Exception:
                pass

        if hasattr(self, "fig"):
            try:
                plt.close(self.fig)
            except Exception:
                pass

        try:
            for after_id in self.tk.call("after", "info"):
                try:
                    self.after_cancel(after_id)
                except Exception:
                    pass
        except Exception:
            pass

        self._destroy_selectors()

        if self.thread.is_alive():
            self.thread.join(timeout=1.5)

        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass

        if not self.csv_file.closed:
            try:
                self.csv_file.flush()
                self.csv_file.close()
            except Exception:
                pass

        self.quit()
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SensorDashboard()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()