import time
import threading
import warnings
import numpy as np
import cflib.crtp
from cflib.crazyflie.swarm import Swarm, CachedCfFactory
from cflib.crazyflie.log import LogConfig

warnings.simplefilter("ignore", DeprecationWarning)

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
URIS = [
    'udp://127.0.0.1:19850',  # D0: ( 0.0,  0.0)
    'udp://127.0.0.1:19851',  # D1: ( 2.0, -1.5)
    'udp://127.0.0.1:19852',  # D2: (-1.5,  1.0)
]

TARGET_ALTITUDE = 0.85
LOOP_DT = 0.05                # 20 Hz (per simulazione in tempo reale RTF ~ 1.0)

# Parametri Boids
PERCEPTION_RADIUS = 10.0
SEPARATION_RADIUS = 0.90

W_SEPARATION = 1.00
W_COHESION = 0.55
W_ALIGNMENT = 0.25
W_BOUNDS = 0.60

MAX_SPEED_XY = 0.35           # m/s
MAX_SPEED_Z = 0.35            # m/s
KP_Z = 1.2

ARENA_BOUNDS = np.array([
    [-2.5, 2.5],
    [-2.5, 2.5]
])


# ============================================================================
# STATO CENTRALIZZATO
# ============================================================================
class CentralizedState:
    def __init__(self, uris):
        self.lock = threading.Lock()
        self.pos = {u: np.zeros(3) for u in uris}
        self.vel = {u: np.zeros(3) for u in uris}
        self.has_data = {u: False for u in uris}
        self.logconfs = {}

    def update(self, uri, x, y, z, vx, vy, vz):
        with self.lock:
            self.pos[uri] = np.array([x, y, z], dtype=float)
            self.vel[uri] = np.array([vx, vy, vz], dtype=float)
            self.has_data[uri] = True

    def all_received(self):
        with self.lock:
            return all(self.has_data.values())

    def get_all(self):
        with self.lock:
            return (
                {u: p.copy() for u, p in self.pos.items()},
                {u: v.copy() for u, v in self.vel.items()}
            )


state_hub = CentralizedState(URIS)


# ============================================================================
# LOGICA BOIDS -> VELOCITÀ MONDO (VX, VY, VZ)
# ============================================================================
def compute_boids_velocities():
    pos_dict, vel_dict = state_hub.get_all()
    vel_cmds = {}

    for u_i in URIS:
        p_i, v_i = pos_dict[u_i], vel_dict[u_i]
        vz = float(np.clip(KP_Z * (TARGET_ALTITUDE - p_i[2]), -MAX_SPEED_Z, MAX_SPEED_Z))

        v_sep = np.zeros(2)
        v_coh = np.zeros(2)
        v_ali = np.zeros(2)
        v_bnd = np.zeros(2)

        center_of_mass = np.zeros(2)
        avg_vel = np.zeros(2)
        neighbors = 0

        for u_j in URIS:
            if u_j == u_i:
                continue
            p_j, v_j = pos_dict[u_j], vel_dict[u_j]
            diff_xy = p_i[:2] - p_j[:2]
            dist_xy = np.linalg.norm(diff_xy)

            if 1e-4 < dist_xy < PERCEPTION_RADIUS:
                neighbors += 1
                center_of_mass += p_j[:2]
                avg_vel += v_j[:2]

                if dist_xy < SEPARATION_RADIUS:
                    strength = (SEPARATION_RADIUS - dist_xy) / SEPARATION_RADIUS
                    v_sep += (diff_xy / dist_xy) * strength

        if neighbors > 0:
            v_coh = (center_of_mass / neighbors) - p_i[:2]
            v_ali = (avg_vel / neighbors) - v_i[:2]

        for axis in range(2):
            low, high = ARENA_BOUNDS[axis]
            if p_i[axis] < low:
                v_bnd[axis] = low - p_i[axis]
            elif p_i[axis] > high:
                v_bnd[axis] = high - p_i[axis]

        v_xy = (
            W_SEPARATION * v_sep +
            W_COHESION * v_coh +
            W_ALIGNMENT * v_ali +
            W_BOUNDS * v_bnd
        )

        speed = np.linalg.norm(v_xy)
        if speed > MAX_SPEED_XY:
            v_xy = (v_xy / speed) * MAX_SPEED_XY

        vel_cmds[u_i] = (float(v_xy[0]), float(v_xy[1]), vz)

    return vel_cmds


# ============================================================================
# TELEMETRIA E RESET KALMAN
# ============================================================================
def reset_and_start_telemetry(scf):
    cf = scf.cf
    uri = cf.link_uri
    print(f"[{uri}] Connesso! Resetto l'estimatore di stato...", flush=True)
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(1.5)

    logconf = LogConfig(name='FlockState', period_in_ms=50)
    for v in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
        logconf.add_variable(f'stateEstimate.{v}', 'float')

    def cb(ts, data, lc):
        state_hub.update(
            uri,
            data['stateEstimate.x'], data['stateEstimate.y'], data['stateEstimate.z'],
            data['stateEstimate.vx'], data['stateEstimate.vy'], data['stateEstimate.vz']
        )

    cf.log.add_config(logconf)
    logconf.data_received_cb.add_callback(cb)
    logconf.start()
    state_hub.logconfs[uri] = logconf
    print(f"[{uri}] Telemetria attiva!", flush=True)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Inizializzazione driver CRTP...", flush=True)
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')

    print(f"Connessione a {len(URIS)} droni: {URIS}", flush=True)
    with Swarm(URIS, factory=factory) as swarm:
        swarm.parallel_safe(reset_and_start_telemetry)

        print("Attendo i primi pacchetti di telemetria...", flush=True)
        while not state_hub.all_received():
            time.sleep(0.2)

        time.sleep(1.0)
        pos_init, _ = state_hub.get_all()
        print("Posizioni iniziali rilevate:", flush=True)
        for idx, u in enumerate(URIS):
            print(f"  -> D{idx}: X={pos_init[u][0]:.2f}, Y={pos_init[u][1]:.2f}, Z={pos_init[u][2]:.2f}", flush=True)

        commanders = {uri: swarm._cfs[uri].cf.commander for uri in URIS}

        try:
            # 1. Sblocco motori
            print("1/4 Sblocco sicurezza motori (thrust = 0)...", flush=True)
            for _ in range(10):
                for cmd in commanders.values():
                    cmd.send_setpoint(0, 0, 0, 0)
                time.sleep(0.05)

            # 2. Decollo (4 secondi)
            print(f"2/4 Decollo verso Z = {TARGET_ALTITUDE:.2f}m...", flush=True)
            for step in range(80):
                pos_dict, _ = state_hub.get_all()
                for uri, cmd in commanders.items():
                    vz = float(np.clip(KP_Z * (TARGET_ALTITUDE - pos_dict[uri][2]), -MAX_SPEED_Z, MAX_SPEED_Z))
                    cmd.send_velocity_world_setpoint(0.0, 0.0, vz, 0.0)
                if step % 20 == 0:
                    z_str = " | ".join([f"D{i}: Z={pos_dict[u][2]:.2f}m" for i, u in enumerate(URIS)])
                    print(f"  Salita [{step}/80] --> {z_str}", flush=True)
                time.sleep(LOOP_DT)

            # 3. Flocking (15 secondi)
            print("3/4 Avvio Flocking Centralizzato...", flush=True)
            for step in range(300):
                vel_cmds = compute_boids_velocities()
                pos_dict, _ = state_hub.get_all()
                for uri, cmd in commanders.items():
                    vx, vy, vz = vel_cmds[uri]
                    cmd.send_velocity_world_setpoint(vx, vy, vz, 0.0)
                if step % 40 == 0:
                    pts = np.array([pos_dict[u][:2] for u in URIS])
                    r_medio = np.mean(np.linalg.norm(pts - np.mean(pts, axis=0), axis=1))
                    coords = " | ".join([
                        f"D{i}:({pos_dict[u][0]:.2f},{pos_dict[u][1]:.2f},{pos_dict[u][2]:.2f})"
                        for i, u in enumerate(URIS)
                    ])
                    print(f"  Flocking [{step}/300] R_medio={r_medio:.2f}m --> {coords}", flush=True)
                time.sleep(LOOP_DT)

            # 4. Atterraggio (3 secondi)
            print("4/4 Atterraggio morbido...", flush=True)
            for step in range(60):
                pos_dict, _ = state_hub.get_all()
                for uri, cmd in commanders.items():
                    cmd.send_velocity_world_setpoint(0.0, 0.0, -0.30, 0.0)
                if step % 20 == 0:
                    z_str = " | ".join([f"D{i}: Z={pos_dict[u][2]:.2f}m" for i, u in enumerate(URIS)])
                    print(f"  Discesa [{step}/60] --> {z_str}", flush=True)
                time.sleep(LOOP_DT)

        finally:
            print("Stop motori e chiusura telemetria...", flush=True)
            for cmd in commanders.values():
                cmd.send_stop_setpoint()
            for lc in state_hub.logconfs.values():
                lc.stop()

    print("Volo terminato!", flush=True)


if __name__ == '__main__':
    main()