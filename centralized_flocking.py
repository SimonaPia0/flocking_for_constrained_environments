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
    'udp://127.0.0.1:19850',  # D0: (-1.4, -0.7)
    'udp://127.0.0.1:19851',  # D1: ( 0.0, -0.7)
    'udp://127.0.0.1:19852',  # D2: ( 1.4, -0.7)
    'udp://127.0.0.1:19853',  # D3: (-1.4,  0.7)
    'udp://127.0.0.1:19854',  # D4: ( 0.0,  0.7)
    'udp://127.0.0.1:19855',  # D5: ( 1.4,  0.7)
]

TARGET_ALTITUDE = 0.85
LOOP_DT = 0.05                # 20 Hz

# Parametri Flocking (Ferrante et al., 2012)
PERCEPTION_RADIUS = 10.0
DESIRED_DISTANCE = 1.0
PROXIMAL_RADIUS = 4.0

ALPHA = 2.0
EPSILON = 0.4
SIGMA = DESIRED_DISTANCE / (2.0 ** (1.0 / ALPHA))

W_PROXIMAL = 2.5
W_ALIGNMENT = 0.6
W_BOUNDS = 2.0
BOUND_MARGIN = 0.5

# Parametri MDMC (Motion Control: u lineare, omega angolare)
U_BIAS = 0.15                 # Velocità base di avanzamento (m/s)
U_MAX = 0.35                  # Velocità lineare massima (m/s)
K1_MDMC = 0.20                # Guadagno lineare su f_parallelo

OMEGA_MAX = 1.2               # Velocità angolare massima (rad/s ~ 69 deg/s)
K2_MDMC = 0.40                # Guadagno angolare su f_perpendicolare

# Controllo quota
MAX_SPEED_Z = 0.35            # m/s
KP_Z = 1.2

ARENA_BOUNDS = np.array([
    [-3.0, 3.0],
    [-3.0, 3.0]
])


# ============================================================================
# STATO CENTRALIZZATO (POSIZIONE, VELOCITÀ, YAW)
# ============================================================================
class CentralizedState:
    def __init__(self, uris):
        self.lock = threading.Lock()
        self.pos = {u: np.zeros(3) for u in uris}
        self.vel = {u: np.zeros(3) for u in uris}
        self.yaw = {u: 0.0 for u in uris}  # radianti
        self.has_pos = {u: False for u in uris}
        self.has_orient = {u: False for u in uris}
        self.logconfs = {}

    def update_state(self, uri, x, y, z, vx, vy, vz):
        with self.lock:
            self.pos[uri] = np.array([x, y, z], dtype=float)
            self.vel[uri] = np.array([vx, vy, vz], dtype=float)
            self.has_pos[uri] = True

    def update_yaw(self, uri, yaw_deg):
        with self.lock:
            self.yaw[uri] = np.radians(yaw_deg)
            self.has_orient[uri] = True

    def all_received(self):
        with self.lock:
            return all(self.has_pos.values()) and all(self.has_orient.values())

    def get_all(self):
        with self.lock:
            return (
                {u: p.copy() for u, p in self.pos.items()},
                {u: v.copy() for u, v in self.vel.items()},
                {u: y for u, y in self.yaw.items()},
            )


state_hub = CentralizedState(URIS)


# ============================================================================
# FLOCKING (f = p + a) + MDMC (u, omega)
# ============================================================================
def compute_mdmc_commands():
    pos_dict, _, yaw_dict = state_hub.get_all()
    cmds = {}

    for u_i in URIS:
        p_i = pos_dict[u_i]
        theta_i = yaw_dict[u_i]
        vz = float(np.clip(KP_Z * (TARGET_ALTITUDE - p_i[2]), -MAX_SPEED_Z, MAX_SPEED_Z))

        # 1. PROXIMAL CONTROL (p) - Media sui vicini prossimali (scalabile a N droni)
        p_vec = np.zeros(2)
        n_prox = 0
        for u_j in URIS:
            if u_j == u_i:
                continue
            p_j = pos_dict[u_j]
            diff_ij = p_j[:2] - p_i[:2]
            dist_xy = np.linalg.norm(diff_ij)

            if 1e-4 < dist_xy < PROXIMAL_RADIUS:
                direction = diff_ij / dist_xy
                ratio = SIGMA / dist_xy
                p_mag = -(4.0 * ALPHA * EPSILON / dist_xy) * (
                    2.0 * (ratio ** (2.0 * ALPHA)) - (ratio ** ALPHA)
                )
                p_mag = float(np.clip(p_mag, -2.0, 2.0))
                p_vec += p_mag * direction
                n_prox += 1

        if n_prox > 0:
            p_vec /= n_prox

        # 2. ALIGNMENT CONTROL (a) - Media vettoriale degli heading (Ferrante et al.)
        heading_sum = np.array([np.cos(theta_i), np.sin(theta_i)])
        for u_j in URIS:
            if u_j == u_i:
                continue
            dist_xy = np.linalg.norm(pos_dict[u_j][:2] - p_i[:2])
            if 1e-4 < dist_xy < PERCEPTION_RADIUS:
                theta_j = yaw_dict[u_j]
                heading_sum += np.array([np.cos(theta_j), np.sin(theta_j)])

        norm_h = np.linalg.norm(heading_sum)
        a_vec = (heading_sum / norm_h) if norm_h > 1e-6 else np.zeros(2)

        # 3. CONFINAMENTO ARENA (con margine di preavviso)
        v_bnd = np.zeros(2)
        for axis in range(2):
            low, high = ARENA_BOUNDS[axis]
            if p_i[axis] < low + BOUND_MARGIN:
                v_bnd[axis] = (low + BOUND_MARGIN) - p_i[axis]
            elif p_i[axis] > high - BOUND_MARGIN:
                v_bnd[axis] = (high - BOUND_MARGIN) - p_i[axis]

        # Vettore risultante f = p + a (+ confinamento arena)
        f = W_PROXIMAL * p_vec + W_ALIGNMENT * a_vec + W_BOUNDS * v_bnd

        # 4. MDMC: proiezione di f nel frame body del drone
        cos_t, sin_t = np.cos(theta_i), np.sin(theta_i)
        f_parallel = f[0] * cos_t + f[1] * sin_t
        f_perp = -f[0] * sin_t + f[1] * cos_t

        # Velocità lineare u (avanzamento) e velocità angolare omega (sterzata)
        u = float(np.clip(K1_MDMC * f_parallel + U_BIAS, 0.0, U_MAX))
        omega = float(np.clip(K2_MDMC * f_perp, -OMEGA_MAX, OMEGA_MAX))

        # Conversione in setpoint mondo per Crazyflie
        vx = u * cos_t
        vy = u * sin_t
        yawrate_deg = np.degrees(omega)

        cmds[u_i] = (float(vx), float(vy), vz, float(yawrate_deg))

    return cmds


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

    # 1. Posizione e velocità
    log_state = LogConfig(name='FlockState', period_in_ms=50)
    for v in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
        log_state.add_variable(f'stateEstimate.{v}', 'float')

    def cb_state(ts, data, lc):
        state_hub.update_state(
            uri,
            data['stateEstimate.x'], data['stateEstimate.y'], data['stateEstimate.z'],
            data['stateEstimate.vx'], data['stateEstimate.vy'], data['stateEstimate.vz']
        )

    cf.log.add_config(log_state)
    log_state.data_received_cb.add_callback(cb_state)
    log_state.start()
    state_hub.logconfs[uri + "_state"] = log_state

    # 2. Orientamento (solo yaw per MDMC)
    log_orient = LogConfig(name='FlockOrient', period_in_ms=50)
    log_orient.add_variable('stabilizer.yaw', 'float')

    def cb_orient(ts, data, lc):
        state_hub.update_yaw(uri, data['stabilizer.yaw'])

    cf.log.add_config(log_orient)
    log_orient.data_received_cb.add_callback(cb_orient)
    log_orient.start()
    state_hub.logconfs[uri + "_orient"] = log_orient

    print(f"[{uri}] Telemetria (stato + yaw) attiva!", flush=True)


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
        pos_init, _, _ = state_hub.get_all()
        print("Posizioni iniziali rilevate:", flush=True)
        for idx, u in enumerate(URIS):
            print(f"  -> D{idx}: X={pos_init[u][0]:.2f}, Y={pos_init[u][1]:.2f}, Z={pos_init[u][2]:.2f}", flush=True)

        commanders = {uri: swarm._cfs[uri].cf.commander for uri in URIS}

        try:
            # 1. Sblocco motori
            print("1/5 Sblocco sicurezza motori (thrust = 0)...", flush=True)
            for _ in range(10):
                for cmd in commanders.values():
                    cmd.send_setpoint(0, 0, 0, 0)
                time.sleep(0.05)

            # 2. Decollo (4 secondi)
            print(f"2/5 Decollo verso Z = {TARGET_ALTITUDE:.2f}m...", flush=True)
            for step in range(80):
                pos_dict, _, _ = state_hub.get_all()
                for uri, cmd in commanders.items():
                    vz = float(np.clip(KP_Z * (TARGET_ALTITUDE - pos_dict[uri][2]), -MAX_SPEED_Z, MAX_SPEED_Z))
                    cmd.send_velocity_world_setpoint(0.0, 0.0, vz, 0.0)
                if step % 20 == 0:
                    z_str = " | ".join([f"D{i}: Z={pos_dict[u][2]:.2f}m" for i, u in enumerate(URIS)])
                    print(f"  Salita [{step}/80] --> {z_str}", flush=True)
                time.sleep(LOOP_DT)

            # 3. Orientamento iniziale casuale (1.5 secondi)
            rng = np.random.default_rng()
            random_yawrates = {uri: float(rng.uniform(-90.0, 90.0)) for uri in URIS}
            print("3/5 Rotazione iniziale casuale per testare l'alignment...", flush=True)
            for _ in range(30):
                pos_dict, _, _ = state_hub.get_all()
                for uri, cmd in commanders.items():
                    vz = float(np.clip(KP_Z * (TARGET_ALTITUDE - pos_dict[uri][2]), -MAX_SPEED_Z, MAX_SPEED_Z))
                    cmd.send_velocity_world_setpoint(0.0, 0.0, vz, random_yawrates[uri])
                time.sleep(LOOP_DT)

            # 4. Flocking + MDMC (15 secondi)
            print("4/5 Avvio Flocking (f = p + a) + MDMC...", flush=True)
            for step in range(300):
                cmds = compute_mdmc_commands()
                _, _, yaw_dict = state_hub.get_all()
                for uri, cmd in commanders.items():
                    vx, vy, vz, yawrate_deg = cmds[uri]
                    cmd.send_velocity_world_setpoint(vx, vy, vz, yawrate_deg)
                if step % 40 == 0:
                    yaws_deg = " | ".join([f"D{i}:{np.degrees(yaw_dict[u]):.0f}°" for i, u in enumerate(URIS)])
                    print(f"  MDMC [{step}/300] Heading --> {yaws_deg}", flush=True)
                time.sleep(LOOP_DT)

            # 5. Atterraggio (3 secondi)
            print("5/5 Atterraggio morbido...", flush=True)
            for step in range(60):
                pos_dict, _, _ = state_hub.get_all()
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


if __name__ == '__main__':
    main()