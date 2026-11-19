import time
import threading
import warnings
import numpy as np
import cflib.crtp
from cflib.crazyflie.swarm import Swarm, CachedCfFactory
from cflib.crazyflie.log import LogConfig

warnings.simplefilter("ignore", DeprecationWarning)

# ============================================================================
# CONFIGURAZIONE PER I TUOI 3 DRONI
# ============================================================================
URIS = [
    'udp://127.0.0.1:19850',  # D0: ( 0.0,  0.0)
    'udp://127.0.0.1:19851',  # D1: ( 2.0, -1.5)
    'udp://127.0.0.1:19852',  # D2: (-1.5,  1.0)
]
NUM_DRONES = len(URIS)
RTF = 0.01

INITIAL_POS = {
    'udp://127.0.0.1:19850': np.array([ 0.0,  0.0, 0.0]),
    'udp://127.0.0.1:19851': np.array([ 2.0, -1.5, 0.0]),
    'udp://127.0.0.1:19852': np.array([-1.5,  1.0, 0.0]),
}

TARGET_ALTITUDE = 0.85        # Quota operativa moderata (più stabile a RTF basso)
MIN_SAFE_Z = 0.35             # Se un drone scende sotto questa quota, forza solo risalita

# Parametri Boids Centralizzati (Senza target dinamico)
PERCEPTION_RADIUS = 10.0
SEPARATION_RADIUS = 0.90      # Distanza minima desiderata tra droni (m)

W_SEPARATION = 1.00           # Repulsione dolce per evitare inclinazioni brusche
W_COHESION = 0.55             # Coesione verso il baricentro
W_ALIGNMENT = 0.25            # Allineamento velocità
W_BOUNDS = 0.60               # Confinamento arena

MAX_STEP_XY = 0.04            # Massimo spostamento del target XY per ciclo (evita inclinazioni eccessive)

ARENA_BOUNDS = np.array([
    [-2.5, 2.5],
    [-2.5, 2.5]
])


# ============================================================================
# STATO GLOBALE CENTRALIZZATO
# ============================================================================
class CentralizedState:
    def __init__(self, uris):
        self.lock = threading.Lock()
        self.pos = {u: INITIAL_POS[u].copy() for u in uris}
        self.vel = {u: np.zeros(3) for u in uris}
        self.target_xy = {u: INITIAL_POS[u][:2].copy() for u in uris}
        self.initialized_target = {u: False for u in uris}
        self.logconfs = {}

    def update(self, uri, x, y, z, vx, vy, vz):
        with self.lock:
            self.pos[uri] = np.array([x, y, z], dtype=float)
            self.vel[uri] = np.array([vx, vy, vz], dtype=float)
            if not self.initialized_target[uri] and z > 0.2:
                self.target_xy[uri] = np.array([x, y], dtype=float)
                self.initialized_target[uri] = True

    def get_all(self):
        with self.lock:
            return (
                {u: p.copy() for u, p in self.pos.items()},
                {u: v.copy() for u, v in self.vel.items()}
            )


state_hub = CentralizedState(URIS)


# ============================================================================
# CALCOLO CENTRALIZZATO BOIDS -> TARGET DI POSIZIONE (X_CMD, Y_CMD, Z_CMD)
# ============================================================================
def compute_boids_position_targets():
    pos_dict, vel_dict = state_hub.get_all()
    pos_cmds = {}

    for u_i in URIS:
        p_i = pos_dict[u_i]
        v_i = vel_dict[u_i]

        with state_hub.lock:
            if not state_hub.initialized_target[u_i]:
                state_hub.target_xy[u_i] = p_i[:2].copy()
                state_hub.initialized_target[u_i] = True
            curr_target_xy = state_hub.target_xy[u_i].copy()

        # Se il drone è sceso troppo in basso, blocca lo spostamento XY e riportalo in quota
        if p_i[2] < MIN_SAFE_Z:
            pos_cmds[u_i] = (float(p_i[0]), float(p_i[1]), float(TARGET_ALTITUDE))
            continue

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
            p_j = pos_dict[u_j]
            v_j = vel_dict[u_j]

            diff_xy = p_i[:2] - p_j[:2]
            dist_xy = np.linalg.norm(diff_xy)

            if 1e-4 < dist_xy < PERCEPTION_RADIUS:
                neighbors += 1
                center_of_mass += p_j[:2]
                avg_vel += v_j[:2]

                if dist_xy < SEPARATION_RADIUS:
                    rep_dir = diff_xy / dist_xy
                    strength = (SEPARATION_RADIUS - dist_xy) / SEPARATION_RADIUS
                    v_sep += rep_dir * (strength ** 1.2)

        if neighbors > 0:
            center_of_mass /= neighbors
            v_coh = center_of_mass - p_i[:2]

            avg_vel /= neighbors
            v_ali = avg_vel - v_i[:2]

        for axis in range(2):
            low, high = ARENA_BOUNDS[axis]
            if p_i[axis] < low:
                v_bnd[axis] = low - p_i[axis]
            elif p_i[axis] > high:
                v_bnd[axis] = high - p_i[axis]

        direction = (
            W_SEPARATION * v_sep +
            W_COHESION * v_coh +
            W_ALIGNMENT * v_ali +
            W_BOUNDS * v_bnd
        )

        # Avanza il setpoint di posizione a piccoli passi limitati
        norm = np.linalg.norm(direction)
        if norm > 1e-4:
            step_vec = (direction / norm) * min(MAX_STEP_XY, norm * 0.05)
        else:
            step_vec = np.zeros(2)

        next_target_xy = curr_target_xy + step_vec
        # Evita che il target si allontani troppo dalla posizione reale (anti-windup)
        err_target = next_target_xy - p_i[:2]
        if np.linalg.norm(err_target) > 0.45:
            next_target_xy = p_i[:2] + (err_target / np.linalg.norm(err_target)) * 0.45

        with state_hub.lock:
            state_hub.target_xy[u_i] = next_target_xy

        pos_cmds[u_i] = (float(next_target_xy[0]), float(next_target_xy[1]), float(TARGET_ALTITUDE))

    return pos_cmds


# ============================================================================
# SETUP TELEMETRIA
# ============================================================================
def reset_and_start_telemetry(scf):
    cf = scf.cf
    uri = cf.link_uri

    print(f"[{uri}] Resetto l'estimatore di stato...")
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(50.0)
    print(f"[{uri}] Estimatore pronto!")

    logconf = LogConfig(name='FlockState', period_in_ms=20)
    for v in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
        logconf.add_variable(f'stateEstimate.{v}', 'float')

    def callback(timestamp, data, logconf):
        state_hub.update(
            uri,
            data['stateEstimate.x'],
            data['stateEstimate.y'],
            data['stateEstimate.z'],
            data['stateEstimate.vx'],
            data['stateEstimate.vy'],
            data['stateEstimate.vz']
        )

    cf.log.add_config(logconf)
    logconf.data_received_cb.add_callback(callback)
    logconf.start()
    state_hub.logconfs[uri] = logconf


# ============================================================================
# MAIN
# ============================================================================
def main():
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')

    print(f"Connessione a {NUM_DRONES} droni simulati: {URIS}")
    with Swarm(URIS, factory=factory) as swarm:
        swarm.parallel_safe(reset_and_start_telemetry)
        time.sleep(2.5)

        pos_init, _ = state_hub.get_all()
        print("Posizioni fisiche iniziali:")
        for idx, u in enumerate(URIS):
            print(f"  -> D{idx} ({u}): X={pos_init[u][0]:.2f}, Y={pos_init[u][1]:.2f}, Z={pos_init[u][2]:.2f}")

        commanders = {uri: swarm._cfs[uri].cf.commander for uri in URIS}

        # 1. SBLOCCO MOTORI
        print("Sblocco sicurezza motori (thrust = 0)...")
        for _ in range(50):
            for cmd in commanders.values():
                cmd.send_setpoint(0.0, 0.0, 0.0, 0)
            time.sleep(0.02)

        # 2. DECOLLO VERSO LA POSIZIONE INIZIALE A TARGET_ALTITUDE
        print(f"Decollo verso Z = {TARGET_ALTITUDE:.2f}m...")
        takeoff_cycles = int(2000 * (0.01 / RTF))
        for step in range(takeoff_cycles):
            z_ref = min(TARGET_ALTITUDE, 0.15 + (TARGET_ALTITUDE - 0.15) * (step / (takeoff_cycles * 0.7)))
            for uri, cmd in commanders.items():
                x0, y0 = INITIAL_POS[uri][0], INITIAL_POS[uri][1]
                cmd.send_position_setpoint(float(x0), float(y0), float(z_ref), 0.0)

            if step % 100 == 0:
                pos_dict, _ = state_hub.get_all()
                z_str = " | ".join([f"D{idx}: Z={pos_dict[u][2]:.2f}m" for idx, u in enumerate(URIS)])
                print(f"Salita [ciclo {step}/{takeoff_cycles}] (z_ref={z_ref:.2f}m) --> {z_str}")

            time.sleep(0.05)

        # 3. FLOCKING CENTRALIZZATO CON POSITION SETPOINT INCREMENTALE
        flocking_cycles = int(4800 * (0.01 / RTF))
        print(f"Avvio Flocking Centralizzato ({flocking_cycles} cicli)...")

        for i in range(flocking_cycles):
            pos_cmds = compute_boids_position_targets()

            for uri, cmd in commanders.items():
                tx, ty, tz = pos_cmds[uri]
                cmd.send_position_setpoint(tx, ty, tz, 0.0)

            if i % 100 == 0:
                pos_dict, _ = state_hub.get_all()
                pts = np.array([pos_dict[u][:2] for u in URIS])
                centroid = np.mean(pts, axis=0)
                avg_radius = np.mean(np.linalg.norm(pts - centroid, axis=1))
                coords = " | ".join([
                    f"D{idx}:({pos_dict[u][0]:.2f}, {pos_dict[u][1]:.2f}, {pos_dict[u][2]:.2f})"
                    for idx, u in enumerate(URIS)
                ])
                print(f"Flocking [{i}/{flocking_cycles}] R_medio={avg_radius:.2f}m --> {coords}")

            time.sleep(0.05)

        # 4. ATTERRAGGIO MORBIDO
        print("Atterraggio morbido dello sciame...")
        land_cycles = int(1200 * (0.01 / RTF))
        for step in range(land_cycles):
            z_ref = max(0.05, TARGET_ALTITUDE * (1.0 - (step / land_cycles)))
            pos_dict, _ = state_hub.get_all()
            for uri, cmd in commanders.items():
                cx, cy = pos_dict[uri][0], pos_dict[uri][1]
                cmd.send_position_setpoint(float(cx), float(cy), float(z_ref), 0.0)
            time.sleep(0.05)

        for uri, cmd in commanders.items():
            cmd.send_stop_setpoint()
            state_hub.logconfs[uri].stop()

        print("Missione di Flocking completata con successo!")


if __name__ == '__main__':
    main()
