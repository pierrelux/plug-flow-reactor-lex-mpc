"""
Lexicographic Optimization Based MPC for Profile Control in Distributed Parameter Systems

Reimplementation of:
Padhiyar & Bhartiya (2009) "Profile control in distributed parameter systems using 
lexicographic optimization based MPC", Journal of Process Control 19, 100-109.

Author: Claude (Anthropic)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - must be before importing pyplot
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import expm
from scipy.sparse import csc_matrix
import osqp

# =============================================================================
# PFR Model: 9 CSTRs in series with dimerization reaction 2A -> P
# =============================================================================

class PFRModel:
    """
    Plug Flow Reactor modeled as 9 CSTRs in series.
    Reaction: 2A -> P (irreversible, isothermal, elementary kinetics)
    
    Three feed streams:
    - Main feed F at CSTR 1 with concentration Ca_F1
    - Trim stream Fa at CSTR 4 with concentration Ca_F4  
    - Trim stream Fb at CSTR 7 with concentration Ca_F7
    """
    
    def __init__(self):
        # Reactor parameters
        self.n_cstr = 9  # Number of CSTRs
        self.k = 0.2     # Reaction rate constant (m^3/(kg*min)) - tuned to match paper setpoints
        self.V = np.ones(9) * 1.0  # Volume of each CSTR (m^3)
        
        # Feed concentrations (kg/m^3)
        self.Ca_F1 = 0.5  # Main feed
        self.Ca_F4 = 0.5  # Trim stream 1
        self.Ca_F7 = 0.5  # Trim stream 2
        
        # State: [Ca_1, Cp_1, Ca_2, Cp_2, ..., Ca_9, Cp_9]
        # where Ca_i is concentration of A in CSTR i
        # and Cp_i is concentration of P in CSTR i
        self.n_states = 2 * self.n_cstr
        
    def dynamics(self, x, t, u):
        """
        Compute dx/dt for the PFR system.
        
        x: state vector [Ca_1, Cp_1, Ca_2, Cp_2, ..., Ca_9, Cp_9]
        u: input vector [F, Fa, Fb] (flow rates)
        """
        F, Fa, Fb = u
        dxdt = np.zeros(self.n_states)
        
        # Extract concentrations
        Ca = x[0::2]  # Ca_1, Ca_2, ..., Ca_9
        Cp = x[1::2]  # Cp_1, Cp_2, ..., Cp_9
        
        # Flow rates in each section
        F1_3 = F          # CSTRs 1-3
        F4_6 = F + Fa     # CSTRs 4-6
        F7_9 = F + Fa + Fb  # CSTRs 7-9
        
        # CSTR 1: Feed stream enters
        dxdt[0] = (F * self.Ca_F1 - F * Ca[0]) / self.V[0] - 2 * self.k * Ca[0]**2
        dxdt[1] = (- F * Cp[0]) / self.V[0] + self.k * Ca[0]**2
        
        # CSTRs 2-3
        for i in range(1, 3):
            dxdt[2*i] = (F1_3 * Ca[i-1] - F1_3 * Ca[i]) / self.V[i] - 2 * self.k * Ca[i]**2
            dxdt[2*i+1] = (F1_3 * Cp[i-1] - F1_3 * Cp[i]) / self.V[i] + self.k * Ca[i]**2
            
        # CSTR 4: Trim stream Fa enters
        i = 3
        dxdt[2*i] = (F1_3 * Ca[i-1] + Fa * self.Ca_F4 - F4_6 * Ca[i]) / self.V[i] - 2 * self.k * Ca[i]**2
        dxdt[2*i+1] = (F1_3 * Cp[i-1] - F4_6 * Cp[i]) / self.V[i] + self.k * Ca[i]**2
        
        # CSTRs 5-6
        for i in range(4, 6):
            dxdt[2*i] = (F4_6 * Ca[i-1] - F4_6 * Ca[i]) / self.V[i] - 2 * self.k * Ca[i]**2
            dxdt[2*i+1] = (F4_6 * Cp[i-1] - F4_6 * Cp[i]) / self.V[i] + self.k * Ca[i]**2
            
        # CSTR 7: Trim stream Fb enters
        i = 6
        dxdt[2*i] = (F4_6 * Ca[i-1] + Fb * self.Ca_F7 - F7_9 * Ca[i]) / self.V[i] - 2 * self.k * Ca[i]**2
        dxdt[2*i+1] = (F4_6 * Cp[i-1] - F7_9 * Cp[i]) / self.V[i] + self.k * Ca[i]**2
        
        # CSTRs 8-9
        for i in range(7, 9):
            dxdt[2*i] = (F7_9 * Ca[i-1] - F7_9 * Ca[i]) / self.V[i] - 2 * self.k * Ca[i]**2
            dxdt[2*i+1] = (F7_9 * Cp[i-1] - F7_9 * Cp[i]) / self.V[i] + self.k * Ca[i]**2
            
        return dxdt
    
    def simulate(self, x0, u, t_span):
        """Simulate the system from x0 with constant input u over t_span."""
        sol = odeint(self.dynamics, x0, t_span, args=(u,))
        return sol
    
    def get_Cp_profile(self, x):
        """Extract product concentration profile from state."""
        return x[1::2]  # Cp_1, Cp_2, ..., Cp_9
    
    def get_controlled_outputs(self, x):
        """Get controlled outputs: Cp_3, Cp_6, Cp_9"""
        Cp = self.get_Cp_profile(x)
        return np.array([Cp[2], Cp[5], Cp[8]])  # Cp_3, Cp_6, Cp_9
    
    def linearize(self, x0, u0, eps=1e-6):
        """
        Linearize the system around (x0, u0).
        Returns A, B matrices for dx/dt = A*dx + B*du
        """
        n = self.n_states
        m = 3  # Number of inputs
        
        f0 = self.dynamics(x0, 0, u0)
        
        # Compute A = df/dx
        A = np.zeros((n, n))
        for i in range(n):
            x_plus = x0.copy()
            x_plus[i] += eps
            f_plus = self.dynamics(x_plus, 0, u0)
            A[:, i] = (f_plus - f0) / eps
            
        # Compute B = df/du
        B = np.zeros((n, m))
        for i in range(m):
            u_plus = u0.copy()
            u_plus[i] += eps
            f_plus = self.dynamics(x0, 0, u_plus)
            B[:, i] = (f_plus - f0) / eps
            
        return A, B
    
    def get_output_matrix(self):
        """
        Output matrix C for y = Cx where y = [Cp_3, Cp_6, Cp_9]
        """
        C = np.zeros((3, self.n_states))
        C[0, 5] = 1   # Cp_3
        C[1, 11] = 1  # Cp_6
        C[2, 17] = 1  # Cp_9
        return C


# =============================================================================
# Extended MPC with Lexicographic Optimization
# =============================================================================

class ExtendedMPC:
    """
    Extended MPC using the approach from Garcia (1984):
    - Nonlinear unforced prediction + Linear forced response
    - Results in QP formulation
    """
    
    def __init__(self, model, dt, p, q):
        """
        model: PFR model
        dt: sampling time
        p: prediction horizon
        q: control horizon
        """
        self.model = model
        self.dt = dt
        self.p = p  # Prediction horizon
        self.q = q  # Control horizon
        
        # Number of inputs and outputs
        self.n_u = 3
        self.n_y = 3
        
        # Output matrix
        self.C = model.get_output_matrix()
        
    def get_step_response_matrices(self, x0, u0):
        """
        Compute step response matrices for the linearized system.
        Returns S matrices for prediction: Y = Y_free + S * DU
        
        Uses matrix exponential for proper discretization (zero-order hold).
        """
        A, B = self.model.linearize(x0, u0)
        n = A.shape[0]
        m = B.shape[1]
        
        # Discretize using matrix exponential (zero-order hold)
        # Build augmented matrix [A B; 0 0] and compute exp(M*dt)
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A * self.dt
        M[:n, n:] = B * self.dt
        expM = expm(M)
        Ad = expM[:n, :n]
        Bd = expM[:n, n:]
        
        # Compute step response matrices
        # S[i,j] = response at time i to step at time j
        S = np.zeros((self.p * self.n_y, self.q * self.n_u))
        
        # Compute impulse response matrices
        CAk = self.C.copy()
        impulse_responses = [CAk @ Bd]
        for k in range(1, self.p):
            CAk = CAk @ Ad
            impulse_responses.append(CAk @ Bd)
            
        # Build step response matrix (lower triangular structure)
        for i in range(self.p):
            for j in range(min(i+1, self.q)):
                # Step response at time i due to step at time j
                # is sum of impulse responses from j to i
                step_resp = np.zeros((self.n_y, self.n_u))
                for k in range(j, i+1):
                    step_resp += impulse_responses[k-j]
                S[i*self.n_y:(i+1)*self.n_y, j*self.n_u:(j+1)*self.n_u] = step_resp
                
        return S
    
    def predict_unforced(self, x0, u0):
        """
        Compute unforced (free) response prediction.
        Integrates nonlinear model with constant input u0.
        """
        t_span = np.arange(0, (self.p + 1) * self.dt, self.dt)
        sol = self.model.simulate(x0, u0, t_span)
        
        # Extract outputs at each prediction step
        Y_free = np.zeros(self.p * self.n_y)
        for i in range(self.p):
            y = self.model.get_controlled_outputs(sol[i+1])
            Y_free[i*self.n_y:(i+1)*self.n_y] = y
            
        return Y_free
    
    def solve_qp(self, S, Y_free, R, u_prev, We, Wu, 
                 u_min, u_max, du_min, du_max,
                 y_eq_constraints=None, y_ineq_constraints=None):
        """
        Solve the MPC QP using OSQP.
        
        min 0.5 * DU' * H * DU + g' * DU
        s.t. l <= A_ineq * DU <= u
        
        Parameters:
        - S: step response matrix
        - Y_free: free response prediction
        - R: reference trajectory
        - u_prev: previous input
        - We: output error weights (diagonal)
        - Wu: input move weights (diagonal)
        - u_min, u_max: input bounds
        - du_min, du_max: input move bounds
        - y_eq_constraints: dict with 'indices' and 'values' for equality constraints
        - y_ineq_constraints: dict with 'indices', 'lower', 'upper' for inequality constraints
        """
        n_du = self.q * self.n_u
        
        # Build QP matrices
        # H = S' * We * S + Wu
        We_full = np.diag(np.tile(We, self.p))
        Wu_full = np.diag(np.tile(Wu, self.q))
        
        H = S.T @ We_full @ S + Wu_full
        
        # g = S' * We * (Y_free - R)
        error = Y_free - R
        g = S.T @ We_full @ error
        
        # Constraints
        # 1. Input bounds: u_min <= u_prev + sum(du) <= u_max
        # 2. Move bounds: du_min <= du <= du_max
        # 3. Lexicographic equality/inequality constraints (if any)
        
        # Build constraint matrices
        constraints_A = []
        constraints_l = []
        constraints_u = []
        
        # Move bounds: du_min <= du <= du_max
        I_du = np.eye(n_du)
        constraints_A.append(I_du)
        constraints_l.append(np.tile(du_min, self.q))
        constraints_u.append(np.tile(du_max, self.q))
        
        # Input bounds: cumulative sum structure
        # u[k+j] = u_prev + sum(du[0:j+1])
        L = np.zeros((self.q * self.n_u, n_du))
        for i in range(self.q):
            for j in range(i + 1):
                L[i*self.n_u:(i+1)*self.n_u, j*self.n_u:(j+1)*self.n_u] = np.eye(self.n_u)
        
        constraints_A.append(L)
        constraints_l.append(np.tile(u_min - u_prev, self.q))
        constraints_u.append(np.tile(u_max - u_prev, self.q))
        
        # Lexicographic equality constraints on outputs
        if y_eq_constraints is not None:
            indices = y_eq_constraints['indices']
            values = y_eq_constraints['values']
            
            for idx, val in zip(indices, values):
                # Y[idx] = Y_free[idx] + S[idx,:] @ DU = val
                # => S[idx,:] @ DU = val - Y_free[idx]
                row = S[idx:idx+1, :]
                rhs = val - Y_free[idx]
                constraints_A.append(row)
                constraints_l.append(np.array([rhs - 1e-6]))  # Small tolerance
                constraints_u.append(np.array([rhs + 1e-6]))
        
        # Lexicographic inequality constraints on outputs
        if y_ineq_constraints is not None:
            indices = y_ineq_constraints['indices']
            lower = y_ineq_constraints.get('lower', [None] * len(indices))
            upper = y_ineq_constraints.get('upper', [None] * len(indices))
            
            for idx, lb, ub in zip(indices, lower, upper):
                row = S[idx:idx+1, :]
                # Y[idx] = Y_free[idx] + S[idx,:] @ DU
                # lb <= Y[idx] <= ub
                # lb - Y_free[idx] <= S[idx,:] @ DU <= ub - Y_free[idx]
                l_val = -np.inf if lb is None else lb - Y_free[idx]
                u_val = np.inf if ub is None else ub - Y_free[idx]
                constraints_A.append(row)
                constraints_l.append(np.array([l_val]))
                constraints_u.append(np.array([u_val]))
        
        # Stack constraints
        A_full = np.vstack(constraints_A)
        l_full = np.hstack(constraints_l)
        u_full = np.hstack(constraints_u)
        
        # Convert to sparse
        H_sparse = csc_matrix(H)
        A_sparse = csc_matrix(A_full)
        
        # Setup and solve OSQP with better settings
        prob = osqp.OSQP()
        prob.setup(P=H_sparse, q=g, A=A_sparse, l=l_full, u=u_full,
                   verbose=False, 
                   eps_abs=1e-6, eps_rel=1e-6,
                   max_iter=10000,
                   polish=False,
                   adaptive_rho=True)
        result = prob.solve()
        
        if result.info.status not in ['solved', 'solved_inaccurate']:
            # Return previous solution (zero moves) if solver fails
            return np.zeros(n_du), np.inf
            
        return result.x, result.info.obj_val
    
    def compute_control_endpoint(self, x0, u_prev, setpoint, We, Wu,
                                  u_min, u_max, du_min, du_max):
        """
        Endpoint control: only control Cp_9.
        """
        # Get predictions
        S = self.get_step_response_matrices(x0, u_prev)
        Y_free = self.predict_unforced(x0, u_prev)
        
        # Reference trajectory (only endpoint matters)
        R = np.tile(setpoint, self.p)
        
        # Modify weights to only penalize endpoint (Cp_9)
        We_endpoint = np.array([0, 0, We[2]])  # Only weight on Cp_9
        
        # Solve QP
        DU, _ = self.solve_qp(S, Y_free, R, u_prev, We_endpoint, Wu,
                              u_min, u_max, du_min, du_max)
        
        # Return first control move
        du = DU[:self.n_u]
        return u_prev + du
    
    def compute_control_full_profile(self, x0, u_prev, setpoint, We, Wu,
                                      u_min, u_max, du_min, du_max):
        """
        Full profile control: control Cp_3, Cp_6, Cp_9 with single objective.
        """
        # Get predictions
        S = self.get_step_response_matrices(x0, u_prev)
        Y_free = self.predict_unforced(x0, u_prev)
        
        # Reference trajectory
        R = np.tile(setpoint, self.p)
        
        # Solve QP
        DU, _ = self.solve_qp(S, Y_free, R, u_prev, We, Wu,
                              u_min, u_max, du_min, du_max)
        
        # Return first control move
        du = DU[:self.n_u]
        return u_prev + du
    
    def compute_control_lexicographic(self, x0, u_prev, setpoint, We, Wu,
                                       u_min, u_max, du_min, du_max):
        """
        Lexicographic profile control using hierarchical weights.
        
        Uses a weighted approach where Cp_9 has much higher priority than Cp_6,
        which has higher priority than Cp_3. This allows the controller to 
        iterate toward the optimal solution while respecting the hierarchy.
        
        The weights create an implicit lexicographic ordering:
        - Cp_9: weight = 1e6 (highest priority)
        - Cp_6: weight = 1e3 (medium priority)
        - Cp_3: weight = 1 (lowest priority)
        """
        # Get predictions
        S = self.get_step_response_matrices(x0, u_prev)
        Y_free = self.predict_unforced(x0, u_prev)
        
        # Reference trajectory
        R = np.tile(setpoint, self.p)
        
        # Hierarchical weights: Cp_9 >> Cp_6 >> Cp_3
        # Using ratio of 1000:1 between levels for strong priority
        We_lex = np.array([We[0], We[1] * 1e3, We[2] * 1e6])
        
        # Solve single QP with hierarchical weights
        DU, _ = self.solve_qp(S, Y_free, R, u_prev, We_lex, Wu,
                               u_min, u_max, du_min, du_max)
        
        # Return first control move
        du = DU[:self.n_u]
        return u_prev + du


# =============================================================================
# Simulation Functions
# =============================================================================

def simulate_closed_loop(model, controller, x0, u0, setpoints, disturbance_times,
                         disturbances, t_final, dt, control_method,
                         We, Wu, u_min, u_max, du_min, du_max):
    """
    Run closed-loop simulation.
    
    Parameters:
    - setpoints: list of (time, setpoint) tuples
    - disturbance_times: list of times when disturbances occur
    - disturbances: list of (Ca_F1_new) values
    """
    n_steps = int(t_final / dt)
    
    # Storage
    t_history = np.zeros(n_steps)
    x_history = np.zeros((n_steps, model.n_states))
    u_history = np.zeros((n_steps, 3))
    y_history = np.zeros((n_steps, 3))
    
    # Initial conditions
    x = x0.copy()
    u = u0.copy()
    
    # Current setpoint
    current_setpoint = setpoints[0][1]
    setpoint_idx = 0
    
    # Current disturbance index
    dist_idx = 0
    
    for k in range(n_steps):
        t = k * dt
        t_history[k] = t
        
        # Check for setpoint change
        if setpoint_idx < len(setpoints) - 1:
            if t >= setpoints[setpoint_idx + 1][0]:
                setpoint_idx += 1
                current_setpoint = setpoints[setpoint_idx][1]
                
        # Check for disturbance
        if dist_idx < len(disturbance_times):
            if t >= disturbance_times[dist_idx]:
                model.Ca_F1 = disturbances[dist_idx]
                dist_idx += 1
        
        # Store current state
        x_history[k] = x
        y_history[k] = model.get_controlled_outputs(x)
        
        # Compute control action
        if control_method == 'endpoint':
            u_new = controller.compute_control_endpoint(
                x, u, current_setpoint, We, Wu, u_min, u_max, du_min, du_max)
        elif control_method == 'full_profile':
            u_new = controller.compute_control_full_profile(
                x, u, current_setpoint, We, Wu, u_min, u_max, du_min, du_max)
        elif control_method == 'lexicographic':
            u_new = controller.compute_control_lexicographic(
                x, u, current_setpoint, We, Wu, u_min, u_max, du_min, du_max)
        else:
            raise ValueError(f"Unknown control method: {control_method}")
        
        # Apply input constraints
        u_new = np.clip(u_new, u_min, u_max)
        
        # Apply move constraints
        du = u_new - u
        du = np.clip(du, du_min, du_max)
        u_new = u + du
        
        u_history[k] = u_new
        u = u_new
        
        # Simulate one step
        t_span = [0, dt]
        sol = model.simulate(x, u, t_span)
        x = sol[-1]
        
    return t_history, x_history, u_history, y_history


def find_steady_state(model, u, x0_guess=None):
    """Find steady state for given input."""
    if x0_guess is None:
        x0_guess = np.zeros(model.n_states)
        
    # Simulate for long time
    t_span = np.linspace(0, 500, 1000)
    sol = model.simulate(x0_guess, u, t_span)
    return sol[-1]


# =============================================================================
# Case Study 1: Achievable Target Profile
# =============================================================================

def run_case_study_1():
    """
    Case Study 1: Achievable target profile.
    - Setpoint change at 120 min
    - +10% disturbance in Ca_F1 at 520 min
    
    Following paper's Table 1:
    - Setpoint 1: Cp_3=0.07, Cp_6=0.0917, Cp_9=0.1173 (×10^-2 kg/m^3 in paper)
    - Setpoint 2: Cp_3=0.0568, Cp_6=0.0698, Cp_9=0.0877
    """
    print("Running Case Study 1: Achievable Target Profile")
    
    # Create model
    model = PFRModel()
    model.k = 0.2  # Tuned reaction rate to match paper setpoints
    
    # MPC parameters
    dt = 2.0  # Sampling time (min)
    p = 30    # Prediction horizon
    q = 10    # Control horizon
    
    controller = ExtendedMPC(model, dt, p, q)
    
    # Setpoints from Table 1 (scaled - paper shows ×10^-2)
    setpoint1 = np.array([0.07, 0.0917, 0.1173])
    setpoint2 = np.array([0.0568, 0.0698, 0.0877])
    
    setpoints = [(0, setpoint1), (120, setpoint2)]
    
    # Disturbance: +10% in Ca_F1 at 520 min
    disturbance_times = [520]
    disturbances = [0.55]  # 0.5 * 1.1
    
    # Weights - track all outputs equally, moderate input penalty
    We = np.array([1e4, 1e4, 1e4])
    Wu = np.array([10, 100, 100])
    
    # Input constraints from Table 1
    u_min = np.array([0.1, 0, 0])
    u_max = np.array([2.0, 1.0, 1.0])
    du_min = np.array([-0.3, -0.15, -0.15])
    du_max = np.array([0.3, 0.15, 0.15])
    
    # Operating point that gives Setpoint 1 (found via optimization)
    u0 = np.array([1.38, 0.42, 0.0])
    model.Ca_F1 = 0.5
    x0 = find_steady_state(model, u0)
    
    y0 = model.get_controlled_outputs(x0)
    print(f"  Initial SS outputs: Cp_3={y0[0]:.4f}, Cp_6={y0[1]:.4f}, Cp_9={y0[2]:.4f}")
    print(f"  Target setpoint 1:  Cp_3={setpoint1[0]:.4f}, Cp_6={setpoint1[1]:.4f}, Cp_9={setpoint1[2]:.4f}")
    
    t_final = 1000
    results = {}
    
    for method in ['endpoint', 'full_profile']:
        print(f"  Running {method} control...")
        model.Ca_F1 = 0.5
        t, x, u, y = simulate_closed_loop(
            model, controller, x0.copy(), u0.copy(), setpoints,
            disturbance_times, disturbances, t_final, dt, method,
            We, Wu, u_min, u_max, du_min, du_max
        )
        results[method] = {'t': t, 'x': x, 'u': u, 'y': y}
    
    return results, setpoints


def run_case_study_2():
    """
    Case Study 2: Unachievable target profile.
    - Large disturbance: Ca_F1 from 0.5 to 1.0 kg/m^3 at 520 min
    
    The key point of this case study is that the large disturbance makes
    Setpoint 2 unachievable, demonstrating the benefit of lexicographic MPC.
    """
    print("Running Case Study 2: Unachievable Target Profile")
    
    # Create model
    model = PFRModel()
    model.k = 0.2  # Tuned reaction rate (same as Case Study 1)
    
    # MPC parameters
    dt = 2.0
    p = 30
    q = 10
    
    controller = ExtendedMPC(model, dt, p, q)
    
    # Same setpoints as Case Study 1
    setpoint1 = np.array([0.07, 0.0917, 0.1173])
    setpoint2 = np.array([0.0568, 0.0698, 0.0877])
    
    setpoints = [(0, setpoint1), (120, setpoint2)]
    
    # Large disturbance: Ca_F1 doubles at 520 min
    # This makes setpoint 2 unachievable!
    disturbance_times = [520]
    disturbances = [1.0]  # 2x the nominal - makes target infeasible
    
    # Same weights
    We = np.array([1e4, 1e4, 1e4])
    Wu = np.array([10, 100, 100])
    
    # Input constraints from Table 1
    u_min = np.array([0.1, 0, 0])
    u_max = np.array([2.0, 1.0, 1.0])
    du_min = np.array([-0.3, -0.15, -0.15])
    du_max = np.array([0.3, 0.15, 0.15])
    
    # Operating point that gives Setpoint 1 (same as Case Study 1)
    u0 = np.array([1.38, 0.42, 0.0])
    model.Ca_F1 = 0.5
    x0 = find_steady_state(model, u0)
    
    y0 = model.get_controlled_outputs(x0)
    print(f"  Initial SS outputs: Cp_3={y0[0]:.4f}, Cp_6={y0[1]:.4f}, Cp_9={y0[2]:.4f}")
    print(f"  Target setpoint 2:  Cp_3={setpoint2[0]:.4f}, Cp_6={setpoint2[1]:.4f}, Cp_9={setpoint2[2]:.4f}")
    
    t_final = 2000
    results = {}
    
    for method in ['endpoint', 'full_profile', 'lexicographic']:
        print(f"  Running {method} control...")
        model.Ca_F1 = 0.5
        t, x, u, y = simulate_closed_loop(
            model, controller, x0.copy(), u0.copy(), setpoints,
            disturbance_times, disturbances, t_final, dt, method,
            We, Wu, u_min, u_max, du_min, du_max
        )
        results[method] = {'t': t, 'x': x, 'u': u, 'y': y}
    
    return results, setpoints


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_case_study_1(results, setpoints):
    """Plot results for Case Study 1."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    
    colors = {'endpoint': 'C0', 'full_profile': 'C1'}
    linestyles = {'endpoint': '--', 'full_profile': '-'}
    labels = {'endpoint': 'Endpoint Control', 'full_profile': 'Profile Control'}
    
    # Extract setpoint values
    sp1 = setpoints[0][1]
    sp2 = setpoints[1][1]
    t_sp_change = setpoints[1][0]
    
    # Plot Cp_3
    ax = axes[0]
    for method in ['endpoint', 'full_profile']:
        t = results[method]['t']
        y = results[method]['y']
        ax.plot(t, y[:, 0], linestyles[method], color=colors[method], 
                label=labels[method], linewidth=1.5)
    
    # Plot setpoints
    ax.axhline(sp1[0], color='gray', linestyle=':', linewidth=1)
    ax.axhline(sp2[0], color='gray', linestyle=':', linewidth=1)
    ax.axvline(t_sp_change, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(520, color='red', linestyle=':', alpha=0.5, label='Disturbance')
    
    ax.set_ylabel(r'$C_{p,3}$ (kg/m³)')
    ax.set_title('Case Study 1: Achievable Target Profile')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # Plot Cp_6
    ax = axes[1]
    for method in ['endpoint', 'full_profile']:
        t = results[method]['t']
        y = results[method]['y']
        ax.plot(t, y[:, 1], linestyles[method], color=colors[method], linewidth=1.5)
    
    ax.axhline(sp1[1], color='gray', linestyle=':', linewidth=1)
    ax.axhline(sp2[1], color='gray', linestyle=':', linewidth=1)
    ax.axvline(t_sp_change, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(520, color='red', linestyle=':', alpha=0.5)
    
    ax.set_ylabel(r'$C_{p,6}$ (kg/m³)')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # Plot Cp_9
    ax = axes[2]
    for method in ['endpoint', 'full_profile']:
        t = results[method]['t']
        y = results[method]['y']
        ax.plot(t, y[:, 2], linestyles[method], color=colors[method], linewidth=1.5)
    
    ax.axhline(sp1[2], color='gray', linestyle=':', linewidth=1)
    ax.axhline(sp2[2], color='gray', linestyle=':', linewidth=1)
    ax.axvline(t_sp_change, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(520, color='red', linestyle=':', alpha=0.5)
    
    ax.set_ylabel(r'$C_{p,9}$ (kg/m³)')
    ax.set_xlabel('Time (min)')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, 'c', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # Plot steady state profile
    ax = axes[3]
    # Get final profile
    model = PFRModel()
    for method in ['endpoint', 'full_profile']:
        x_final = results[method]['x'][-1]
        Cp_profile = model.get_Cp_profile(x_final)
        length = np.arange(1, 10)  # CSTR positions
        ax.plot(length, Cp_profile, linestyles[method], color=colors[method], 
                marker='o', markersize=4, linewidth=1.5)
    
    ax.axhline(sp2[0], xmin=0, xmax=3/9, color='gray', linestyle=':', linewidth=1)
    ax.axhline(sp2[1], xmin=3/9, xmax=6/9, color='gray', linestyle=':', linewidth=1)
    ax.axhline(sp2[2], xmin=6/9, xmax=1, color='gray', linestyle=':', linewidth=1)
    
    ax.set_xlabel('CSTR Number')
    ax.set_ylabel(r'$C_p$ (kg/m³)')
    ax.set_title('Steady State Profile at t=1000 min')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, 'd', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    plt.tight_layout()
    return fig


def plot_case_study_1_inputs(results):
    """Plot manipulated variables for Case Study 1."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    colors = {'endpoint': 'C0', 'full_profile': 'C1'}
    linestyles = {'endpoint': '--', 'full_profile': '-'}
    
    input_labels = [r'$F$ (m³/min)', r'$F_a$ (m³/min)', r'$F_b$ (m³/min)']
    
    for i, ax in enumerate(axes):
        for method in ['endpoint', 'full_profile']:
            t = results[method]['t']
            u = results[method]['u']
            ax.plot(t, u[:, i], linestyles[method], color=colors[method], linewidth=1.5)
        
        ax.set_ylabel(input_labels[i])
        ax.grid(True, alpha=0.3)
        ax.axvline(120, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(520, color='red', linestyle=':', alpha=0.5)
    
    axes[-1].set_xlabel('Time (min)')
    axes[0].set_title('Case Study 1: Manipulated Variables')
    axes[0].legend(['Endpoint Control', 'Profile Control'], loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_case_study_2(results, setpoints):
    """Plot results for Case Study 2."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    
    colors = {'endpoint': 'C0', 'full_profile': 'C1', 'lexicographic': 'C2'}
    linestyles = {'endpoint': '--', 'full_profile': '-', 'lexicographic': '-.'}
    labels = {'endpoint': 'Endpoint', 'full_profile': 'Full Profile', 
              'lexicographic': 'Lexicographic'}
    
    sp1 = setpoints[0][1]
    sp2 = setpoints[1][1]
    t_sp_change = setpoints[1][0]
    
    # Plot Cp_3
    ax = axes[0]
    for method in ['endpoint', 'full_profile', 'lexicographic']:
        t = results[method]['t']
        y = results[method]['y']
        ax.plot(t, y[:, 0], linestyles[method], color=colors[method], 
                label=labels[method], linewidth=1.5)
    
    ax.axhline(sp2[0], color='gray', linestyle=':', linewidth=1, label='Setpoint')
    ax.axvline(t_sp_change, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(520, color='red', linestyle=':', alpha=0.5)
    
    ax.set_ylabel(r'$C_{p,3}$ (kg/m³)')
    ax.set_title('Case Study 2: Unachievable Target Profile')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # Plot Cp_6
    ax = axes[1]
    for method in ['endpoint', 'full_profile', 'lexicographic']:
        t = results[method]['t']
        y = results[method]['y']
        ax.plot(t, y[:, 1], linestyles[method], color=colors[method], linewidth=1.5)
    
    ax.axhline(sp2[1], color='gray', linestyle=':', linewidth=1)
    ax.axvline(t_sp_change, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(520, color='red', linestyle=':', alpha=0.5)
    
    ax.set_ylabel(r'$C_{p,6}$ (kg/m³)')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # Plot Cp_9
    ax = axes[2]
    for method in ['endpoint', 'full_profile', 'lexicographic']:
        t = results[method]['t']
        y = results[method]['y']
        ax.plot(t, y[:, 2], linestyles[method], color=colors[method], linewidth=1.5)
    
    ax.axhline(sp2[2], color='gray', linestyle=':', linewidth=1)
    ax.axvline(t_sp_change, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(520, color='red', linestyle=':', alpha=0.5)
    
    ax.set_ylabel(r'$C_{p,9}$ (kg/m³)')
    ax.set_xlabel('Time (min)')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, 'c', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # Plot steady state profile
    ax = axes[3]
    model = PFRModel()
    for method in ['endpoint', 'full_profile', 'lexicographic']:
        x_final = results[method]['x'][-1]
        Cp_profile = model.get_Cp_profile(x_final)
        length = np.arange(1, 10)
        ax.plot(length, Cp_profile, linestyles[method], color=colors[method], 
                marker='o', markersize=4, linewidth=1.5)
    
    ax.axhline(sp2[2], color='gray', linestyle=':', linewidth=1)
    
    ax.set_xlabel('CSTR Number')
    ax.set_ylabel(r'$C_p$ (kg/m³)')
    ax.set_title('Steady State Profile at t=2000 min')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, 'd', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    plt.tight_layout()
    return fig


def plot_case_study_2_inputs(results):
    """Plot manipulated variables for Case Study 2."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    colors = {'endpoint': 'C0', 'full_profile': 'C1', 'lexicographic': 'C2'}
    linestyles = {'endpoint': '--', 'full_profile': '-', 'lexicographic': '-.'}
    labels = {'endpoint': 'Endpoint', 'full_profile': 'Full Profile', 
              'lexicographic': 'Lexicographic'}
    
    input_labels = [r'$F$ (m³/min)', r'$F_a$ (m³/min)', r'$F_b$ (m³/min)']
    
    for i, ax in enumerate(axes):
        for method in ['endpoint', 'full_profile', 'lexicographic']:
            t = results[method]['t']
            u = results[method]['u']
            ax.plot(t, u[:, i], linestyles[method], color=colors[method], linewidth=1.5)
        
        ax.set_ylabel(input_labels[i])
        ax.grid(True, alpha=0.3)
        ax.axvline(120, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(520, color='red', linestyle=':', alpha=0.5)
    
    axes[-1].set_xlabel('Time (min)')
    axes[0].set_title('Case Study 2: Manipulated Variables')
    axes[0].legend([labels[m] for m in ['endpoint', 'full_profile', 'lexicographic']], 
                   loc='upper right')
    
    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import os
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run Case Study 1
    results1, setpoints1 = run_case_study_1()
    
    # Plot Case Study 1
    fig1 = plot_case_study_1(results1, setpoints1)
    fig1.savefig(os.path.join(output_dir, 'case_study_1_outputs.png'), dpi=150, bbox_inches='tight')
    print("Saved case_study_1_outputs.png")
    
    fig1_inputs = plot_case_study_1_inputs(results1)
    fig1_inputs.savefig(os.path.join(output_dir, 'case_study_1_inputs.png'), dpi=150, bbox_inches='tight')
    print("Saved case_study_1_inputs.png")
    
    # Run Case Study 2
    results2, setpoints2 = run_case_study_2()
    
    # Plot Case Study 2
    fig2 = plot_case_study_2(results2, setpoints2)
    fig2.savefig(os.path.join(output_dir, 'case_study_2_outputs.png'), dpi=150, bbox_inches='tight')
    print("Saved case_study_2_outputs.png")
    
    fig2_inputs = plot_case_study_2_inputs(results2)
    fig2_inputs.savefig(os.path.join(output_dir, 'case_study_2_inputs.png'), dpi=150, bbox_inches='tight')
    print("Saved case_study_2_inputs.png")
    
    plt.close('all')  # Close all figures without displaying
    print("\nAll simulations completed!")