%% generate_fourtank_data.m
% =========================================================================
% Q-1 : Data generation for the Quadruple-Tank Process (Johansson, 2000)
%
% Solves the NON-LINEAR 4-tank equations with forward-difference (Euler)
% integration and produces N = 10,000 samples of:
%   x_true : true tank levels h1..h4         (4 x N)   -> "ground truth"
%   y_meas : noisy sensor readings kc*h1, kc*h2 (2 x N) -> what the KF/PF see
% An ode45 run of the noise-free model is also done to check the Euler
% integration is accurate.
% =========================================================================
clc; clear; close all;
rng(42);                        % fixed seed -> same "random" data every run

%% ------------------------------------------------------------------------
% 1. Physical parameters (Slide 4)
% -------------------------------------------------------------------------
A1 = 28;    A2 = 32;    A3 = 28;    A4 = 32;     % tank cross-sections [cm^2]
a1 = 0.071; a2 = 0.057; a3 = 0.071; a4 = 0.057;  % outlet hole areas  [cm^2]
g  = 981;                                        % gravity [cm/s^2]
gamma1 = 0.7;  gamma2 = 0.6;                     % valve split ratios
k1 = 3.33;     k2 = 3.35;                        % pump gains [cm^3/(V s)]
kc = 0.5;                                        % level sensor gain [V/cm]
v1 = 3;        v2 = 3;                           % pump voltages [V]
h0 = [12.4; 12.7; 1.8; 1.4];                     % initial levels [cm]

% Pack everything into one struct so the RHS function gets it in one go
p.A     = [A1 A2 A3 A4];
p.a     = [a1 a2 a3 a4];
p.g     = g;
p.gamma = [gamma1 gamma2];
p.k     = [k1 k2];
p.kc    = kc;

%% ------------------------------------------------------------------------
% 2. Simulation settings
% -------------------------------------------------------------------------
N  = 10000;          % number of samples asked for in Q-1
dt = 0.1;            % sample time [s]  -> total run = 1000 s
t  = (0:N-1)*dt;     % time vector (1 x N)

% Why dt = 0.1 s?  The tank time constants (Slide 3) come out to roughly
% T1~63 s, T2~90 s, T3~24 s, T4~30 s. Euler needs dt << smallest T for
% accuracy and stability; 0.1 s is ~240x smaller than T3. 1000 s of data
% covers about 10 time constants of the slowest tank.

%% ------------------------------------------------------------------------
% 3. Inputs (pump voltages)
% -------------------------------------------------------------------------
% Default: constant 3 V on both pumps (Slide 4).
% Set use_input_steps = true to add step changes. This excites the
% dynamics more, which makes the KF/PF comparison more interesting.
use_input_steps = false;

dv1 =  0.5*use_input_steps;   % v1: 3 -> 3.5 V at t = 300 s (0 if steps off)
dv2 = -0.5*use_input_steps;   % v2: 3 -> 2.5 V at t = 600 s (0 if steps off)
u_fun = @(tt) [v1 + dv1*(tt >= 300); v2 + dv2*(tt >= 600)];  % works for scalar or vector tt
u = u_fun(t);                 % 2 x N input history

%% ------------------------------------------------------------------------
% 4. Noise model
% -------------------------------------------------------------------------
% Process noise w : models unmodelled effects (flow ripple, pump
%                   variation, and so on). It enters the state equation.
% Measurement noise v : sensor noise on the level readings.
sigma_w = 0.05;      % process-noise intensity [cm/sqrt(s)]
sigma_v = 0.05;      % measurement-noise std   [V]  (= 0.1 cm in level)

Qd = (sigma_w^2 * dt) * eye(4);   % discrete process-noise covariance [cm^2]
R  = (sigma_v^2)      * eye(2);   % measurement-noise covariance      [V^2]
H  = [kc 0  0 0;                  % measurement matrix (Slide 3):
      0  kc 0 0];                 % only h1 and h2 are measured

%% ------------------------------------------------------------------------
% 5. Forward-difference (Euler-Maruyama) integration WITH noise
% -------------------------------------------------------------------------
%   x(k+1) = x(k) + dt * f(x(k), u(k)) + w(k),    w ~ N(0, Qd)
%   y(k)   = H * x(k) + v(k),                     v ~ N(0, R)
x_true = zeros(4, N);
y_meas = zeros(2, N);
Lq = chol(Qd, 'lower');           % Qd = Lq*Lq'  -> Lq*randn has cov Qd

x_true(:,1) = h0;
y_meas(:,1) = H*x_true(:,1) + sigma_v*randn(2,1);

for k = 1:N-1
    f = fourtank_rhs(x_true(:,k), u(:,k), p);    % dh/dt from Slide 2
    w = Lq * randn(4,1);                         % process noise sample
    x_true(:,k+1) = x_true(:,k) + dt*f + w;      % Euler step + noise
    x_true(:,k+1) = max(x_true(:,k+1), 0);       % a level can't go negative
    y_meas(:,k+1) = H*x_true(:,k+1) + sigma_v*randn(2,1);
end

%% ------------------------------------------------------------------------
% 6. Sanity check: noise-free Euler vs ode45
% -------------------------------------------------------------------------
% If Euler at this dt is accurate, both should agree closely. This
% confirms the integration error is small compared with the noise.
x_det = zeros(4, N);
x_det(:,1) = h0;
for k = 1:N-1
    x_det(:,k+1) = x_det(:,k) + dt*fourtank_rhs(x_det(:,k), u(:,k), p);
end

opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, 'MaxStep', 1);
[~, x_ode] = ode45(@(tt,h) fourtank_rhs(h, u_fun(tt), p), t, h0, opts);
x_ode = x_ode.';                                 % make it 4 x N

fprintf('Max |Euler - ode45| over the run (noise-free): %.2e cm\n', ...
        max(abs(x_det - x_ode), [], 'all'));

%% ------------------------------------------------------------------------
% 7. Analytical steady state (for intuition)
% -------------------------------------------------------------------------
% Set dh/dt = 0 in Slide 2. Tanks 3 and 4 have one inflow and one outflow:
%   a3*sqrt(2g h3) = (1-gamma2) k2 v2  ->  h3 = ((1-gamma2)k2 v2 / a3)^2 / 2g
% Tanks 1 and 2 also receive the outflow of the tank above them.
h3s = ((1-gamma2)*k2*v2/a3)^2 / (2*g);
h4s = ((1-gamma1)*k1*v1/a4)^2 / (2*g);
h1s = ((gamma1*k1*v1 + a3*sqrt(2*g*h3s))/a1)^2 / (2*g);
h2s = ((gamma2*k2*v2 + a4*sqrt(2*g*h4s))/a2)^2 / (2*g);
fprintf('Steady state (v1=v2=3 V): h = [%.2f %.2f %.2f %.2f] cm\n', ...
        h1s, h2s, h3s, h4s);

%% ------------------------------------------------------------------------
% 8. Plots
% -------------------------------------------------------------------------
figure('Name', 'Q-1: Generated data', 'Position', [100 100 1100 700]);
for i = 1:4
    subplot(3,2,i);
    plot(t, x_true(i,:), 'b'); hold on;
    plot(t, x_ode(i,:), 'r--', 'LineWidth', 1.2);
    grid on; xlabel('t [s]'); ylabel(sprintf('h_%d [cm]', i));
    title(sprintf('Tank %d level', i));
    if i == 1, legend('true (with process noise)', 'noise-free (ode45)'); end
end
subplot(3,2,5);
plot(t, y_meas(1,:), 'Color', [0.6 0.6 0.6]); hold on;
plot(t, kc*x_true(1,:), 'b'); grid on;
xlabel('t [s]'); ylabel('y_1 [V]'); title('Measurement y_1 = k_c h_1 + v_1');
legend('measured', 'true k_c h_1');
subplot(3,2,6);
plot(t, y_meas(2,:), 'Color', [0.6 0.6 0.6]); hold on;
plot(t, kc*x_true(2,:), 'b'); grid on;
xlabel('t [s]'); ylabel('y_2 [V]'); title('Measurement y_2 = k_c h_2 + v_2');
legend('measured', 'true k_c h_2');

%% ------------------------------------------------------------------------
% 9. Save for the KF (Q-2) and PF (Q-4) scripts
% -------------------------------------------------------------------------
% Save in the same folder as this script, not MATLAB's current folder
out_file = fullfile(fileparts(mfilename('fullpath')), 'fourtank_data.mat');
save(out_file, 't', 'dt', 'N', 'x_true', 'y_meas', 'u', ...
     'x_det', 'x_ode', 'h0', 'p', 'Qd', 'R', 'H', 'sigma_w', 'sigma_v');
fprintf('Saved %s (%d samples, dt = %.2f s)\n', out_file, N, dt);

%% ========================================================================
% Local function: non-linear 4-tank dynamics (Slide 2)
% =========================================================================
function dh = fourtank_rhs(h, u, p)
    h  = max(h, 0);                       % keep sqrt real
    A  = p.A;  a = p.a;  g = p.g;
    g1 = p.gamma(1);  g2 = p.gamma(2);
    k1 = p.k(1);      k2 = p.k(2);

    q_out = a(:) .* sqrt(2*g*h);          % Torricelli outflow of each tank [cm^3/s]

    dh = zeros(4,1);
    dh(1) = (-q_out(1) + q_out(3) + g1*k1*u(1))     / A(1);
    dh(2) = (-q_out(2) + q_out(4) + g2*k2*u(2))     / A(2);
    dh(3) = (-q_out(3) + (1-g2)*k2*u(2))            / A(3);
    dh(4) = (-q_out(4) + (1-g1)*k1*u(1))            / A(4);
end