%% Kalman_Filter.m
% =========================================================================
% Q-2 : Kalman Filter for the Quadruple-Tank Process
%       - estimates h3, h4 (NOT measured)
%       - gives filtered h1, h2 (measured, noisy)
%       - verifies all four against the true data from Q-1
% Q-3 : all the plots listed on Slide 5
%
% Needs fourtank_data.mat (from Data_Generation.m) in the same folder.
% =========================================================================
clc; clear; close all;

%% ========================================================================
% 0. FILTER TUNING 
% =========================================================================
% Q_filter, R_filter are what the FILTER believes the noise is.
% The data was generated with its own true Qd and R (saved in the .mat).
% Matched filter = filter uses the true values (scale = 1).
%
% Scale the true values:
%   1   -> filter knows the true noise level
%   >1  -> filter thinks that noise is BIGGER than it really is
%   <1  -> filter thinks that noise is SMALLER than it really is
Q_scale = [1 1 1 1];     % one factor per state     [h1 h2 h3 h4]
R_scale = [1 1];         % one factor per sensor    [y1 y2]

%% ------------------------------------------------------------------------
% 1. Load the Q-1 dataset
% -------------------------------------------------------------------------
here = fileparts(mfilename('fullpath'));
D = load(fullfile(here, 'fourtank_data.mat'));

t      = D.t;        dt = D.dt;     N = D.N;
x_true = D.x_true;   % 4 x N  true levels (used ONLY for verification)
y_meas = D.y_meas;   % 2 x N  noisy sensor readings  -> filter input
u      = D.u;        % 2 x N  pump voltages          -> filter input
p      = D.p;        % plant parameters
Qd     = D.Qd;       % process-noise covariance used to generate data
R      = D.R;        % measurement-noise covariance used to generate data
C      = D.H;        % measurement matrix [kc 0 0 0; 0 kc 0 0]

At = p.A;  ao = p.a;  g = p.g;            % tank areas, outlet areas, gravity
gam = p.gamma;  kp = p.k;  kc = p.kc;     % valve splits, pump gains, sensor gain

% ---- Build the filter's Q and R from the tuning section ----------------
% Scaling as S*Qd*S with S = diag(sqrt(scale)) multiplies each variance by
% its factor and keeps the matrix symmetric and positive-definite.
Sq = diag(sqrt(Q_scale));   Q_filter = Sq*Qd*Sq;
Sr = diag(sqrt(R_scale));   R_filter = Sr*R*Sr;

fprintf('True  Q diag (data) : [%s]\n', sprintf('%.2e ', diag(Qd)));
fprintf('Filter Q diag       : [%s]\n', sprintf('%.2e ', diag(Q_filter)));
fprintf('True  R diag (data) : [%s]\n', sprintf('%.2e ', diag(R)));
fprintf('Filter R diag       : [%s]\n\n', sprintf('%.2e ', diag(R_filter)));
tune_str = sprintf('Q/Q_{true} = [%s], R/R_{true} = [%s]', ...
    strtrim(sprintf('%.3g ', diag(Q_filter)./diag(Qd))), ...
    strtrim(sprintf('%.3g ', diag(R_filter)./diag(R))));

%% ------------------------------------------------------------------------
% 2. Operating point (where we linearize)
% -------------------------------------------------------------------------
% A linear model is only valid near an EQUILIBRIUM (where dh/dt = 0).
% Slide 4's h0 is the paper's rounded value; it is close to, but not
% exactly, the equilibrium for v1 = v2 = 3 V with these parameters.
% So we compute the exact equilibrium for the nominal input u0 and
% linearize there (same formulas as section 7 of Data_Generation.m).
u0  = u(:,1);                                         % nominal pump voltages
h3s = ((1-gam(2))*kp(2)*u0(2)/ao(3))^2 / (2*g);
h4s = ((1-gam(1))*kp(1)*u0(1)/ao(4))^2 / (2*g);
h1s = ((gam(1)*kp(1)*u0(1) + ao(3)*sqrt(2*g*h3s))/ao(1))^2 / (2*g);
h2s = ((gam(2)*kp(2)*u0(2) + ao(4)*sqrt(2*g*h4s))/ao(2))^2 / (2*g);
x0  = [h1s; h2s; h3s; h4s];                           % operating point [cm]

%% ------------------------------------------------------------------------
% 3. Linearized continuous-time model (Slide 3)
% -------------------------------------------------------------------------
% d(dx)/dt = Ac*dx + Bc*du,   dx = h - x0,  du = u - u0
% Ti = (Ai/ai)*sqrt(2*hi0/g) : how fast tank i drains back to equilibrium
T = (At ./ ao) .* sqrt(2*x0.'/g);                     % 1 x 4 time constants [s]

Ac = [-1/T(1)   0        At(3)/(At(1)*T(3))   0;
       0       -1/T(2)   0                    At(4)/(At(2)*T(4));
       0        0       -1/T(3)               0;
       0        0        0                   -1/T(4)];

Bc = [ gam(1)*kp(1)/At(1)        0;
       0                         gam(2)*kp(2)/At(2);
       0                         (1-gam(2))*kp(2)/At(3);
       (1-gam(1))*kp(1)/At(4)    0];

%% ------------------------------------------------------------------------
% 4. Discretize (forward difference, same as the data generation)
% -------------------------------------------------------------------------
% dx(k+1) = dx(k) + dt*(Ac*dx(k) + Bc*du(k))  =>  F = I + Ac*dt, G = Bc*dt
% (expm(Ac*dt) gives practically the same F at dt = 0.1 s.)
F = eye(4) + Ac*dt;
G = Bc*dt;

% Observability check: can h1, h2 measurements reveal all 4 states?
% O = [C; CF; CF^2; CF^3] must have rank 4. It does because tank 3 drains
% into tank 1 and tank 4 into tank 2 (the F13, F24 coupling terms).
O = [C; C*F; C*F^2; C*F^3];
fprintf('Operating point x0 = [%.3f %.3f %.3f %.3f] cm\n', x0);
fprintf('Time constants  T  = [%.1f %.1f %.1f %.1f] s\n', T);
fprintf('Observability matrix rank = %d (need 4)\n\n', rank(O));

%% ------------------------------------------------------------------------
% 5. Filter initialization
% -------------------------------------------------------------------------
% h1, h2 : start from the first sensor reading (divide out kc)
% h3, h4 : no sensor, so start from a deliberately WRONG guess,
%          to show the filter pulling the estimate to the truth
x_hat0 = [y_meas(1,1)/kc;
          y_meas(2,1)/kc;
          3.0;                 % true h3(0) = 1.8 cm
          0.5];                % true h4(0) = 1.4 cm

% P0 = how unsure we are about x_hat0 (variances, cm^2)
%   h1, h2 : about sensor accuracy -> (sigma_v/kc)^2 = 0.1^2
%   h3, h4 : could be off by ~1 cm  -> 1^2
P0 = diag([R_filter(1,1)/kc^2, R_filter(2,2)/kc^2, 1, 1]);

%% ------------------------------------------------------------------------
% 6. Storage for every quantity Slide 5 asks to plot
% -------------------------------------------------------------------------
x_prior = zeros(4, N);      % x(k-)  prior estimate
x_post  = zeros(4, N);      % x(k+)  posterior estimate
P_prior = zeros(4, 4, N);   % P(k-)
P_post  = zeros(4, 4, N);   % P(k+)
K_gain  = zeros(4, 2, N);   % K(k)
r_prior = zeros(2, N);      % r(k-) = y - C*x(k-)   (innovation)
r_post  = zeros(2, N);      % r(k+) = y - C*x(k+)
S_innov = zeros(2, 2, N);   % innovation covariance (for residual bounds)

I4 = eye(4);

%% ------------------------------------------------------------------------
% 7. The Kalman filter loop
% -------------------------------------------------------------------------
for k = 1:N
    % ---------- PREDICT (time update) ------------------------------------
    % Push the previous best estimate through the model.
    % Uncertainty grows: F*P*F' carries old uncertainty forward,
    % + Q_filter adds the new process noise of this step.
    if k == 1
        xm = x_hat0;
        Pm = P0;
    else
        xm = x0 + F*(x_post(:,k-1) - x0) + G*(u(:,k-1) - u0);
        Pm = F*P_post(:,:,k-1)*F.' + Q_filter;
    end

    % ---------- UPDATE (measurement update) ------------------------------
    rm = y_meas(:,k) - C*xm;          % innovation: measured - predicted
    S  = C*Pm*C.' + R_filter;         % predicted spread of the innovation
    K  = Pm*C.' / S;                  % Kalman gain (4 x 2)
    xp = xm + K*rm;                   % correct the prediction
    % Joseph form: same result as (I-KC)*Pm in exact math, but stays
    % symmetric and positive-definite despite rounding errors
    Pp = (I4 - K*C)*Pm*(I4 - K*C).' + K*R_filter*K.';
    rp = y_meas(:,k) - C*xp;          % residual after the correction

    % ---------- store ----------------------------------------------------
    x_prior(:,k)   = xm;   x_post(:,k)   = xp;
    P_prior(:,:,k) = Pm;   P_post(:,:,k) = Pp;
    K_gain(:,:,k)  = K;    S_innov(:,:,k) = S;
    r_prior(:,k)   = rm;   r_post(:,k)   = rp;
end

%% ------------------------------------------------------------------------
% 8. Verification against the Q-1 dataset
% -------------------------------------------------------------------------
err_post  = x_true - x_post;                        % estimation error
rmse_post = sqrt(mean(err_post.^2, 2));
rmse_meas = sqrt(mean((y_meas/kc - x_true(1:2,:)).^2, 2));   % raw sensor
sig_post  = sqrt([squeeze(P_post(1,1,:)), squeeze(P_post(2,2,:)), ...
                  squeeze(P_post(3,3,:)), squeeze(P_post(4,4,:))]).';
inside2s  = 100*mean(abs(err_post) <= 2*sig_post, 2);   % should be ~95%

fprintf('------------- Verification (whole run) -------------\n');
fprintf('State | RMSE raw sensor | RMSE KF | %% inside +/-2 sigma\n');
for i = 1:4
    if i <= 2, raw = sprintf('%8.3f cm', rmse_meas(i)); else, raw = '   (none)  '; end
    fprintf('  h%d  |   %s   | %.3f cm |    %5.1f %%\n', i, raw, rmse_post(i), inside2s(i));
end
fprintf('\nSteady-state Kalman gain K (last step):\n'); disp(K_gain(:,:,end));
fprintf('Steady-state posterior std devs [cm]: [%.3f %.3f %.3f %.3f]\n', sig_post(:,end));

%% ------------------------------------------------------------------------
% 9. Plots (Slide 5)
% -------------------------------------------------------------------------
% 9.1 Prior estimate x(k-)
plot_states(t, x_true, x_prior, y_meas/kc, 'Prior estimate x(k^-)', 'prior est. x(k^-)');

% 9.2 Posterior estimate x(k+)
plot_states(t, x_true, x_post, y_meas/kc, 'Posterior estimate x(k^+)', 'posterior est. x(k^+)');

% 9.3 / 9.4 Prior and posterior covariance
plot_cov(t, P_prior, 'Prior covariance P(k^-)');
plot_cov(t, P_post,  'Posterior covariance P(k^+)');

% 9.5 Kalman gain K(k): 4 states x 2 measurements = 8 entries
figure('Name', 'Kalman gain K(k)', 'Position', [80 60 1000 750]);
for i = 1:4
    for j = 1:2
        subplot(4, 2, 2*(i-1) + j);
        plot(t, squeeze(K_gain(i,j,:)), 'LineWidth', 1.2); grid on;
        ylabel(sprintf('K_{%d%d}', i, j));
        title(sprintf('effect of y_%d residual on h_%d', j, i));
        if i == 4, xlabel('t [s]'); end
    end
end
sgtitle('Kalman gain K(k)');

% 9.6 Prior residuals (innovations) with +/-2 sqrt(S) bounds
figure('Name', 'Prior residuals r(k-)', 'Position', [100 80 1000 550]);
for j = 1:2
    subplot(2,1,j);
    bnd = 2*sqrt(squeeze(S_innov(j,j,:))).';
    plot(t, r_prior(j,:), 'Color', [0.3 0.5 0.9]); hold on;
    plot(t,  bnd, 'r--', t, -bnd, 'r--', 'LineWidth', 1.2); grid on;
    ylabel(sprintf('r_%d(k^-) [V]', j)); xlabel('t [s]');
    title(sprintf('Innovation y_%d - C x(k^-)  (%.1f%% inside \\pm2\\surdS)', ...
          j, 100*mean(abs(r_prior(j,:)) <= bnd)));
    legend('r(k^-)', '\pm 2\surdS');
end
sgtitle('Prior residuals / innovations r(k^-)');

% 9.7 Posterior residuals
figure('Name', 'Posterior residuals r(k+)', 'Position', [120 100 1000 550]);
for j = 1:2
    subplot(2,1,j);
    plot(t, r_post(j,:), 'Color', [0.2 0.6 0.3]); hold on;
    plot(t, r_prior(j,:), 'Color', [0.7 0.7 0.7 0.5]);
    grid on; ylabel(sprintf('r_%d [V]', j)); xlabel('t [s]');
    title(sprintf('Posterior residual y_%d - C x(k^+)', j));
    legend('r(k^+)', 'r(k^-) for comparison');
end
sgtitle('Posterior residuals r(k^+)');

% 9.8 Verification: estimation error with +/-2 sigma bounds from P(k+)
figure('Name', 'Estimation error', 'Position', [140 120 1100 650]);
for i = 1:4
    subplot(2,2,i);
    plot(t, err_post(i,:), 'b'); hold on;
    plot(t,  2*sig_post(i,:), 'r--', t, -2*sig_post(i,:), 'r--', 'LineWidth', 1.2);
    grid on; xlabel('t [s]'); ylabel(sprintf('error in h_%d [cm]', i));
    title(sprintf('h_%d error: RMSE %.3f cm, %.1f%% inside \\pm2\\sigma', ...
          i, rmse_post(i), inside2s(i)));
    if i == 1, legend('true - estimate', '\pm 2\sigma from P(k^+)'); end
end
sgtitle({'Verification: error vs filter''s own uncertainty', tune_str});

%% ------------------------------------------------------------------------
% 10. Save results (for comparison with the Particle Filter later)
% -------------------------------------------------------------------------
save(fullfile(here, 'kf_results.mat'), 'x_prior', 'x_post', 'P_prior', ...
     'P_post', 'K_gain', 'r_prior', 'r_post', 'S_innov', 'rmse_post', ...
     'F', 'G', 'Ac', 'Bc', 'x0', 'u0', 'x_hat0', 'P0', 'Q_filter', 'R_filter');
fprintf('\nSaved kf_results.mat\n');

%% ========================================================================
% Local plotting functions
% =========================================================================
function plot_states(t, x_true, x_hat, h_meas, fig_title, est_label)
    % 4 subplots: true level, estimate, and (for h1, h2) the raw sensor
    figure('Name', fig_title, 'Position', [60 40 1100 650]);
    for i = 1:4
        subplot(2,2,i); hold on;
        if i <= 2
            plot(t, h_meas(i,:), 'Color', [0.75 0.75 0.75]);   % raw sensor
        end
        plot(t, x_true(i,:), 'b', 'LineWidth', 1.2);
        plot(t, x_hat(i,:),  'r', 'LineWidth', 1.0);
        grid on; xlabel('t [s]'); ylabel(sprintf('h_%d [cm]', i));
        title(sprintf('Tank %d', i));
        if i <= 2
            legend('measured y/k_c', 'true', est_label);
        else
            legend('true', est_label);
        end
    end
    sgtitle(fig_title);
end

function plot_cov(t, P, fig_title)
    % variances (diagonal) on a log scale + the two key cross terms
    figure('Name', fig_title, 'Position', [70 50 1100 650]);
    for i = 1:4
        subplot(2,3,i);
        semilogy(t, squeeze(P(i,i,:)), 'LineWidth', 1.2); grid on;
        xlabel('t [s]'); ylabel(sprintf('P_{%d%d} [cm^2]', i, i));
        title(sprintf('variance of h_%d error', i));
    end
    subplot(2,3,5);
    plot(t, squeeze(P(1,3,:)), 'LineWidth', 1.2); grid on;
    xlabel('t [s]'); ylabel('P_{13} [cm^2]'); title('h_1-h_3 error correlation');
    subplot(2,3,6);
    plot(t, squeeze(P(2,4,:)), 'LineWidth', 1.2); grid on;
    xlabel('t [s]'); ylabel('P_{24} [cm^2]'); title('h_2-h_4 error correlation');
    sgtitle(fig_title);
end
