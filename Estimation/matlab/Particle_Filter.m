%% Particle_Filter.m
% =========================================================================
% Q-4 : Particle Filter (bootstrap / SIR) for the Quadruple-Tank Process
%       - estimates h3, h4 (NOT measured)
%       - gives filtered h1, h2 (measured, noisy)
%       - uses the full NON-LINEAR model (no linearization, unlike the KF)
%       - compares against the true data from Q-1 and against the KF
%
% Needs  : fourtank_data.mat (from Data_Generation.m) in the same folder
% Optional: kf_results.mat   (from Kalman_Filter.m) for the KF comparison
% =========================================================================
clc; clear; close all;

%% ========================================================================
% 0. FILTER TUNING  <-- edit these
% =========================================================================
Np = 2000;               % number of particles (try 100, 500, 1000, 3000)
                         % fewer -> faster but h3/h4 spread is underestimated

% Noise the FILTER assumes, as multiples of the true noise in the data
% (same meaning as in Kalman_Filter.m: 1 = matched)
Q_scale = [1 1 1 1];     % per state  [h1 h2 h3 h4]
R_scale = [1 1];         % per sensor [y1 y2]

% Resampling: resample only when the effective number of particles
% N_eff = 1/sum(w.^2) drops below this fraction of Np.
%   1   -> resample every step
%   0.5 -> common choice
resample_frac = 0.5;

% Initial guess: SAME as Kalman_Filter.m so the comparison is fair
x_hat0_h34 = [3.0; 0.5]; % deliberately wrong (true h3 = 1.8, h4 = 1.4)
P0_h34     = [1; 1];     % variance of that guess [cm^2]

rng(1);                  % fixed seed -> same particles every run

%% ------------------------------------------------------------------------
% 1. Load the Q-1 dataset
% -------------------------------------------------------------------------
here = fileparts(mfilename('fullpath'));
D = load(fullfile(here, 'fourtank_data.mat'));

t      = D.t;        dt = D.dt;     N = D.N;
x_true = D.x_true;   % 4 x N true levels (ONLY for verification)
y_meas = D.y_meas;   % 2 x N noisy readings -> filter input
u      = D.u;        % 2 x N pump voltages  -> filter input
p      = D.p;        % plant parameters (same struct the data used)
Qd     = D.Qd;       % true process-noise covariance  [cm^2 per step]
R      = D.R;        % true measurement-noise covariance [V^2]
C      = D.H;        % [kc 0 0 0; 0 kc 0 0]
kc     = p.kc;

% ---- the filter's Q and R ----------------------------------------------
Sq = diag(sqrt(Q_scale));  Q_filter = Sq*Qd*Sq;   % scale each true variance
Sr = diag(sqrt(R_scale));  R_filter = Sr*R*Sr;
Lq   = chol(Q_filter, 'lower');   % Lq*randn(4,Np) has covariance Q_filter
Rinv = inv(R_filter);             % used in every likelihood (2x2, cheap)

fprintf('Particles: %d,  resample when N_eff < %.0f\n', Np, resample_frac*Np);
fprintf('Filter Q diag : [%s]\n', sprintf('%.2e ', diag(Q_filter)));
fprintf('Filter R diag : [%s]\n\n', sprintf('%.2e ', diag(R_filter)));

%% ------------------------------------------------------------------------
% 2. Initial particle cloud
% -------------------------------------------------------------------------
% Same x_hat0 and P0 as the KF: h1, h2 from the first reading,
% h3, h4 from the deliberately wrong guess above.
x_hat0 = [y_meas(1,1)/kc; y_meas(2,1)/kc; x_hat0_h34];
P0     = diag([R_filter(1,1)/kc^2, R_filter(2,2)/kc^2, P0_h34(:).']);

X = x_hat0 + chol(P0, 'lower')*randn(4, Np);   % 4 x Np, one column = one particle
X = max(X, 0);                                 % a level can't be negative
w = ones(1, Np)/Np;                            % equal weights to start

%% ------------------------------------------------------------------------
% 3. Storage
% -------------------------------------------------------------------------
x_prior = zeros(4, N);     P_prior = zeros(4, 4, N);   % before the measurement
x_post  = zeros(4, N);     P_post  = zeros(4, 4, N);   % after the measurement
r_prior = zeros(2, N);     r_post  = zeros(2, N);      % residuals
Neff    = zeros(1, N);     did_resample = false(1, N);

% snapshots of the h3 particle cloud, to see it converge
snap_t   = [0 2 5 10 30 100];                          % [s]
snap_k   = round(snap_t/dt) + 1;
snap_X3  = zeros(numel(snap_k), Np);
snap_w   = zeros(numel(snap_k), Np);

%% ------------------------------------------------------------------------
% 4. The particle filter loop
% -------------------------------------------------------------------------
tic;
for k = 1:N
    % ---------- PREDICT: move every particle through the non-linear model
    % Each particle gets its OWN process-noise sample, so the cloud spreads
    % out -> this is the PF version of "P grows by Q".
    if k > 1
        X = X + dt*fourtank_rhs_vec(X, u(:,k-1), p) + Lq*randn(4, Np);
        X = max(X, 0);
    end
    [x_prior(:,k), P_prior(:,:,k)] = weighted_stats(X, w);

    % ---------- WEIGHT: how well does each particle explain y(k)? --------
    % likelihood p(y | x_i) = exp(-0.5 * e' * R^-1 * e),  e = y - C*x_i
    % Done in LOGS and shifted by the max, otherwise exp() underflows to 0
    % for every particle and the weights become 0/0.
    e     = y_meas(:,k) - C*X;                      % 2 x Np residuals
    loglk = -0.5*sum(e .* (Rinv*e), 1);             % 1 x Np
    logw  = log(w) + loglk;
    logw  = logw - max(logw);
    w     = exp(logw);   w = w/sum(w);

    % ---------- ESTIMATE: weighted mean and covariance of the cloud ------
    [x_post(:,k), P_post(:,:,k)] = weighted_stats(X, w);
    r_prior(:,k) = y_meas(:,k) - C*x_prior(:,k);
    r_post(:,k)  = y_meas(:,k) - C*x_post(:,k);

    % snapshot (taken before resampling, with its weights)
    s = find(snap_k == k, 1);
    if ~isempty(s), snap_X3(s,:) = X(3,:); snap_w(s,:) = w; end

    % ---------- RESAMPLE when the weights have collapsed -----------------
    % N_eff = Np when all weights are equal, 1 when one particle has all.
    Neff(k) = 1/sum(w.^2);
    if Neff(k) < resample_frac*Np
        idx = systematic_resample(w);
        X   = X(:, idx);                 % good particles copied, bad ones gone
        w   = ones(1, Np)/Np;
        did_resample(k) = true;
    end
end
pf_time = toc;
fprintf('PF run time: %.1f s (%d steps x %d particles)\n', pf_time, N, Np);
fprintf('Resampled on %.1f%% of steps, mean N_eff = %.0f\n\n', ...
        100*mean(did_resample), mean(Neff));

%% ------------------------------------------------------------------------
% 5. Verification against the Q-1 dataset (and the KF if available)
% -------------------------------------------------------------------------
after = t >= 100;                                   % skip the start-up transient
err_pf   = x_true - x_post;
sig_pf   = sqrt([squeeze(P_post(1,1,:)), squeeze(P_post(2,2,:)), ...
                 squeeze(P_post(3,3,:)), squeeze(P_post(4,4,:))]).';
rmse_pf      = sqrt(mean(err_pf.^2, 2));
rmse_pf_ss   = sqrt(mean(err_pf(:,after).^2, 2));
inside2s_pf  = 100*mean(abs(err_pf) <= 2*sig_pf, 2);
rmse_meas    = sqrt(mean((y_meas/kc - x_true(1:2,:)).^2, 2));

kf_file = fullfile(here, 'kf_results.mat');
have_kf = isfile(kf_file);
if have_kf
    KF = load(kf_file, 'x_post');
    err_kf     = x_true - KF.x_post;
    rmse_kf    = sqrt(mean(err_kf.^2, 2));
    rmse_kf_ss = sqrt(mean(err_kf(:,after).^2, 2));
end

fprintf('------------------ Verification ------------------\n');
fprintf('RMSE in cm (whole run | after 100 s)\n');
fprintf('State |  raw sensor |      PF          | in +/-2sig');
if have_kf, fprintf(' |      KF'); end
fprintf('\n');
for i = 1:4
    if i <= 2, raw = sprintf('%6.3f', rmse_meas(i)); else, raw = '  none'; end
    fprintf('  h%d  |   %s    | %6.3f | %6.3f  |  %5.1f%%', ...
            i, raw, rmse_pf(i), rmse_pf_ss(i), inside2s_pf(i));
    if have_kf, fprintf('    | %6.3f | %6.3f', rmse_kf(i), rmse_kf_ss(i)); end
    fprintf('\n');
end
if ~have_kf
    fprintf('(run Kalman_Filter.m first to get the KF comparison)\n');
end

%% ------------------------------------------------------------------------
% 6. Plots
% -------------------------------------------------------------------------
% 6.1 Prior and posterior estimates vs truth
plot_states(t, x_true, x_prior, y_meas/kc, 'PF prior estimate x(k^-)', 'PF prior x(k^-)');
plot_states(t, x_true, x_post,  y_meas/kc, 'PF posterior estimate x(k^+)', 'PF posterior x(k^+)');

% 6.2 Prior and posterior covariance of the particle cloud
plot_cov(t, P_prior, 'PF prior covariance P(k^-)');
plot_cov(t, P_post,  'PF posterior covariance P(k^+)');

% 6.3 Residuals
figure('Name', 'PF residuals', 'Position', [100 80 1000 550]);
for j = 1:2
    subplot(2,1,j);
    plot(t, r_prior(j,:), 'Color', [0.3 0.5 0.9]); hold on;
    plot(t, r_post(j,:),  'Color', [0.2 0.6 0.3]); grid on;
    xlabel('t [s]'); ylabel(sprintf('r_%d [V]', j));
    title(sprintf('Residuals for y_%d', j));
    legend('prior  y - C x(k^-)', 'posterior  y - C x(k^+)');
end
sgtitle('PF residuals');

% 6.4 Effective number of particles
figure('Name', 'Effective particles', 'Position', [120 100 1000 380]);
plot(t, Neff, 'Color', [0.4 0.4 0.4]); hold on;
yline(resample_frac*Np, 'r--', 'resample threshold', 'LineWidth', 1.2);
grid on; xlabel('t [s]'); ylabel('N_{eff}');
title(sprintf('Effective number of particles (resampled on %.1f%% of steps)', ...
      100*mean(did_resample)));

% 6.5 How the h3 particle cloud converges
figure('Name', 'h3 particle cloud', 'Position', [140 120 1100 600]);
for s = 1:numel(snap_k)
    subplot(2, 3, s);
    [ctr, cnt] = weighted_hist(snap_X3(s,:), snap_w(s,:), 30);
    bar(ctr, cnt, 1, 'FaceColor', [0.55 0.7 0.95], 'EdgeColor', 'none'); hold on;
    xline(x_true(3, snap_k(s)), 'r', 'LineWidth', 2);
    grid on; xlabel('h_3 [cm]'); ylabel('weight');
    title(sprintf('t = %g s', snap_t(s)));
    if s == 1, legend('particles (weighted)', 'true h_3'); end
end
sgtitle('h_3 particle cloud collapsing onto the truth (h_3 is never measured)');

% 6.6 Verification: error with the cloud's own +/-2 sigma
figure('Name', 'PF estimation error', 'Position', [160 140 1100 650]);
for i = 1:4
    subplot(2,2,i);
    plot(t, err_pf(i,:), 'b'); hold on;
    plot(t, 2*sig_pf(i,:), 'r--', t, -2*sig_pf(i,:), 'r--', 'LineWidth', 1.2);
    grid on; xlabel('t [s]'); ylabel(sprintf('error in h_%d [cm]', i));
    title(sprintf('h_%d: RMSE %.3f cm, %.1f%% inside \\pm2\\sigma', ...
          i, rmse_pf(i), inside2s_pf(i)));
    if i == 1, legend('true - PF estimate', '\pm 2\sigma of the cloud'); end
end
sgtitle(sprintf('PF verification (%d particles)', Np));

% 6.7 KF vs PF
if have_kf
    figure('Name', 'KF vs PF', 'Position', [180 160 1100 650]);
    for i = 1:4
        subplot(2,2,i);
        plot(t, err_kf(i,:), 'Color', [0.85 0.33 0.1]); hold on;
        plot(t, err_pf(i,:), 'Color', [0 0.45 0.74]);
        grid on; xlabel('t [s]'); ylabel(sprintf('error in h_%d [cm]', i));
        title(sprintf('h_%d: KF %.3f cm, PF %.3f cm (RMSE after 100 s)', ...
              i, rmse_kf_ss(i), rmse_pf_ss(i)));
        if i == 1, legend('KF error', 'PF error'); end
    end
    sgtitle('Estimation error: Kalman filter vs particle filter');
end

%% ------------------------------------------------------------------------
% 7. Save
% -------------------------------------------------------------------------
save(fullfile(here, 'pf_results.mat'), 'x_prior', 'x_post', 'P_prior', ...
     'P_post', 'r_prior', 'r_post', 'Neff', 'did_resample', 'rmse_pf', ...
     'rmse_pf_ss', 'Np', 'Q_filter', 'R_filter', 'x_hat0', 'P0', 'pf_time');
fprintf('\nSaved pf_results.mat\n');

%% ========================================================================
% Local functions
% =========================================================================
function dX = fourtank_rhs_vec(X, u, p)
    % Non-linear 4-tank model (Slide 2) for ALL particles at once.
    % X is 4 x Np, each column one particle; returns dh/dt, 4 x Np.
    X = max(X, 0);                                   % keep sqrt real
    A = p.A(:);  a = p.a(:);  g = p.g;
    g1 = p.gamma(1);  g2 = p.gamma(2);
    k1 = p.k(1);      k2 = p.k(2);
    q  = a .* sqrt(2*g*X);                           % outflow of every tank
    dX = [(-q(1,:) + q(3,:) + g1*k1*u(1))     / A(1);
          (-q(2,:) + q(4,:) + g2*k2*u(2))     / A(2);
          (-q(3,:) + (1-g2)*k2*u(2))          / A(3);
          (-q(4,:) + (1-g1)*k1*u(1))          / A(4)];
end

function [m, P] = weighted_stats(X, w)
    % Weighted mean and covariance of the particle cloud.
    m  = X*w(:);
    dX = X - m;
    P  = (dX .* w(:).') * dX.';
end

function idx = systematic_resample(w)
    % Systematic resampling: one random offset, then Np evenly spaced
    % pointers. Particle i is picked about Np*w(i) times. Lower variance
    % than drawing every particle independently.
    Np    = numel(w);
    edges = cumsum(w);  edges(end) = 1;              % guard against rounding
    ptrs  = (rand + (0:Np-1))/Np;
    idx   = zeros(1, Np);
    j = 1;
    for i = 1:Np
        while ptrs(i) > edges(j)
            j = j + 1;
        end
        idx(i) = j;
    end
end

function [ctr, cnt] = weighted_hist(x, w, nb)
    % Histogram where each sample counts with its weight.
    lo = min(x);  hi = max(x);
    if hi - lo < 1e-9, hi = lo + 1e-3; end
    edges = linspace(lo, hi, nb + 1);
    bin   = min(floor((x - lo)/(hi - lo)*nb) + 1, nb);
    cnt   = accumarray(bin(:), w(:), [nb 1]).';
    ctr   = (edges(1:end-1) + edges(2:end))/2;
end

function plot_states(t, x_true, x_hat, h_meas, fig_title, est_label)
    figure('Name', fig_title, 'Position', [60 40 1100 650]);
    for i = 1:4
        subplot(2,2,i); hold on;
        if i <= 2
            plot(t, h_meas(i,:), 'Color', [0.75 0.75 0.75]);
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
    figure('Name', fig_title, 'Position', [70 50 1100 650]);
    for i = 1:4
        subplot(2,3,i);
        semilogy(t, squeeze(P(i,i,:)), 'LineWidth', 1.2); grid on;
        xlabel('t [s]'); ylabel(sprintf('P_{%d%d} [cm^2]', i, i));
        title(sprintf('variance of h_%d', i));
    end
    subplot(2,3,5);
    plot(t, squeeze(P(1,3,:)), 'LineWidth', 1.2); grid on;
    xlabel('t [s]'); ylabel('P_{13} [cm^2]'); title('h_1-h_3 correlation');
    subplot(2,3,6);
    plot(t, squeeze(P(2,4,:)), 'LineWidth', 1.2); grid on;
    xlabel('t [s]'); ylabel('P_{24} [cm^2]'); title('h_2-h_4 correlation');
    sgtitle(fig_title);
end
