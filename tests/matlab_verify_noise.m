% matlab_verify_noise.m
%
% Exports deterministic noise components from the MATLAB reference (gari_pmNoise.m)
% to CSV files, so plot_noise_verification.py can compare them to the Python port.
%
% Usage
% -----
%   Run this script from any directory:
%        run('/path/to/tests/matlab_verify_noise.m')
%
% Outputs
% -------
%             tests/matlab_verification/
%       spm_drift.csv          — full (N x n_basis) drift basis matrix
%       drift_signal.csv       — drift component signal (N x 1)
%       cardiac_jitter0.csv    — cardiac sinusoid, jitter=0 (N x 1)
%       respiratory_jitter0.csv — respiratory sinusoid, jitter=0 (N x 1)
%       time_axis.csv          — time vector in seconds (N x 1)
%       params.csv             — scalar parameters for Python to verify against

% ─── output directory ────────────────────────────────────────────────────────
script_dir = fileparts(mfilename('fullpath'));
outdir     = fullfile(script_dir, 'matlab_verification');
if ~exist(outdir, 'dir'), mkdir(outdir); end
fprintf('Output directory: %s\n\n', outdir);

% ─── shared parameters (match Python test defaults) ──────────────────────────
TR   = 1.5;          % seconds
N    = 100;          % number of time points
t    = (0:N-1) * TR; % time axis (row vector, seconds)

% 'mid' voxel preset (from gari_pmNoise.m defaultsGet)
a_white    = 0.032;
a_cardiac  = 0.01;   f_cardiac  = 1.05;
a_resp     = 0.01;   f_resp     = 0.3;
a_drift    = 0.01;   f_drift    = 120.0; % seconds (period)

% ─── 1. spm_drift matrix ─────────────────────────────────────────────────────
% n_basis mirrors the formula in gari_pmNoise.m line 391:
%   n = floor(2 * (timePointsN * TR) / lowfrequ_frequ + 1)
n_basis = floor(2 * (N * TR) / f_drift + 1);
fprintf('n_basis for drift: %d\n', n_basis);

% Inline copy of THIS PROJECT's spm_drift.m (prf_models/spm_drift.m), not
% generic SPM12's own spm_drift.m of the same name. The two differ: this
% project's non-DC columns carry an extra *10 ("scaled pretty arbitrarily by
% sqrt(2/N)*10", per that file's own comment), which SPM12's canonical
% function does not apply. Calling whichever "spm_drift" happens to be on
% the MATLAB path (e.g. after addpath('/path/to/spm12')) risked silently
% picking up SPM12's un-scaled version instead, which would export a drift
% matrix 10x too small in its non-DC columns relative to the Python port
% (which deliberately reproduces the *10 project convention) -- an
% inconsistency that would look like a Python bug in fig1_deterministic.png
% but was actually just this script computing the wrong reference.
%
% Note: the real spm_drift.m also has a trailing
% "vectOnes(2)=1; vectOnes(3)=1; C = C*diag(vectOnes)" step, but vectOnes
% starts as ones(1,K), so that step is a no-op as written -- omitted here.
i_vec = (1:N)';
k_vec = 2:n_basis;              % MATLAB column index k, matching the
                                 % original's "for k = 2:K" loop exactly
C = zeros(N, n_basis);
C(:, 1) = 1 / sqrt(N);
C(:, 2:end) = sqrt(2 / N) * 10 * cos(pi * (2 * i_vec - 1) .* (k_vec - 1) / (2 * N));

writematrix(C, fullfile(outdir, 'spm_drift.csv'));
fprintf('Saved spm_drift.csv  shape: %dx%d\n', size(C,1), size(C,2));

% ─── 2. Drift signal ─────────────────────────────────────────────────────────
% Sum non-DC columns (columns 2:end), scaled by amplitude.
% Matches gari_pmNoise.m lines 396-399 with jitter=0.
drift_basis  = C(:, 2:end);
drift_signal = a_drift * sum(drift_basis, 2);   % (N x 1)

writematrix(drift_signal, fullfile(outdir, 'drift_signal.csv'));
fprintf('Saved drift_signal.csv\n');

% ─── 3. Cardiac sinusoid (jitter = 0) ────────────────────────────────────────
% With jitter=0, no randn draws are made; the sinusoid is fully deterministic.
% Matches gari_pmNoise.m lines 347-355 with jitter=[0,0,0].
cardiac = (a_cardiac * sin(2 * pi * t * f_cardiac))';   % (N x 1)

writematrix(cardiac, fullfile(outdir, 'cardiac_jitter0.csv'));
fprintf('Saved cardiac_jitter0.csv\n');

% ─── 4. Respiratory sinusoid (jitter = 0) ────────────────────────────────────
resp = (a_resp * sin(2 * pi * t * f_resp))';   % (N x 1)

writematrix(resp, fullfile(outdir, 'respiratory_jitter0.csv'));
fprintf('Saved respiratory_jitter0.csv\n');

% ─── 5. Time axis ────────────────────────────────────────────────────────────
writematrix(t', fullfile(outdir, 'time_axis.csv'));
fprintf('Saved time_axis.csv\n');

% ─── 6. Scalar parameters (for Python to cross-check) ────────────────────────
params = table(TR, N, n_basis, a_white, a_cardiac, f_cardiac, ...
               a_resp, f_resp, a_drift, f_drift, ...
               'VariableNames', {'TR','N','n_basis','a_white', ...
               'a_cardiac','f_cardiac','a_resp','f_resp','a_drift','f_drift'});
writetable(params, fullfile(outdir, 'params.csv'));
fprintf('Saved params.csv\n');

% ─── summary ─────────────────────────────────────────────────────────────────
fprintf('\nColumn norms of spm_drift (should be [1, 10, 10, ...], project convention):\n');
disp(sqrt(sum(C.^2)));

fprintf('\nDrift signal range: [%.6f, %.6f]\n', min(drift_signal), max(drift_signal));
fprintf('Cardiac range:      [%.6f, %.6f]\n', min(cardiac), max(cardiac));
fprintf('Respiratory range:  [%.6f, %.6f]\n', min(resp), max(resp));

fprintf('\nDone. Now run:  python tests/plot_noise_verification.py\n');
