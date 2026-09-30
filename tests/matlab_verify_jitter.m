% matlab_verify_jitter.m
%
% Recovers the (frequency, amplitude, phase) each of 100 seeded jitter draws
% used, for cardiac/respiratory/drift, at jitter = 0.1, 0.4, and 0.5 (the original
% author's own demo values -- see gari_pmNoise.m's commented example usage
% blocks near the end of the file).
%
% Cross-language sample-by-sample comparison is impossible here (MATLAB's
% randn and NumPy's standard_normal diverge even from an identical seed --
% see pm_noise.py's _make_rng() docstring), so instead we recover the exact
% scalar (f, a, p) each draw used by replicating gari_pmNoise.m's own RNG
% draw order with a fresh rng(seed) per draw:
%     cardiac / respiratory : 3 draws, order = [freq, amp, phase]
%     drift                 : 2 draws, order = [freq, amp]  (no phase term)
% This mirrors tests/jitter_recovery_utils.py exactly, which was verified
% bit-exact against the real pm_noise.py code path
% (see verify_recovery_matches_pm_noise()).
%
% Usage: run('/path/to/tests/matlab_verify_jitter.m')
% Outputs: tests/matlab_verification/jitter_<component>_j<level>.csv
%          columns: seed, f, a, p, valid
%          (valid = 0 for drift draws where the perturbed period yields
%           n_basis < 3 -- an expected edge case, see the .py module)

script_dir = fileparts(mfilename('fullpath'));
outdir     = fullfile(script_dir, 'matlab_verification');
if ~exist(outdir, 'dir'), mkdir(outdir); end

TR = 1.5;
N  = 100;

nominal.cardiac     = struct('freq', 1.05,  'amp', 0.01);
nominal.respiratory = struct('freq', 0.30,  'amp', 0.01);
nominal.drift        = struct('freq', 120.0, 'amp', 0.01);

jitter_levels = [0.1, 0.4, 0.5];  % 0.4 is the original author's own demo value,
                                   % mislabeled "0.5 jitter" in that file's comment
n_seeds       = 100;
components    = {'cardiac', 'respiratory', 'drift'};

for jidx = 1:numel(jitter_levels)
    j = jitter_levels(jidx);
    for cidx = 1:numel(components)
        comp = components{cidx};
        nom  = nominal.(comp);

        seed_col  = (0:(n_seeds-1))';
        f_col     = zeros(n_seeds, 1);
        a_col     = zeros(n_seeds, 1);
        p_col     = zeros(n_seeds, 1);
        valid_col = ones(n_seeds, 1);

        for i = 1:n_seeds
            seed = seed_col(i);
            rng(seed, 'twister');
            f = nom.freq * (1 + j * randn(1,1));
            a = nom.amp  * (1 + j * randn(1,1));
            if strcmp(comp, 'drift')
                p = 0;
                n_basis = floor(2 * (N * TR) / f + 1);
                valid_col(i) = n_basis >= 3;
            else
                p = 2 * pi * 0 * randn(1,1);  % phase jitter = 0 (scalar form)
            end
            f_col(i) = f;
            a_col(i) = a;
            p_col(i) = p;
        end

        T = table(seed_col, f_col, a_col, p_col, valid_col, ...
                  'VariableNames', {'seed','f','a','p','valid'});
        fname = sprintf('jitter_%s_j%.2f.csv', comp, j);
        writetable(T, fullfile(outdir, fname));
        fprintf('Saved %s  (n_valid=%d/%d)\n', fname, sum(valid_col), n_seeds);
    end
end

fprintf('\nDone. Now run:  python tests/plot_jitter_verification.py\n');
