%VALIDATE_FP_SEEDS  収束した方策を固定したまま、FP（100細胞シミュレーション）を
%                   乱数シードだけ変えて複数回実行し、結果がシードに依存しないかを確かめる。
%
%   run_mfg_picard.m では反復中のMCノイズを消すために FP のシードを固定している
%   （共通乱数）。そのため得られる不動点は「その1本の乱数列の下での不動点」であり、
%   報告前に別のシードでも同じ b(t) が再現されることを確認しておく必要がある。
%   ばらつき（帯の幅）が大きい場合は fpOpts.numCells を増やすこと。

clearvars; clc;

seeds = 1:5;   % 検証に使うシード（run_mfg_picard.m は seed=1）

files = dir('mfg_iter_*.mat');
if isempty(files)
    error('mfg_iter_XX.mat が見つかりません。run_mfg_picard.m を一度は実行してください。');
end
nums = zeros(numel(files),1);
for i = 1:numel(files)
    tok = regexp(files(i).name, 'mfg_iter_(\d+)\.mat', 'tokens');
    nums(i) = str2double(tok{1}{1});
end
[~, idx] = max(nums);
fname = files(idx).name;
fprintf('読み込み中: %s\n', fname);
S = load(fname, 'policyNet', 'bTraj', 'fpInfo');

fpOpts = S.fpInfo.opts;
fpOpts.doPlot = false;
bInit = S.bTraj.val(:,1);

nS = numel(seeds);
for s = 1:nS
    fpOpts.seed = seeds(s);
    fprintf('\n--- seed = %d (%d / %d) ---\n', seeds(s), s, nS);
    [bTrajS, trajS] = fp_liver_particles(S.policyNet, bInit, fpOpts);
    if s == 1
        t = bTrajS.t;
        bAll = zeros(4, numel(t), nS);
        xAll = zeros(6, numel(t), nS);
    end
    bAll(:,:,s) = bTrajS.val;
    xAll(:,:,s) = trajS.xMean;
end

bMean = mean(bAll, 3); bStd = std(bAll, 0, 3);
xMean = mean(xAll, 3); xStd = std(xAll, 0, 3);

%% ---- 血中因子 b(t)：シード間の平均 ± 標準偏差 ----
names = {'GLU','VLDL','FFA','GLR'};
figure('Name','シード間のばらつき：血中因子 b(t)');
for i = 1:4
    subplot(2,2,i);
    plot_band(t/3600, bMean(i,:), bStd(i,:));
    grid on; xlabel('time [hour]'); ylabel(names{i});
    title(sprintf('%s  (mean \\pm std, %d seeds)', names{i}, nS));
end

%% ---- コストに効く細胞内状態 x05, coa(x08) ----
figure('Name','シード間のばらつき：x05, coa');
subplot(2,1,1);
plot_band(t/3600, xMean(1,:), xStd(1,:));
grid on; xlabel('time [hour]'); ylabel('x05 (population mean)');
title(sprintf('x05  (mean \\pm std, %d seeds)', nS));
subplot(2,1,2);
plot_band(t/3600, xMean(4,:), xStd(4,:));
grid on; xlabel('time [hour]'); ylabel('coa=x08 (population mean)');
title(sprintf('coa  (mean \\pm std, %d seeds)', nS));

%% ---- 数値でも確認：相対ばらつき（std/|mean| の時間方向の最大値） ----
fprintf('\n=== シード間の相対ばらつき max_t std/|mean| ===\n');
for i = 1:4
    fprintf('  %-5s : %.3g\n', names{i}, max(bStd(i,:) ./ max(abs(bMean(i,:)), 1e-8)));
end
fprintf('  x05   : %.3g\n', max(xStd(1,:) ./ max(abs(xMean(1,:)), 1e-8)));
fprintf('  coa   : %.3g\n', max(xStd(4,:) ./ max(abs(xMean(4,:)), 1e-8)));

save('fp_seed_validation.mat', 'seeds', 't', 'bAll', 'xAll', 'bMean', 'bStd', 'xMean', 'xStd');

% =====================================================================
function plot_band(t, m, sd)
    t = t(:)'; m = m(:)'; sd = sd(:)';
    fill([t, fliplr(t)], [m+sd, fliplr(m-sd)], [0.6 0.75 1.0], ...
        'EdgeColor','none', 'FaceAlpha',0.5); hold on;
    plot(t, m, 'b', 'LineWidth', 1.3);
end
