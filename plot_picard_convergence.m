%PLOT_PICARD_CONVERGENCE  保存済みの mfg_iter_XX.mat から、Picard反復の収束履歴を
%                         再構成して表示するスクリプト（run_mfg_picard.m の再実行は不要）。
%
%   各反復のファイルには、ダンピング後の b(t)（bTraj）と、その反復の FP が作った
%   b(t)（traj.b）が保存されている。そこから
%     ・各反復の終端値 b(T)（run_mfg_picard.m の最後の図と同じもの）
%     ・不動点残差 ||Φ(b_k) - b_k|| / ||b_k||（run_mfg_picard.m が表示していたもの）
%   を計算し直す。

clearvars; clc;

files = dir('mfg_iter_*.mat');
if isempty(files)
    error('mfg_iter_XX.mat が見つかりません。run_mfg_picard.m を一度は実行してください。');
end
nums = zeros(numel(files),1);
for i = 1:numel(files)
    tok = regexp(files(i).name, 'mfg_iter_(\d+)\.mat', 'tokens');
    nums(i) = str2double(tok{1}{1});
end
nums = sort(nums);
nK = numel(nums);

bT     = zeros(4, nK+1);   % 各反復の b(T)（ダンピング後）。列1は初期値
relRes = zeros(1, nK);     % 不動点残差
bOld = [];
for k = 1:nK
    S = load(sprintf('mfg_iter_%02d.mat', nums(k)), 'bTraj', 'traj');
    if isempty(bOld)
        % 1回目の反復の入力は「b0 のまま時間一定」（run_mfg_picard.m と同じ）
        b0 = S.traj.b(:,1);
        bOld = struct('t', [0, S.bTraj.t(end)], 'val', [b0, b0]);
        bT(:,1) = b0;
    end
    tNew = S.traj.t(:)';
    bOldOnNew = interp1(bOld.t, bOld.val', tNew, 'linear', 'extrap')';
    relRes(k) = norm(S.traj.b - bOldOnNew, 'fro') / max(norm(bOldOnNew, 'fro'), 1e-8);
    bT(:,k+1) = S.bTraj.val(:,end);
    bOld = S.bTraj;
end

figure('Name','Picard反復の収束履歴（終端値 b(T)）');
plot(0:nK, bT', '-o'); grid on;
xlabel('outer iteration k'); ylabel('blood level at t=T');
legend({'GLU','VLDL','FFA','GLR'}, 'Location','best');
title('MFG fixed point: convergence of b(T) across outer iterations');

figure('Name','Picard反復の不動点残差');
semilogy(nums, relRes, '-o', 'LineWidth',1.3); grid on; hold on;
yline(1e-3, 'r--', 'tolRel = 1e-3');
xlabel('outer iteration k'); ylabel('||\Phi(b_k) - b_k|| / ||b_k||');
title('不動点残差（下がり続けていれば収束に向かっている）');

fprintf('\n=== Picard反復の不動点残差 ===\n');
for k = 1:nK
    fprintf('  反復 %2d : %.4g\n', nums(k), relRes(k));
end
