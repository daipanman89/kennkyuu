%PLOT_PICARD_CONVERGENCE  保存済みの mfg_iter_XX.mat から、Picard反復の収束履歴を
%                         再構成して表示するスクリプト（run_mfg_picard.m の再実行は不要）。
%
%   各反復のファイルには、ダンピング後の b(t)（bTraj）と、その反復の FP が作った
%   b(t)（traj.b）が保存されている。そこから
%     ・各反復の終端値 b(T)（run_mfg_picard.m の最後の図と同じもの）
%     ・不動点残差 ||Φ(b_k) - b_k|| / ||b_k||（run_mfg_picard.m が表示していたもの）
%   を計算し直す。

clearvars; clc;

% PINN 版は 'mfg_iter_'、FBSM 版（run_mfg_fbsm.m）の結果を見るときは 'fbsm_iter_' にする
prefix = 'mfg_iter_';

files = dir([prefix '*.mat']);
if isempty(files)
    error('mfg_iter_XX.mat が見つかりません。run_mfg_picard.m を一度は実行してください。');
end
nums = zeros(numel(files),1);
for i = 1:numel(files)
    tok = regexp(files(i).name, [prefix '(\d+)\.mat'], 'tokens');
    nums(i) = str2double(tok{1}{1});
end
nums = sort(nums);
nK = numel(nums);

bT     = zeros(4, nK+1);   % 各反復の b(T)（ダンピング後）。列1は初期値
relRes = zeros(1, nK);     % 不動点残差
bOld = [];
phiB = []; phiX = [];      % 各反復の FP 出力 Φ(b_k)（血中 b(t)）と母集団平均 x(t)
for k = 1:nK
    S = load(sprintf('%s%02d.mat', prefix, nums(k)), 'bTraj', 'traj');
    if isempty(phiB)
        tPhi = S.traj.t(:)';
        phiB = zeros(4, numel(tPhi), nK);
        phiX = zeros(6, numel(tPhi), nK);
    end
    phiB(:,:,k) = S.traj.b;
    phiX(:,:,k) = S.traj.xMean;
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

%% ---- 最後の反復たちの最適応答 Φ(b_k) を重ねて描く ----
% 残差が下がらない原因が「最適応答が2つ以上の形の間で切り替わっている（bang-bang）」のか
% 「毎回少しずつランダムにずれる（HJB学習の誤差）」のかを見分ける。
nShow = min(10, nK);
ks = nK-nShow+1:nK;
cmap = parula(nShow);
th = tPhi/3600;
figure('Name','最後の反復たちの最適応答 Φ(b_k)');
subplot(2,1,1); hold on;
for j = 1:nShow, plot(th, phiB(1,:,ks(j)), 'Color', cmap(j,:), 'LineWidth',1.1); end
grid on; xlabel('time [hour]'); ylabel('GLU'); title(sprintf('Φ(b_k) の血中GLU（反復 %d〜%d、色が明るいほど後）', nums(ks(1)), nums(ks(end))));
subplot(2,1,2); hold on;
for j = 1:nShow, plot(th, phiX(4,:,ks(j)), 'Color', cmap(j,:), 'LineWidth',1.1); end
yline(1000,'k--'); grid on; xlabel('time [hour]'); ylabel('coa');
title('各反復の FP の coa（母集団平均）');

%% ---- fictitious play の平均（これが平均場均衡の近似） ----
% α_k = 1/(k+1) の混合では、b は全反復の最適応答の平均になる。均衡として報告すべき
% 母集団の状態も、最後の1回の FP ではなく、同じ平均を取った x である。
% 全反復の平均と、後半だけの平均（初期の過渡を除く）の両方を描く。
half = ceil(nK/2):nK;
xAvgAll  = mean(phiX, 3);
xAvgHalf = mean(phiX(:,:,half), 3);
bAvgHalf = mean(phiB(:,:,half), 3);
names = {'x05','x06','x07','x08 (coa)','x09','x10'};
figure('Name','fictitious play の平均：母集団平均 x(t)');
for i = 1:6
    subplot(2,3,i);
    plot(th, xAvgAll(i,:), 'LineWidth',1.2); hold on;
    plot(th, xAvgHalf(i,:), '--', 'LineWidth',1.2);
    if i == 1, yline(5000*5/1.5, 'k:'); elseif i == 4, yline(1000, 'k:'); end
    grid on; xlabel('time [hour]'); title(names{i});
    if i == 1, legend({'全反復の平均', sprintf('後半(%d〜%d)の平均', nums(half(1)), nums(end))}, 'Location','best'); end
end
figure('Name','fictitious play の平均：血中 b(t)');
plot(th, bAvgHalf', 'LineWidth',1.3); grid on; hold on;
yline(5000, 'k:');
xlabel('time [hour]'); ylabel('blood level');
legend({'GLU','VLDL','FFA','GLR'}, 'Location','best');
title(sprintf('後半(%d〜%d)の Φ(b_k) の平均', nums(half(1)), nums(end)));

fprintf('\n=== Picard反復の不動点残差 ===\n');
for k = 1:nK
    fprintf('  反復 %2d : %.4g\n', nums(k), relRes(k));
end
