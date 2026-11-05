%RUN_MFG_FBSM  MFG の平均場均衡を、最適応答を FBSM（前向き・後ろ向きスイープ法）で
%              求めて計算する。PINN 版（run_mfg_picard.m）の参照解を作るためのもの。
%
%   外側の反復は PINN 版とまったく同じ:
%     b(t) を所与 → 各細胞の最適応答 → FP（100細胞）→ 新しい b(t) → fictitious play で混合
%   違うのは「最適応答」の求め方だけで、
%     PINN 版: HJB を PINN で解き、a* = argmin_a H(x, a, ∇V)
%     FBSM 版: 随伴方程式（ポントリャーギン）を前向き・後ろ向きに解き、a* = argmin_a H(x, a, p)
%   最適軌道の上では p(t) = ∇V(t, x(t))（特性曲線）なので、両者は同じ MFG 系を解いている。
%   モデル（liver_cell_drift_limited）、コスト、制御の範囲、FP、初期の細胞、乱数列は
%   PINN 版と共通。
%
%   結果は fbsm_iter_XX.mat に保存する。plot_picard_convergence.m で prefix を
%   'fbsm_iter_' にすれば、PINN 版と同じ図で収束を確認できる。

clearvars; clc;

%% ---- 設定（PINN 版と揃える） ----
T = 24*3600;
maxOuterIter = 40;
tolRel       = 1e-3;
resumeFrom   = 0;    % 途中から再開する場合は反復番号（fbsm_state.mat の制御も使う）

costOpts = mfg_cost_opts();
costOpts.T = T;
brOpts = costOpts;          % 最適応答（FBSM）の設定
brOpts.dt        = 1;       % 前向き・後ろ向きの時間刻み[秒]（FP と同じ）
brOpts.dtCtrl    = 10;      % 制御を一定に保つ区間[秒]（PINN 版 FP の controlEvery と同じ）
brOpts.maxSweeps = 20;
brOpts.tolRel    = 1e-4;

N = 100;
fpOpts = struct('T', T, 'numCells', N, 'dt', 1, 'doPlot', false, ...
                'seed', 1, 'controlEvery', brOpts.dtCtrl / 1);
R = load('re_params.mat', 're'); re = R.re;

b0 = [6000; 2000; 1000; 100];   % DDPG・PINN 版と同じ血中初期値

% 細胞の初期状態: fp_liver_particles.m と同じ乱数・同じ式で作る（PINN 版と同じ細胞）
rng(fpOpts.seed);
x0 = costOpts.scaleX .* (1 + 0.1 .* randn(6, N));
x0 = max(x0, 0);
fpOpts.x0 = x0;

nCtrl = round(T / brOpts.dtCtrl);
bTraj = struct('t', [0, T], 'val', [b0, b0]);
A = [];   % 制御（6×N×nCtrl）。反復ごとに前回の最適応答から始める

if resumeFrom > 0
    Rz = load(sprintf('fbsm_iter_%02d.mat', resumeFrom), 'bTraj');
    bTraj = Rz.bTraj;
    Sz = load('fbsm_state.mat', 'A');
    A = Sz.A;
    fprintf('fbsm_iter_%02d.mat から再開します。\n', resumeFrom);
end

for k = resumeFrom+1:maxOuterIter
    fprintf('\n========== FBSM-MFG 反復 %d / %d ==========\n', k, maxOuterIter);
    tic;

    % --- 1) 最適応答: b(t) を所与として、各細胞の最適制御を FBSM で求める ---
    [A, brInfo] = fbsm_best_response(x0, bTraj, re, brOpts, A);
    % 次の反復は、今回うまくいった緩和係数の2倍から始める（小さくなりすぎないよう下限つき）
    brOpts.omega0 = min(1, max(2 .* brInfo.omega, 1/64));

    % --- 2) FP: その制御で100細胞をシミュレーションし、新しい b(t) を作る ---
    Ahold = A;
    fpPolicy = @(t, x, b) Ahold(:, :, min(floor(t / brOpts.dtCtrl) + 1, nCtrl));
    [bTrajNew, traj, fpInfo] = fp_liver_particles(fpPolicy, bTraj.val(:,1), fpOpts); %#ok<ASGLU>

    % --- 3) fictitious play で混合、収束判定（PINN 版と同じ） ---
    bOldOnNewGrid = interp1(bTraj.t, bTraj.val', bTrajNew.t, 'linear', 'extrap')';
    alpha = 1 / (k + 1);
    bUpdatedVal = (1 - alpha) .* bOldOnNewGrid + alpha .* bTrajNew.val;
    relChange = norm(bTrajNew.val - bOldOnNewGrid, 'fro') / max(norm(bOldOnNewGrid, 'fro'), 1e-8);

    fprintf('b_new(T)(FPより) = [%s]\n', mat2str(bTrajNew.val(:,end)', 4));
    fprintf('b_updated(T)     = [%s]   (不動点残差 %.4g, 混合率 %.3g, %.0f秒)\n', ...
        mat2str(bUpdatedVal(:,end)', 4), relChange, alpha, toc);

    bTraj = struct('t', bTrajNew.t, 'val', bUpdatedVal);
    aMean = squeeze(mean(A, 2));     % 6×nCtrl（保存用。A そのものは大きいので fbsm_state.mat のみ）
    brHist = brInfo.hist; brJ = brInfo.J;
    save(sprintf('fbsm_iter_%02d.mat', k), 'bTraj', 'traj', 'fpInfo', 'aMean', 'brHist', 'brJ', ...
        'costOpts', 'brOpts', '-v7.3');
    save('fbsm_state.mat', 'A', 'x0', 'bTraj', '-v7.3');

    if relChange < tolRel
        fprintf('\n収束しました（不動点残差 %.4g < 許容値 %.4g）。反復 %d で終了。\n', relChange, tolRel, k);
        break;
    end
end

%% ---- 最終結果の簡単な図 ----
th = traj.t / 3600;
figure('Name', 'FBSM-MFG: 最終反復の母集団平均と制御');
subplot(2,2,1); plot(th, traj.xMean(1,:), 'LineWidth',1.3); hold on;
yline(costOpts.targetX05, 'k--'); grid on; xlabel('time [hour]'); title('x05');
subplot(2,2,2); plot(th, traj.xMean(4,:), 'LineWidth',1.3); hold on;
yline(costOpts.targetCoA, 'k--'); grid on; xlabel('time [hour]'); title('coa (x08)');
subplot(2,2,3); plot(bTraj.t/3600, bTraj.val', 'LineWidth',1.3); grid on;
xlabel('time [hour]'); legend({'GLU','VLDL','FFA','GLR'}, 'Location','best'); title('b(t)（混合後）');
subplot(2,2,4); plot((0:nCtrl-1)*brOpts.dtCtrl/3600, aMean', 'LineWidth',1.1); grid on;
xlabel('time [hour]'); legend({'a1','a2','a3','a4','a5','a6'}, 'Location','best'); title('制御の細胞平均');
