%COMPARE_PINN_FBSM  同じ血中軌跡 b(t) を与えたときの、PINN 版の最適応答と FBSM の最適応答を比べる。
%
%   PINN 版の反復 K（mfg_iter_K.mat）の価値ネット V は、反復 K の入力 b(t)
%   （= 反復 K-1 の混合後の bTraj。K=1 なら b0 のまま一定）に対して学習されている。
%   その b(t) を固定して、
%     ・PINN: 各細胞が a = argmin_a H(x, a, ∇V) を10秒ごとに使ったときの軌道とコスト
%     ・FBSM: PINN の制御を初期値にして、FBSM でコストをさらに下げた最適応答
%   を計算する。FBSM はコストを単調に下げるので J_FBSM <= J_PINN となり、その差が
%   「PINN の最適応答がどれだけ最適から離れているか」（最適性ギャップ）になる。
%   また、FBSM の軌道に沿った共状態 p と、PINN の ∇V を比べる。
%   いずれもノイズなし（σ=0）・b(t) 固定で計算する。

clearvars; clc;

K = [];   % 比較する PINN 版の反復番号（[] なら最新）

files = dir('mfg_iter_*.mat');
if isempty(files), error('mfg_iter_XX.mat が見つかりません。'); end
nums = cellfun(@(s) str2double(regexp(s, 'mfg_iter_(\d+)\.mat', 'tokens', 'once')), {files.name});
if isempty(K), K = max(nums); end
fprintf('PINN 版 mfg_iter_%02d.mat の価値ネットを使います。\n', K);
S = load(sprintf('mfg_iter_%02d.mat', K), 'valueNet', 'hjbInfo', 'fpInfo');
opts = S.hjbInfo.opts;                 % PINN と同じコスト・範囲・モデル設定
R = load('re_params.mat', 're'); re = R.re;

% --- 反復 K の入力 b(t) ---
if K > 1 && any(nums == K-1)
    P = load(sprintf('mfg_iter_%02d.mat', K-1), 'bTraj');
    bIn = P.bTraj;
else
    b0 = S.fpInfo.bInit;
    bIn = struct('t', [0, opts.T], 'val', [b0, b0]);
end

% --- 細胞: PINN 版の FP と同じ初期状態（同じ乱数・同じ式） ---
N = 20;                                % 比較に使う細胞数（先頭 N 個）
fpo = S.fpInfo.opts;
rng(fpo.seed);
x0all = max(fpo.scaleX .* (1 + fpo.initSpreadFrac .* randn(6, fpo.numCells)), 0);
x0 = x0all(:, 1:N);

brOpts = opts;
brOpts.dt = 1; brOpts.dtCtrl = 10; brOpts.maxSweeps = 20; brOpts.tolRel = 1e-5;
nCtrl = round(opts.T / brOpts.dtCtrl);
spc = brOpts.dtCtrl / brOpts.dt;
tk = (0:nCtrl-1) * brOpts.dtCtrl;
bk = interp1(bIn.t(:)', bIn.val', tk, 'linear', 'extrap')';

%% ---- PINN の最適応答（閉ループ: 10秒ごとに ∇V から argmin H） ----
fprintf('PINN の制御で前向き計算中...\n');
Apinn = zeros(6, N, nCtrl);
x = x0;
for k = 1:nCtrl
    Apinn(:,:,k) = argmin_policy(S.valueNet, tk(k), x, bk(:,k), re, opts);
    for s = 1:spc
        x = max(x + liver_cell_drift_limited(x, Apinn(:,:,k), bk(:,k), re, max(opts.tauFast, 1)), 0);
    end
end

%% ---- FBSM の最適応答（PINN の制御から出発してコストを下げる） ----
o0 = brOpts; o0.maxSweeps = 0;
[~, outPinn] = fbsm_best_response(x0, bIn, re, o0, Apinn);    % PINN の制御のコストと軌道
[Afbsm, outF] = fbsm_best_response(x0, bIn, re, brOpts, Apinn);

Jp = outPinn.J; Jf = outF.J;
gap = (Jp - Jf) ./ max(abs(Jf), 1e-12);
fprintf('\n=== 最適性ギャップ（同じ b(t)、ノイズなし） ===\n');
fprintf('PINN のコスト（細胞平均）: %.6g\n', mean(Jp));
fprintf('FBSM のコスト（細胞平均）: %.6g\n', mean(Jf));
fprintf('相対ギャップ (J_PINN - J_FBSM)/J_FBSM: 平均 %.3g %%, 最大 %.3g %%\n', 100*mean(gap), 100*max(gap));

%% ---- 図1: 状態と制御の比較（細胞平均） ----
th = [tk, opts.T] / 3600;
XpM = squeeze(mean(outPinn.Xk, 2)); XfM = squeeze(mean(outF.Xk, 2));
figure('Name', sprintf('PINN (mfg_iter_%02d) vs FBSM: 状態', K));
subplot(1,2,1); plot(th, XpM(1,:), 'b', th, XfM(1,:), 'r--', 'LineWidth',1.3); hold on;
yline(opts.targetX05, 'k:'); grid on; xlabel('time [hour]'); title('x05（細胞平均）');
legend({'PINN','FBSM'}, 'Location','best');
subplot(1,2,2); plot(th, XpM(4,:), 'b', th, XfM(4,:), 'r--', 'LineWidth',1.3); hold on;
yline(opts.targetCoA, 'k:'); grid on; xlabel('time [hour]'); title('coa（細胞平均）');

ApM = squeeze(mean(Apinn, 2)); AfM = squeeze(mean(Afbsm, 2));
figure('Name', sprintf('PINN (mfg_iter_%02d) vs FBSM: 制御', K));
an = {'a1','a2','a3','a4','a5','a6'};
for i = 1:6
    subplot(3,2,i);
    plot(tk/3600, ApM(i,:), 'b', tk/3600, AfM(i,:), 'r--', 'LineWidth',1.1);
    grid on; xlabel('time [hour]'); title(an{i});
    if i == 1, legend({'PINN','FBSM'}, 'Location','best'); end
end

%% ---- 図2: FBSM の共状態 p と PINN の ∇V（FBSM の軌道上、細胞1） ----
xF = squeeze(outF.Xk(:,1,1:nCtrl)); pF = squeeze(outF.P(:,1,1:nCtrl));
[~, gV] = dlfeval(@vgrad, S.valueNet, dlarray([xF; tk], 'CB'), opts);
gV = double(extractdata(gV(1:6,:)));
figure('Name', 'FBSM の共状態 p と PINN の ∇V（細胞1）');
idx = [1 4]; lbl = {'x05','x08 (coa)'};
for j = 1:2
    subplot(2,1,j);
    plot(tk/3600, pF(idx(j),:), 'k-', tk/3600, gV(idx(j),:), 'b--', 'LineWidth',1.2); yline(0,'k:');
    grid on; xlabel('time [hour]'); title(['\partialV/\partial' lbl{j}]);
    legend({'FBSM の共状態 p','PINN の \nablaV'}, 'Location','best');
end
sgn = mean(sign(gV(idx,:)) == sign(pF(idx,:)), 2);
fprintf('FBSM の共状態と PINN の ∇V の符号一致: x05 %.0f %%, x08 %.0f %%\n', 100*sgn(1), 100*sgn(2));

save(sprintf('compare_pinn_fbsm_%02d.mat', K), 'K', 'Jp', 'Jf', 'gap', 'Apinn', 'Afbsm', 'outPinn', 'outF', 'bIn', 'x0');

% =====================================================================
function [V, g] = vgrad(valueNet, X, opts)
    cfg = struct('hard', opts.hardTerminal, 'T', opts.T, 'scaleX', opts.scaleX, ...
        'targetCoA', opts.targetCoA, 'targetX05', opts.targetX05, 'w_coa', opts.w_coa, 'w_x05', opts.w_x05);
    V = value_eval(valueNet, X, cfg);
    g = dlgradient(sum(V, 'all'), X);
end
