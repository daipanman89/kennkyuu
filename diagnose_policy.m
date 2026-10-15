%DIAGNOSE_POLICY  学習済みの方策ネットが、価値関数 V が示す最適制御に
%                 従っているかを調べる診断スクリプト（学習はしないのですぐ終わる）。
%
%   HJB の理論上、最適制御は  a*(t,x) = argmin_a H(x, a, ∇V(t,x))
%     H(x,a,p) = L(x,a) + p・f(x,a,b)   （L: 走行コスト, f: liver_cell_drift）
%   で決まる。最新の mfg_iter_XX.mat の母集団平均の軌道上で
%     ・方策ネットの出力 a_policy
%     ・∇V から直接求めた a*（座標ごとのグリッド探索で H を最小化）
%   を比べる。
%
%   読み方:
%     a* が a_policy と大きく違う（例: a* は a3 を上げ a6 を0にしている）
%        → V は正しい方向を示しているのに、方策ネットが追従できていない（Actor側の問題）
%     a* も a_policy と同じ（a3を上げない、coaを捨てる）
%        → V（∇V）自体が coa を回復させる価値を表せていない（Critic側の問題）

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
[~, idx] = max(nums);
fname = files(idx).name;
fprintf('読み込み中: %s\n', fname);
S = load(fname, 'valueNet', 'policyNet', 'traj', 'hjbInfo');

R = load('re_params.mat', 're');
re = R.re;

opts = S.hjbInfo.opts;
if ~isfield(opts, 'aRef'),         opts.aRef = zeros(6,1); end
if ~isfield(opts, 'hardTerminal'), opts.hardTerminal = false; end
scaleX = opts.scaleX; T = opts.T;
aLow = opts.aLow; aHigh = opts.aHigh;

%% ---- 評価する時刻（母集団平均の軌道から間引く） ----
nEval = 100;
kIdx = unique(round(linspace(1, numel(S.traj.t), nEval)));
K  = numel(kIdx);
tK = S.traj.t(kIdx)';            % 1×K
xK = S.traj.xMean(:, kIdx);      % 6×K
bK = S.traj.b(:, kIdx);          % 4×K

%% ---- ∇V と方策の出力 ----
X = dlarray([xK; tK], 'CB');
[Vk, gradV] = dlfeval(@value_and_grad, S.valueNet, X, opts);
p = double(extractdata(gradV(1:6,:)));   % 6×K  ∂V/∂x
pT = double(extractdata(gradV(7,:)));    % 1×K  ∂V/∂t
Vk = double(extractdata(Vk));

if isfield(opts, 'controlMode') && strcmp(opts.controlMode, 'argminH')
    % argminH では方策ネットは使っていないので、FPで実際に使われた制御（母集団平均）と比べる
    fprintf('controlMode=argminH: 青線は FP で実際に使われた制御の母集団平均です。\n');
    aPol = S.traj.aMean(:, kIdx);
else
    aPol = double(extractdata(forward(S.policyNet, dlarray(net_input(xK, tK, scaleX, T), 'CB'))));
end
aPol = min(max(aPol, aLow), aHigh);

%% ---- a* = argmin_a H を座標ごとのグリッド探索で求める ----
nGrid = 401;
nSweep = 4;
aOpt = zeros(6, K);
for k = 1:K
    x = xK(:,k); b = bK(:,k); pk = p(:,k);
    a = min(max(opts.aRef, aLow), aHigh);
    for sweep = 1:nSweep
        for i = 1:6
            % a1 は Ka=10 付近で感度が高いので 0 近くを細かく取る
            grid = unique([linspace(aLow(i), aHigh(i), nGrid), ...
                           aLow(i) + logspace(-3, log10(aHigh(i)-aLow(i)), 200)]);
            A = repmat(a, 1, numel(grid));
            A(i,:) = grid;
            Hg = hamiltonian(repmat(x,1,numel(grid)), A, b, pk, re, opts);
            [~, j] = min(Hg);
            a(i) = grid(j);
        end
    end
    aOpt(:,k) = a;
end

Hpol = zeros(1,K); Hopt = zeros(1,K);
for k = 1:K
    Hpol(k) = hamiltonian(xK(:,k), aPol(:,k), bK(:,k), p(:,k), re, opts);
    Hopt(k) = hamiltonian(xK(:,k), aOpt(:,k), bK(:,k), p(:,k), re, opts);
end

%% ---- 図1: 方策 vs ∇V から求めた最適制御 ----
th = tK/3600;
aNames = {'a1 (act\_Ins/Gca)','a2 (act\_MTP)','a3 (coa流入: グルコース由来)', ...
          'a4 (coa流入: 脂肪酸由来)','a5 (coa排出)','a6 (coa排出)'};
figure('Name', sprintf('[%s] 方策ネット vs argmin H (∇Vから求めた最適制御)', fname));
for i = 1:6
    subplot(3,2,i);
    plot(th, aPol(i,:), 'b-', 'LineWidth',1.3); hold on;
    plot(th, aOpt(i,:), 'r--', 'LineWidth',1.3);
    grid on; xlabel('time [hour]'); title(aNames{i});
    ylim([aLow(i), aHigh(i)] + [-0.05 0.05]*(aHigh(i)-aLow(i)));
    if i == 1, legend({'方策ネット a_{policy}','argmin_a H (∇V由来)'}, 'Location','best'); end
end

%% ---- 図2: 価値関数の勾配と H のギャップ ----
figure('Name', sprintf('[%s] ∇V と Hのギャップ', fname));
subplot(3,1,1);
plot(th, p(1,:), 'LineWidth',1.3); yline(0,'k:'); grid on;
xlabel('time [hour]'); ylabel('\partialV/\partialx05');
title('\partialV/\partialx05（x05が目標より低いなら負が正しい）');
subplot(3,1,2);
plot(th, p(4,:), 'LineWidth',1.3); yline(0,'k:'); grid on;
xlabel('time [hour]'); ylabel('\partialV/\partialx08');
title('\partialV/\partialx08（coaが目標より低いなら負が正しい）');
subplot(3,1,3);
plot(th, Hpol - Hopt, 'LineWidth',1.3); grid on;
xlabel('time [hour]'); ylabel('H(a_{policy}) - H(a^*)');
title('方策ネットの最適性ギャップ（0に近いほど方策が∇Vに従っている）');

%% ---- 図3: 相対残差の分母（scaleRes）と residualEps の比較 ----
% hjb_liver_pinn.m の相対残差は residual / (|∂tV| + |L| + Σ|∂V/∂x_i f_i| + |½σ²ΔV| + eps)。
% 軌道上で分母の各項が eps に比べてどれくらいかを見て、eps が下限として効いて
% 感度を鈍らせていないか（特に目標付近、coa方向の項 |∂V/∂x08 f08|）を確認する。
% （½σ²ΔV は σ が非常に小さいので省略）
eps0 = 1;
if isfield(opts, 'residualEps'), eps0 = opts.residualEps; end
Lk = zeros(1,K); pf = zeros(6,K);
for k = 1:K
    [Lk(k), pf(:,k)] = hjb_terms(xK(:,k), aOpt(:,k), bK(:,k), p(:,k), re, opts);
end
scaleK = abs(pT) + abs(Lk) + sum(abs(pf), 1);
figure('Name', sprintf('[%s] 相対残差の分母と residualEps', fname));
semilogy(th, scaleK, 'k-', 'LineWidth',1.5); hold on;
semilogy(th, abs(pT), 'LineWidth',1.1);
semilogy(th, abs(Lk), 'LineWidth',1.1);
semilogy(th, sum(abs(pf),1), 'LineWidth',1.1);
semilogy(th, abs(pf(4,:)), 'LineWidth',1.3);
yline(eps0, 'r--', 'LineWidth',1.3);
grid on; xlabel('time [hour]'); ylabel('magnitude [1/sec]');
legend({'分母の合計(epsなし)','|\partial_tV|','|L|','\Sigma|\partialV/\partialx_i f_i|', ...
        '|\partialV/\partialx08 \cdot f08| (coa方向)', sprintf('residualEps = %g', eps0)}, 'Location','best');
title('相対残差の分母の内訳（赤線より下の項は eps に埋もれて感度が鈍る）');

%% ---- 数値のまとめ ----
low = xK(4,:) < opts.targetCoA;
fprintf('\n=== 診断結果 ===\n');
fprintf('coaが目標未満の時刻のうち dV/dx08 < 0 （正しい符号）の割合: %.0f %%\n', 100*mean(p(4,low) < 0));
fprintf('x05が目標未満の時刻のうち dV/dx05 < 0 （正しい符号）の割合: %.0f %%\n', ...
    100*mean(p(1, xK(1,:) < opts.targetX05) < 0));
fprintf('H(a_policy) - H(a*) の平均: %.4g   （H(a_policy) の平均絶対値 %.4g）\n', ...
    mean(Hpol - Hopt), mean(abs(Hpol)));
fprintf('分母(epsなし) < residualEps となる時刻の割合: %.0f %%   （分母の中央値 %.3g, eps = %g）\n', ...
    100*mean(scaleK < eps0), median(scaleK), eps0);
fprintf('coa方向の項 |dV/dx08*f08| の中央値: %.3g   （eps に対する比 %.3g）\n', ...
    median(abs(pf(4,:))), median(abs(pf(4,:)))/eps0);
names = {'a1','a2','a3','a4','a5','a6'};
fprintf('時間平均  方策 / argmin H :\n');
for i = 1:6
    fprintf('  %s : %8.3f / %8.3f\n', names{i}, mean(aPol(i,:)), mean(aOpt(i,:)));
end

% =====================================================================
function [V, gradV] = value_and_grad(valueNet, X, opts)
    Xn = [X(1:6,:) ./ opts.scaleX; X(7,:) ./ opts.T];
    NN = forward(valueNet, Xn);
    if opts.hardTerminal
        V = terminal_cost(X(1:6,:), opts) + (opts.T - X(7,:)) .* NN;
    else
        V = NN;
    end
    gradV = dlgradient(sum(V, 'all'), X);
end

function g = terminal_cost(x, opts)
    g = opts.w_coa .* (x(4,:)./opts.targetCoA - 1).^2 ...
      + opts.w_x05 .* (x(1,:)./opts.targetX05 - 1).^2;
end

function [L, pf] = hjb_terms(x, a, b, p, re, opts)
%   走行コスト L と、∇V・f の成分ごとの値 p_i*f_i（6×1）
    tauFast = 1;
    if isfield(opts, 'tauFast'), tauFast = opts.tauFast; end
    dxdt = liver_cell_drift_limited(x, a, b, re, tauFast);
    H = hamiltonian(x, a, b, p, re, opts);
    pf = p .* dxdt;
    L = H - sum(pf, 1);
end

function H = hamiltonian(x, a, b, p, re, opts)
%   hjb_liver_pinn.m の running_cost と同じ走行コスト + p・f
    tauFast = 1;
    if isfield(opts, 'tauFast'), tauFast = opts.tauFast; end
    dxdt = liver_cell_drift_limited(x, a, b, re, tauFast);
    coaExcess = max((x(4,:) - opts.targetCoA) ./ opts.targetCoA, 0);
    bonus = opts.w_coaPush .* coaExcess .* (a(5,:)./opts.aHigh(5) + a(6,:)./opts.aHigh(6));
    L = opts.w_coa .* (x(4,:)./opts.targetCoA - 1).^2 ...
      + opts.w_x05 .* (x(1,:)./opts.targetX05 - 1).^2 ...
      + opts.w_ctrl .* sum((a - opts.aRef).^2, 1) ...
      - bonus;
    H = L + sum(p .* dxdt, 1);
end
