%RUN_MFG_PICARD  HJB(代表エージェント,有限ホライズン,Deep BSDE法) と FP(100細胞) を
%                 交互に解いて、血中因子の軌跡 b(t) の不動点を求める外側ループ。
%
%   このモデルには継続的な外部供給が無い（DDPGでも初期値を目標値+1000に
%   ずらして、あとは自然推移させる設計）ため、b は定数ではなく
%   時間の関数 b(t)（軌跡）として扱う。
%
%   構造（Picard反復 / fictitious play）:
%     1) bTraj_k を所与として hjb_liver_deepbsde.m で有限ホライズンの
%        最適方策 a(t,x) を求める（血中→細胞、一方向の結合）。
%        前回のFPロールアウトの実軌道(refTraj)があれば、チャンク開始点の
%        サンプリングに混ぜる（重点サンプリング）。
%     2) その方策で fp_liver_particles.m を実行し、100細胞を
%        [0,T]で前向きシミュレーション。母集団の代謝フラックスを
%        集計して新しい軌跡 bTraj_{k+1} を再構成する（細胞→血中）
%     3) bTraj_{k+1} と bTraj_k の差が小さくなるまで 1)-2) を繰り返す
%        （ダンピングを入れて振動を抑える。両者は同じ時間グリッドを
%         共有するので、時刻ごとに単純平均でダンピングできる）
%
%   事前準備: re_params.mat に既存モデルの反応速度パラメータ re を保存しておくこと。

clearvars; clc;

%% ---- 共通設定 ----
T = 24*3600; % 24時間（DDPGのt1=360sec×240step相当）

%% ---- Picard反復の設定 ----
maxOuterIter = 20;
damping      = 0.2;
tolRel       = 1e-3;

% PINN版のオプション
hjbOpts = struct('T', T, 'numIters', 40000, 'batchSize', 256, ...
                  'lrValue', 1e-3, 'lrPolicy', 1e-4, 'printEvery', 1000);

fpOpts = struct('T', T, 'numCells', 100, 'dt', 1, 'doPlot', false); % dt=10は不安定と判明(diagnose_stability.m)

%% ---- 初期値（DDPGと同じ「目標値+1000」） ----
targetB = [5000; 1000; 1000; 100]; % [GLU;VLDL;FFA;GLR]
b0 = targetB + 1000;

% 初回のbTraj推定：時間一定（b0のまま）と仮定してスタート
bTraj = struct('t', [0, T], 'val', [b0, b0]);
refTraj = []; % 初回はまだFPの実軌道が無いので、チャンク開始点は一様サンプリングのみ

bHistoryFinal = zeros(4, maxOuterIter+1); % 各反復での b(T)（終端値）の推移を記録（収束の目安表示用）
bHistoryFinal(:,1) = b0;

for k = 1:maxOuterIter
    fprintf('\n========== Picard反復 %d / %d ==========\n', k, maxOuterIter);
    fprintf('現在の b(0) = [%s]\n', mat2str(bTraj.val(:,1)',4));

    % --- 1) HJBを解く（代表エージェントの最適方策、有限ホライズン、PINN法） ---
    % refTraj: 前回のFPロールアウトの実軌道（k=1では空 = 一様サンプリングのみ）
    [valueNet, policyNet, hjbInfo] = hjb_liver_pinn(bTraj, hjbOpts, refTraj); %#ok<ASGLU>

    % --- 2) FPを解く（100細胞シミュレーションで新しいb(t)を再構成） ---
    [bTrajNew, traj, fpInfo] = fp_liver_particles(policyNet, bTraj.val(:,1), fpOpts); %#ok<ASGLU>

    % --- 次回のHJB学習で使う「実際に訪れた点」として今回のtrajを保存 ---
    refTraj = struct('t', traj.t, 'xAll', traj.xAll);

    % --- 3) 時刻グリッドを揃えてダンピング、収束判定 ---
    bOldOnNewGrid = interp1(bTraj.t, bTraj.val', bTrajNew.t, 'linear', 'extrap')';
    bUpdatedVal = (1-damping) .* bOldOnNewGrid + damping .* bTrajNew.val;

    relChange = norm(bUpdatedVal - bOldOnNewGrid, 'fro') / max(norm(bOldOnNewGrid,'fro'), 1e-8);

    fprintf('b_new(T)(FPより) = [%s]\n', mat2str(bTrajNew.val(:,end)',4));
    fprintf('b_updated(T)     = [%s]   (相対変化 %.4g)\n', mat2str(bUpdatedVal(:,end)',4), relChange);

    bTraj = struct('t', bTrajNew.t, 'val', bUpdatedVal);
    bHistoryFinal(:,k+1) = bUpdatedVal(:,end);

    save(sprintf('mfg_iter_%02d.mat', k), 'bTraj', 'valueNet', 'policyNet', 'traj', 'hjbInfo', 'fpInfo', '-v7.3');

    if relChange < tolRel
        fprintf('\n収束しました（相対変化 %.4g < 許容値 %.4g）。反復 %d で終了。\n', ...
            relChange, tolRel, k);
        bHistoryFinal = bHistoryFinal(:,1:k+1);
        break;
    end
end

save('mfg_state.mat', 'bTraj'); % 次回起動時のウォームスタート用

figure('Name','Picard反復の収束履歴（終端値 b(T)）');
plot(0:size(bHistoryFinal,2)-1, bHistoryFinal', '-o'); grid on;
xlabel('outer iteration k'); ylabel('blood level at t=T');
legend({'GLU','VLDL','FFA','GLR'}, 'Location','best');
title('MFG fixed point: convergence of b(T) across outer iterations');

figure('Name','最終的な血中因子の軌跡 b(t)');
plot(bTraj.t/3600, bTraj.val'); grid on;
xlabel('time [hour]'); ylabel('blood level');
legend({'GLU','VLDL','FFA','GLR'}, 'Location','best');
title('MFG equilibrium trajectory b*(t)');

% --- コストに直接効いている細胞内状態（x05, coa=x08）も表示する ---
targetX05 = 5000 * (5/1.5); % hjb_liver_pinn.m のデフォルトと合わせること
targetCoA = 1000;

figure('Name','最終反復：x05(肝グリコーゲン相当)とcoa(x08)の母集団平均');
subplot(2,1,1);
plot(traj.t/3600, traj.xMean(1,:), 'LineWidth',1.3); hold on;
yline(targetX05, 'k--', 'DisplayName','x05 target');
grid on; xlabel('time [hour]'); ylabel('x05 (mean)');
legend({'x05 (population mean)','target'}, 'Location','best');
title('x05: 肝グリコーゲン相当（血中GLUと3.33倍の釣り合い関係）');

subplot(2,1,2);
plot(traj.t/3600, traj.xMean(4,:), 'LineWidth',1.3); hold on;
yline(targetCoA, 'k--', 'DisplayName','coa target');
grid on; xlabel('time [hour]'); ylabel('coa=x08 (mean)');
legend({'coa (population mean)','target'}, 'Location','best');
title('coa(x08): ATP変換経路への基質');

fprintf('\n最終的な血中因子の軌跡（平均場均衡） b*(T) = [%s]\n', mat2str(bTraj.val(:,end)',4));
fprintf('最終反復のx05(T)平均 = %.4g （目標 %.4g）\n', traj.xMean(1,end), targetX05);
fprintf('最終反復のcoa(T)平均 = %.4g （目標 %.4g）\n', traj.xMean(4,end), targetCoA);
