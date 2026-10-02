%PLOT_LAST_RESULTS  run_mfg_picard.m を再実行せずに、保存済みの
%                    mfg_iter_XX.mat から最新（最大番号）の結果を読み込んで
%                    Figureを再表示するスクリプト。

clearvars; clc;

files = dir('mfg_iter_*.mat');
if isempty(files)
    error('mfg_iter_XX.mat が見つかりません。run_mfg_picard.m を一度は実行してください。');
end

% --- 番号が一番大きいファイル(=最後に完了した反復)を選ぶ ---
nums = zeros(numel(files),1);
for i = 1:numel(files)
    tok = regexp(files(i).name, 'mfg_iter_(\d+)\.mat', 'tokens');
    nums(i) = str2double(tok{1}{1});
end
[~, idx] = max(nums);
fname = files(idx).name;
fprintf('読み込み中: %s (反復 %d)\n', fname, nums(idx));

S = load(fname); % bTraj, valueNet, policyNet, traj, hjbInfo, fpInfo が入っている

%% ---- 1) HJBの学習曲線(loss)を見る ----
figure('Name', sprintf('[%s] HJBのloss推移', fname));
if isfield(S.hjbInfo, 'lossPDEHistory')
    semilogy(S.hjbInfo.lossHistory); hold on;
    semilogy(S.hjbInfo.lossPDEHistory);
    
    if isfield(S.hjbInfo.opts, 'terminalWeight')
        semilogy(S.hjbInfo.opts.terminalWeight .* S.hjbInfo.lossTermHistory);
    else
        iters = (1:length(S.hjbInfo.lossTermHistory))';
        numIters = S.hjbInfo.opts.numIters;
        weightProgress = min(1, iters / (numIters * 0.5));
        dynamicWeights = S.hjbInfo.opts.termWeightInit + weightProgress .* (S.hjbInfo.opts.termWeightFinal - S.hjbInfo.opts.termWeightInit);
        semilogy(dynamicWeights .* S.hjbInfo.lossTermHistory);
    end

    legendEntries = {'Value合計','PDE残差','終端条件(重み込み)'};
    if isfield(S.hjbInfo, 'lossHamHistory')
        semilogy(abs(S.hjbInfo.lossHamHistory));
        legendEntries{end+1} = '|H(policy)| (方策のハミルトニアン)';
    end
    grid on;
    xlabel('HJB iteration'); ylabel('loss (log scale)');
    legend(legendEntries, 'Location','best');
else
    semilogy(S.hjbInfo.lossHistory); grid on;
    xlabel('HJB iteration'); ylabel('loss (log scale)');
end
title(sprintf('HJB loss history (Picard反復 %d時点)', nums(idx)));

%% ---- 2) 血中因子の軌跡 b(t) ----
figure('Name', sprintf('[%s] 血中因子の軌跡', fname));
plot(S.bTraj.t/3600, S.bTraj.val', 'LineWidth', 1.5); grid on;
xlabel('time [hour]'); ylabel('blood level');
legend({'GLU','VLDL','FFA','GLR'}, 'Location','best');
title(sprintf('b(t)  (Picard反復 %d時点)', nums(idx)));

%% ---- 3) 全細胞内状態(x05~x10)の母集団平均を一斉表示 ----
targetX05 = 5000 * (5/1.5);
targetCoA = 1000;

figure('Name', sprintf('[%s] 細胞内状態の母集団平均 (x05~x10)', fname));
% 各状態の簡易ラベル（モデルに合わせて適宜読み替えてください）
state_names = {'x05 (グリコーゲン相当)', 'x06', 'x07', 'x08 (CoA)', 'x09', 'x10'};

for i = 1:6
    subplot(2,3,i);
    plot(S.traj.t/3600, S.traj.xMean(i,:), 'LineWidth',1.5); hold on;
    
    % 目標値が設定されているx05とx08にはターゲットラインを引く
    if i == 1
        yline(targetX05, 'k--', 'LineWidth', 1.2, 'DisplayName','target');
    elseif i == 4
        yline(targetCoA, 'k--', 'LineWidth', 1.2, 'DisplayName','target');
    end
    
    grid on; 
    xlabel('time [hour]'); 
    ylabel(sprintf('x%02d (mean)', i+4));
    title(state_names{i});
    
    if i == 1 || i == 4
        legend({'population mean','target'}, 'Location','best');
    else
        legend({'population mean'}, 'Location','best');
    end
end

fprintf('\nこの反復のloss最終値 = %.6g\n', S.hjbInfo.lossHistory(end));
fprintf('b(T) = [%s]\n', mat2str(S.bTraj.val(:,end)',4));

%% ---- 4) 方策の出力a(t)（coaの流入/流出の綱引きを確認） ----
if isfield(S.traj, 'aMean')
    figure('Name', sprintf('[%s] 方策の出力a(t)の母集団平均', fname));
    subplot(2,1,1);
    plot(S.traj.t/3600, S.traj.aMean(1,:), 'LineWidth',1.3);
    grid on; xlabel('time [hour]'); ylabel('a1 (act\_Ins/act\_Gcaの元)');
    title('a1: 高いほどインスリン優位、低いほどグルカゴン優位');

    subplot(2,1,2);
    plot(S.traj.t/3600, S.traj.aMean(5,:), 'LineWidth',1.3); hold on;
    plot(S.traj.t/3600, S.traj.aMean(6,:), 'LineWidth',1.3);
    grid on; xlabel('time [hour]'); ylabel('gain');
    legend({'a5 (gain\_r16, インスリン依存の排出)','a6 (gain\_r17, 常時使える排出)'}, 'Location','best');
    title('coaの排出経路のゲイン');
end