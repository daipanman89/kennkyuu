function [bTrajNew, traj, info] = fp_liver_particles(policyNet, bInit, opts)
%FP_LIVER_PARTICLES  100細胞（母集団）を有限ホライズン[0,T]でSDE前向き
%                     シミュレーションし、Fokker-Planck方程式の経験分布
%                     近似（パーティクル法）と、母集団が再構成する
%                     血中因子の軌跡 b(t) を得る。
%
%   [bTrajNew, traj, info] = fp_liver_particles(policyNet, bInit, opts)
%
%   このモデルには継続的な外部供給が無いため（DDPGでも初期値を目標値+1000に
%   ずらすだけで、あとは自然に推移させる設計だった）、定常分布は存在しない。
%   そのためこの関数は「burn-inを捨てて後半を平均する」のではなく、
%   [0,T]全体の軌跡 b(t) をそのまま次のPicard反復の入力として返す。
%
%   入力
%     policyNet : hjb_liver_pinn.m で学習した方策ネット（dlnetwork、
%                 入力は正規化済み[x;t]の7次元）。
%                 単体テスト用に function_handle @(t,x) も渡せる
%                 （t: スカラー[sec], x: 6×N → 6×N の制御を返す）。
%     bInit     : 4×1  血中因子の初期値（省略時は目標値+1000）
%     opts      : struct（省略可）。PARSE_OPTS 参照。
%
%   出力
%     bTrajNew : struct  .t(1×M), .val(4×M)  母集団が再構成した血中因子の軌跡
%     traj     : struct  可視化用の詳細な時系列。.xAll(6×N×M)は全100細胞の
%                状態をまるごと記録したもので、次回のhjb_liver_pinn.m呼び出し
%                時にrefTrajとして渡すと、実際に訪れた状態付近を重点的に
%                学習させることができる。
%     info     : struct  シミュレーション設定など

    if nargin < 2 || isempty(bInit)
        targetB = [5000; 1000; 1000; 100];
        bInit = targetB + 1000;
    end
    if nargin < 3, opts = struct(); end
    opts = parse_opts(opts);

    if isfile('re_params.mat')
        S = load('re_params.mat','re');
        re = S.re;
    else
        error(['re_params.mat が見つかりません。model_single.m と同じ ', ...
               're ベクトルを re_params.mat に保存してください。']);
    end

    N       = opts.numCells;
    VoB     = 5; VoL = 1.5; %#ok<NASGU> liver_cell_drift.m 内でのみ使用（bfluxに内蔵済み）
    scaleX  = opts.scaleX;
    sigma   = opts.sigmaFrac .* scaleX;
    T       = opts.T;
    dt      = opts.dt;
    nSteps  = round(T / dt);

    % --- 血管側（model_single.m の "ODEs of Blood vessels" に対応） ---
    r45fun = @(x02, re) re(45) .* x02 ./ (re(46) + x02);
    r46fun = @(x01, re) re(9)  .* x01 ./ (re(10) + x01);
    c3 = 3;

    isFuncPolicy = isa(policyNet, 'function_handle');

    % 共通乱数（common random numbers）: Picard反復ごとに同じ初期分布・同じノイズ列を
    % 使うことで、b(t)の変化を「方策の変化」だけに帰着させ、反復間のMCノイズを消す
    if ~isempty(opts.seed), rng(opts.seed); end

    % --- 初期値の生成（母集団の初期分布。DDPGと同じ「目標値+1000」を中心に） ---
    x0center = scaleX; % 細胞内状態は目標値そのままスタート(+1000は血中因子側のみ)
    x = x0center .* (1 + opts.initSpreadFrac .* randn(6, N));
    x = max(x, 0);
    b = bInit;

    % --- ログ用バッファ ---
    logEvery = max(1, floor(nSteps / opts.numLogPoints));
    nLog = floor(nSteps / logEvery) + 1;
    tLog = zeros(nLog,1);
    bLog = zeros(4, nLog);
    xMeanLog = zeros(6, nLog);
    xStdLog  = zeros(6, nLog);
    xAllLog  = zeros(6, N, nLog); % 100細胞ぶんの状態をまるごと記録
                                   % (次回HJB学習で「実際に訪れた点」として使う)
    aMeanLog = zeros(6, nLog);    % 方策出力aの母集団平均（診断用）
    li = 1;
    tLog(1) = 0; bLog(:,1) = b; xMeanLog(:,1) = mean(x,2); xStdLog(:,1) = std(x,0,2);
    xAllLog(:,:,1) = x;

    aLow  = opts.aLow;
    aHigh = opts.aHigh;

    for step = 1:nSteps
        tCur = (step-1) * dt;

        % --- 方策の評価 ---
        if isFuncPolicy
            a = policyNet(tCur, x); % 6×N の double
        else
            Xn = dlarray(net_input(x, tCur, scaleX, T), 'CB');
            aRaw = forward(policyNet, Xn);
            a = extractdata(aRaw);
        end
        a = min(max(a, aLow), aHigh);

        % --- 各細胞の細胞内ドリフト + 血中側フラックス ---
        bRep = repmat(b, 1, N);
        [dxdt, bflux] = liver_cell_drift(x, a, bRep, re);

        % --- 細胞内状態の更新（Euler-Maruyama） ---
        noise = sigma .* sqrt(dt) .* randn(6, N);
        x = x + dxdt .* dt + noise;
        x = max(x, 0); % 非負制約

        % --- 血中プールの更新 ---
        % liver_cell_drift.m の bflux は「フルVoLの代表エージェント」として
        % 既に VoL/VoB スケーリング込みで計算されている。N細胞に一般化する際は
        % 1細胞あたり VoL/N を負担させて総和する ( = sum(bflux_k)/N = mean(bflux) )
        % のと等価なので、ここでは単純平均するだけでよい（VoL/VoBを再度掛けない）。
        cellContribToBlood = mean(bflux, 2); % 4×1

        r45 = r45fun(b(2), re);
        r46 = r46fun(b(1), re);
        dGLU  = cellContribToBlood(1) - r46;
        dVLDL = cellContribToBlood(2) - r45;
        dFFA  = cellContribToBlood(3) + c3*r45;
        dGLR  = cellContribToBlood(4) + r45;

        b = b + [dGLU; dVLDL; dFFA; dGLR] .* dt;
        b = max(b, 0);

        if mod(step, logEvery) == 0
            li = li + 1;
            tLog(li) = step*dt;
            bLog(:,li) = b;
            xMeanLog(:,li) = mean(x,2);
            xStdLog(:,li)  = std(x,0,2);
            xAllLog(:,:,li) = x;
            aMeanLog(:,li) = mean(a,2);
        end
    end
    tLog = tLog(1:li); bLog = bLog(:,1:li);
    xMeanLog = xMeanLog(:,1:li); xStdLog = xStdLog(:,1:li);
    xAllLog = xAllLog(:,:,1:li);
    aMeanLog = aMeanLog(:,1:li);

    bTrajNew = struct('t', tLog', 'val', bLog);

    traj = struct('t', tLog, 'b', bLog, 'xMean', xMeanLog, 'xStd', xStdLog, ...
                   'xAll', xAllLog, 'aMean', aMeanLog, 'xFinal', x, 'aFinal', a);
    info = struct('opts', opts, 'bInit', bInit);

    save('fp_solution.mat', 'traj', 'info', 'bTrajNew');
    fprintf('[FP] シミュレーション完了。 b(0) = [%s] -> b(T) = [%s]\n', ...
        mat2str(bInit',4), mat2str(bLog(:,end)',4));

    if opts.doPlot
        plot_fp_results(traj);
    end
end

% =====================================================================
function plot_fp_results(traj)
    figure('Name','FP: population statistics');
    subplot(2,1,1);
    plot(traj.t/3600, traj.b'); grid on;
    xlabel('time [hour]'); ylabel('blood level');
    legend({'GLU','VLDL','FFA','GLR'}, 'Location','best');
    title('Blood pool trajectory (shared across 100 cells, finite horizon)');

    subplot(2,1,2);
    plot(traj.t/3600, traj.xMean'); grid on;
    xlabel('time [hour]'); ylabel('mean intracellular level');
    legend({'x05','x06','x07','x08(coa)','x09','x10'}, 'Location','best');
    title('Population mean of intracellular state');

    figure('Name','FP: final population density (empirical, at t=T)');
    labels = {'x05','x06','x07','x08(coa)','x09','x10'};
    for i = 1:6
        subplot(2,3,i);
        histogram(traj.xFinal(i,:), 20);
        title(labels{i});
    end
end

% =====================================================================
function opts = parse_opts(opts)
    defaults = struct( ...
        'numCells',       100, ...
        'scaleX',         [5000;200000;1000;1000;1000;1000], ...
        'sigmaFrac',      0.05 / sqrt(24*3600), ...
        'initSpreadFrac', 0.1, ...
        'dt',             10, ...    % [sec] Euler-Maruyamaの刻み幅（24hホライズンなので少し粗くする）
        'T',              24*3600, ...% [sec] ホライズン（hjb_liver_pinn.mと合わせること）
        'numLogPoints',   500, ...
        'aLow',           zeros(6,1), ...
        'aHigh',          [100;100;10;10;30;30], ... % hjb_liver_pinn.mと揃える(a5,a6の天井を引き上げ)
        'doPlot',         true, ...
        'seed',           [] ...     % 乱数シード（[]なら設定しない）
    );
    fn = fieldnames(defaults);
    for k = 1:numel(fn)
        if ~isfield(opts, fn{k}) || isempty(opts.(fn{k}))
            opts.(fn{k}) = defaults.(fn{k});
        end
    end
end
