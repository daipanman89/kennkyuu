function cs = costate_targets(traj, re, opts)
%COSTATE_TARGETS  FP で使われた制御に沿って随伴方程式（ポントリャーギンの最大原理）を
%                 後ろ向きに解き、軌道上の共状態 p(t) = ∇V(t, x(t)) を求める。
%
%   cs = costate_targets(traj, re, opts)
%
%   HJB 方程式を最適軌道に沿って微分すると（特性曲線法）、価値関数の勾配
%   p(t) = ∇_x V(t, x(t)) は次の随伴方程式を満たす:
%       dp/dt = -∂L/∂x(x, a) - (∂f/∂x(x, a, b))^T p,     p(T) = ∇g(x(T))
%   制御 a(t) を固定した前向き計算で x(t) を求め、これを後ろ向きに積分する。
%   最適制御では包絡線定理により制御の x 依存の項が消えるので、Picard 反復の
%   不動点ではこの p が HJB の ∇V と一致する。hjb_liver_pinn.m はこれを
%   「軌道上で ∇V が p に一致する」という追加の条件として損失に入れる。
%
%   入力
%     traj : fp_liver_particles.m の出力（recordCells>0 で記録した traj.ctrlRec を使う）
%     re   : 反応速度パラメータ
%     opts : hjb_liver_pinn.m の opts（hjbInfo.opts）。コストの重み・目標・tauFast を使う
%   出力
%     cs.t : 1×M      サンプル時刻[sec]
%     cs.x : 6×K×M    状態（K: 記録した細胞数）
%     cs.p : 6×K×M    共状態 ∂V/∂x

    R  = traj.ctrlRec;
    K  = size(R.x0, 2);
    dt = R.dt;
    nSteps = round(R.T / dt);
    nCtrl  = size(R.a, 3);
    stepsPerCtrl = round(R.dtCtrl / dt);
    ctrlIdx = @(s) min(floor((s-1) / stepsPerCtrl) + 1, nCtrl);
    tauFast = max(opts.tauFast, dt);
    sampleEvery = max(1, round(opts.costateEvery / dt));

    % --- 前向き: 記録した制御・血中状態で、ノイズなしで状態を再計算 ---
    X = zeros(6, K, nSteps + 1);
    x = R.x0;
    X(:,:,1) = x;
    for s = 1:nSteps
        k = ctrlIdx(s);
        dx = liver_cell_drift_limited(x, R.a(:,:,k), R.b(:,k), re, tauFast);
        x = max(x + dx .* dt, 0);
        X(:,:,s+1) = x;
    end

    % --- 後ろ向き: 随伴方程式 ---
    h = 1e-3 .* opts.scaleX;              % ヤコビアンの差分刻み
    p = terminal_grad(X(:,:,end), opts);  % p(T) = ∇g(x(T))
    % 時刻 s*dt（s = 0, sampleEvery, 2*sampleEvery, ...）でサンプルする
    sampSteps = 0:sampleEvery:nSteps;
    M = numel(sampSteps);
    cs.t = sampSteps .* dt;
    cs.x = zeros(6, K, M);
    cs.p = zeros(6, K, M);
    if sampSteps(end) == nSteps
        cs.x(:,:,M) = X(:,:,end); cs.p(:,:,M) = p;
    end

    E = kron(eye(6), ones(1, K));          % 6×6K: 列ブロック i が e_i
    hE = h .* E;
    for s = nSteps:-1:1
        k = ctrlIdx(s);
        x = X(:,:,s);
        a = R.a(:,:,k);
        % J^T p を前進差分で: (J^T p)_i = Σ_j p_j ∂f_j/∂x_i
        xx = [x, repmat(x, 1, 6) + hE];
        aa = repmat(a, 1, 7);
        F  = liver_cell_drift_limited(xx, aa, R.b(:,k), re, tauFast);
        f0 = F(:, 1:K);
        JTp = zeros(6, K);
        for i = 1:6
            fi = F(:, K*i + (1:K));
            JTp(i,:) = sum(p .* (fi - f0), 1) ./ h(i);
        end
        p = p + dt .* (running_cost_grad(x, a, opts) + JTp);   % p(t-dt) = p(t) + dt(∂L/∂x + J^T p)
        if mod(s-1, sampleEvery) == 0
            m = (s-1) / sampleEvery + 1;
            cs.x(:,:,m) = x; cs.p(:,:,m) = p;
        end
    end
end

% =====================================================================
function g = terminal_grad(x, opts)
    g = zeros(size(x));
    g(1,:) = 2 .* opts.w_x05 .* (x(1,:) ./ opts.targetX05 - 1) ./ opts.targetX05;
    g(4,:) = 2 .* opts.w_coa .* (x(4,:) ./ opts.targetCoA - 1) ./ opts.targetCoA;
end

function Lx = running_cost_grad(x, a, opts)
%   hjb_liver_pinn.m の running_cost の x による微分（制御コストは x に依存しない）
    Lx = terminal_grad(x, opts);
    over = x(4,:) > opts.targetCoA;
    Lx(4,:) = Lx(4,:) - opts.w_coaPush ./ opts.targetCoA .* over ...
        .* (a(5,:) ./ opts.aHigh(5) + a(6,:) ./ opts.aHigh(6));
end
