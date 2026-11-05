function [A, out] = fbsm_best_response(x0, bTraj, re, opts, A0)
%FBSM_BEST_RESPONSE  血中軌跡 b(t) を所与として、各細胞の最適制御（最適応答）を
%                    前向き・後ろ向きスイープ法（FBSM, ポントリャーギンの最大原理）で求める。
%
%   [A, out] = fbsm_best_response(x0, bTraj, re, opts, A0)
%
%   各細胞 m は b(t) を動かせない（平均場を所与とする）ので、細胞ごとに独立な
%   最適制御問題
%       min_a  ∫_0^T L(x, a) dt + g(x(T)),   dx/dt = f(x, a, b(t))
%   を解く。1回のスイープは
%     1) 前向き:  今の制御 A で状態 x(t) を計算
%     2) 後ろ向き: 随伴方程式 dp/dt = -∂L/∂x - (∂f/∂x)^T p,  p(T) = ∇g で共状態 p(t)
%     3) 更新:    A_new = argmin_a H(x, a, p)（optimal_control.m。PINN 版と同じ関数）
%   で、A ← A + ω(A_new - A) とする。ω は細胞ごとに、コストが下がったときだけ
%   受け入れる（下がらなければ ω を半分にして次のスイープでやり直す）ので、
%   各細胞のコストは単調に減る（緩和つき FBSM。McAsey et al., 2012 などと同じ考え方）。
%
%   入力
%     x0    : 6×N   各細胞の初期状態
%     bTraj : struct .t(1×M), .val(4×M)  所与の血中軌跡
%     re    : 反応速度パラメータ
%     opts  : mfg_cost_opts() または hjbInfo.opts に以下を加えたもの
%             .dt (1), .dtCtrl (10), .maxSweeps (20), .tolRel (1e-4),
%             .omega0 (0.25: 緩和係数の初期値。1×N で細胞ごとに与えてもよい)
%     A0    : 6×N×nCtrl  制御の初期値（省略時は aRef を範囲内に収めたもの）
%   出力
%     A   : 6×N×nCtrl   最適制御（dtCtrl 秒ごとのゼロ次ホールド）
%     out : .J（細胞ごとのコスト）, .Xk（区間の始点の状態 6×N×(nCtrl+1)）,
%           .P（区間の始点の共状態）, .hist（スイープごとの総コスト・PMP残差）

    opts = fill_defaults(opts);
    N = size(x0, 2);
    dt = opts.dt;
    spc = round(opts.dtCtrl / dt);                % 1制御区間あたりのステップ数
    nCtrl = round(opts.T / opts.dtCtrl);
    tk = (0:nCtrl-1) * opts.dtCtrl;
    bk = interp1(bTraj.t(:)', bTraj.val', tk, 'linear', 'extrap')';   % 4×nCtrl（区間内は一定）

    if nargin < 5 || isempty(A0)
        A0 = repmat(min(max(opts.aRef, opts.aLow), opts.aHigh), [1, N, nCtrl]);
    end
    A = A0;
    omega = opts.omega0 .* ones(1, N);   % 細胞ごとの緩和係数（前回の値を引き継げる）

    [Xk, J] = forward_pass(x0, A, bk, re, opts, spc);
    hist = struct('J', sum(J), 'pmpRes', NaN, 'nAccepted', N);
    fprintf('[FBSM] sweep %2d   総コスト %.6g\n', 0, sum(J));

    for sweep = 1:opts.maxSweeps
        P = backward_pass(Xk, A, bk, re, opts, spc);
        Anew = update_controls(Xk, P, bk, re, opts);
        % PMP残差: 今の制御が、自分の共状態に対する argmin H からどれだけ離れているか
        pmpRes = norm(Anew(:) - A(:)) / max(norm(A(:)), 1e-8);

        Atry = A + reshape(omega, 1, N) .* (Anew - A);
        [Xtry, Jtry] = forward_pass(x0, Atry, bk, re, opts, spc);
        acc = Jtry < J;
        Jold = J;
        A(:, acc, :) = Atry(:, acc, :);
        Xk(:, acc, :) = Xtry(:, acc, :);
        J(acc) = Jtry(acc);
        % 受け入れた細胞は ω をそのまま使い続け（倍にすると次で失敗しやすい）、
        % 失敗した細胞だけ ω を半分にする
        omega(~acc) = omega(~acc) / 2;

        relGain = (sum(Jold) - sum(J)) / max(abs(sum(Jold)), 1e-12);
        hist(end+1) = struct('J', sum(J), 'pmpRes', pmpRes, 'nAccepted', nnz(acc)); %#ok<AGROW>
        fprintf('[FBSM] sweep %2d   総コスト %.6g   PMP残差 %.3g   改善した細胞 %d/%d   相対改善 %.3g\n', ...
            sweep, sum(J), pmpRes, nnz(acc), N, relGain);

        % 改善した細胞があって、その改善がごく小さくなったら収束。改善した細胞が無いときは
        % ω を半分にしてやり直す（ω がすべて十分小さくなっても改善しなければ打ち切り）
        if (nnz(acc) > 0 && relGain < opts.tolRel && sweep >= 2) || all(omega < 1e-3)
            break;
        end
    end

    out = struct('J', J, 'Xk', Xk, 'hist', hist, 'bk', bk, 'tk', tk, 'omega', omega);
    if nargout > 1
        out.P = backward_pass(Xk, A, bk, re, opts, spc);
    end
end

% =====================================================================
function opts = fill_defaults(opts)
    d = struct('dt', 1, 'dtCtrl', 10, 'maxSweeps', 20, 'tolRel', 1e-4, 'omega0', 0.25);
    fn = fieldnames(d);
    for i = 1:numel(fn)
        if ~isfield(opts, fn{i}) || isempty(opts.(fn{i})), opts.(fn{i}) = d.(fn{i}); end
    end
end

function [Xk, J] = forward_pass(x0, A, bk, re, opts, spc)
%   制御 A（区間ごとに一定）と所与の b で前向きに計算し、区間の始点の状態とコストを返す
    [~, N, nCtrl] = size(A);
    Xk = zeros(6, N, nCtrl + 1);
    J = zeros(1, N);
    x = x0;
    for k = 1:nCtrl
        a = A(:,:,k);
        Xk(:,:,k) = x;
        for s = 1:spc
            J = J + opts.dt .* running_cost(x, a, opts);
            x = max(x + liver_cell_drift_limited(x, a, bk(:,k), re, max(opts.tauFast, opts.dt)) .* opts.dt, 0);
        end
    end
    Xk(:,:,end) = x;
    J = J + terminal_cost(x, opts);
end

function P = backward_pass(Xk, A, bk, re, opts, spc)
%   随伴方程式を後ろ向きに解き、区間の始点の共状態を返す。
%   区間の中の状態は、始点 Xk から前向きに計算し直す（全ステップを保存しないため）
    [~, N, nCtrl] = size(A);
    tau = max(opts.tauFast, opts.dt);
    h = 1e-3 .* opts.scaleX;
    E = kron(eye(6), ones(1, N));
    hE = h .* E;
    P = zeros(6, N, nCtrl + 1);
    p = terminal_grad(Xk(:,:,end), opts);
    P(:,:,end) = p;
    xs = zeros(6, N, spc);
    for k = nCtrl:-1:1
        a = A(:,:,k);
        x = Xk(:,:,k);
        for s = 1:spc
            xs(:,:,s) = x;
            x = max(x + liver_cell_drift_limited(x, a, bk(:,k), re, tau) .* opts.dt, 0);
        end
        aa = repmat(a, 1, 7);
        for s = spc:-1:1
            x = xs(:,:,s);
            F = liver_cell_drift_limited([x, repmat(x, 1, 6) + hE], aa, bk(:,k), re, tau);
            f0 = F(:, 1:N);
            JTp = zeros(6, N);
            for i = 1:6
                JTp(i,:) = sum(p .* (F(:, N*i + (1:N)) - f0), 1) ./ h(i);
            end
            p = p + opts.dt .* (running_cost_grad(x, a, opts) + JTp);
        end
        P(:,:,k) = p;
    end
end

function Anew = update_controls(Xk, P, bk, re, opts)
%   各区間の始点で a = argmin_a H(x, a, p)。区間をまとめて optimal_control に渡す
    [~, N, nK1] = size(Xk);
    nCtrl = nK1 - 1;
    Anew = zeros(6, N, nCtrl);
    chunk = 50;
    for k0 = 1:chunk:nCtrl
        ks = k0:min(k0 + chunk - 1, nCtrl);
        x = reshape(Xk(:,:,ks), 6, []);
        p = reshape(P(:,:,ks), 6, []);
        b = kron(bk(:, ks), ones(1, N));
        Anew(:,:,ks) = reshape(optimal_control(x, b, p, re, opts), 6, N, numel(ks));
    end
end

function L = running_cost(x, a, opts)
%   hjb_liver_pinn.m の running_cost と同じ
    coaExcess = max((x(4,:) - opts.targetCoA) ./ opts.targetCoA, 0);
    bonus = opts.w_coaPush .* coaExcess .* (a(5,:) ./ opts.aHigh(5) + a(6,:) ./ opts.aHigh(6));
    L = opts.w_coa .* (x(4,:) ./ opts.targetCoA - 1).^2 ...
      + opts.w_x05 .* (x(1,:) ./ opts.targetX05 - 1).^2 ...
      + opts.w_ctrl .* sum((a - opts.aRef).^2, 1) ...
      - bonus;
end

function g = terminal_cost(x, opts)
    g = opts.w_coa .* (x(4,:) ./ opts.targetCoA - 1).^2 ...
      + opts.w_x05 .* (x(1,:) ./ opts.targetX05 - 1).^2;
end

function G = terminal_grad(x, opts)
    G = zeros(size(x));
    G(1,:) = 2 .* opts.w_x05 .* (x(1,:) ./ opts.targetX05 - 1) ./ opts.targetX05;
    G(4,:) = 2 .* opts.w_coa .* (x(4,:) ./ opts.targetCoA - 1) ./ opts.targetCoA;
end

function Lx = running_cost_grad(x, a, opts)
    Lx = terminal_grad(x, opts);
    over = x(4,:) > opts.targetCoA;
    Lx(4,:) = Lx(4,:) - opts.w_coaPush ./ opts.targetCoA .* over ...
        .* (a(5,:) ./ opts.aHigh(5) + a(6,:) ./ opts.aHigh(6));
end
