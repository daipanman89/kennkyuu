function a = optimal_control(x, b, p, re, opts)
%OPTIMAL_CONTROL  価値関数の勾配 p=∇V から最適制御  a* = argmin_a H(x,a,p)  を求める。
%
%   a = optimal_control(x, b, p, re, opts)
%
%   H(x,a,p) = L(x,a) + p・f(x,a,b)
%     L : hjb_liver_pinn.m の running_cost と同じ走行コスト
%         （状態コスト + w_ctrl*|a-aRef|^2 - coaPushBonus）
%     f : liver_cell_drift.m の細胞内ドリフト
%
%   方策ネットを学習させる代わりに、H の構造を使って直接最小化する:
%     ・a2 (act_MTP) は r10 にしか入らないので、他と独立に1次元グリッドで最小化
%     ・a1 を固定すると、ゲイン a3..a6 は H の中で互いに独立な2次関数
%         w_ctrl*(a_i - aRef_i)^2 + c_i*a_i
%       になる（f が各ゲインについて線形で、ゲイン同士の積の項が無いため）。
%       よって a_i = clip(aRef_i - c_i/(2*w_ctrl), aLow_i, aHigh_i) が厳密な最小解。
%     ・a1 は act_Ins/act_Gca を通して非線形なので、1次元グリッドで探索し、
%       各候補でゲインを上の閉形式で最適化した中から H 最小のものを選ぶ。
%
%   入力
%     x : 6×N  細胞内状態（double）
%     b : 4×1 または 4×N  血中状態
%     p : 6×N  ∂V/∂x
%     re, opts : hjb_liver_pinn.m と同じ（opts は hjbInfo.opts をそのまま渡せばよい）
%   出力
%     a : 6×N  最適制御

    N = size(x, 2);
    if size(b, 2) == 1
        b = repmat(b, 1, N);
    end
    aLow = opts.aLow; aHigh = opts.aHigh; aRef = opts.aRef; w = opts.w_ctrl;
    nG = opts.nGridA;

    % --- a2: 独立に1次元グリッド探索 ---
    g2 = control_grid(aLow(2), aHigh(2), nG);
    n2 = numel(g2);
    A = repmat(aRef, 1, N*n2);
    A(3:6,:) = 0;
    A(2,:) = kron(g2, ones(1,N));
    H2 = phi(repmat(x,1,n2), A, repmat(b,1,n2), repmat(p,1,n2), re, opts) ...
       + w .* (A(2,:) - aRef(2)).^2;
    [~, j2] = min(reshape(H2, N, n2), [], 2);
    a2 = g2(j2(:)');                                   % 1×N

    % --- a1: グリッド探索（各候補でゲイン a3..a6 は閉形式） ---
    g1 = control_grid(aLow(1), aHigh(1), nG);
    n1 = numel(g1);
    xR = repmat(x, 1, n1); bR = repmat(b, 1, n1); pR = repmat(p, 1, n1);
    A0 = zeros(6, N*n1);
    A0(1,:) = kron(g1, ones(1,N));
    A0(2,:) = repmat(a2, 1, n1);
    phi0 = phi(xR, A0, bR, pR, re, opts);              % ゲイン全て0のときの p・f - bonus

    gains = zeros(4, N*n1);
    Hgain = zeros(1, N*n1);
    for k = 1:4
        i = k + 2;
        Ai = A0; Ai(i,:) = 1;
        c = phi(xR, Ai, bR, pR, re, opts) - phi0;      % ゲイン i の1次係数
        ai = min(max(aRef(i) - c ./ (2*w), aLow(i)), aHigh(i));
        gains(k,:) = ai;
        Hgain = Hgain + c .* ai + w .* (ai - aRef(i)).^2;
    end
    H1 = phi0 + Hgain + w .* (A0(1,:) - aRef(1)).^2;
    [~, j1] = min(reshape(H1, N, n1), [], 2);
    col = (j1(:)' - 1) * N + (1:N);                    % 各細胞で選ばれた列

    a = [g1(j1(:)'); a2; gains(:, col)];
end

% =====================================================================
function v = phi(x, a, b, p, re, opts)
%   H のうち制御に依存する非2次部分: p・f(x,a,b) - coaPushBonus
    dxdt = liver_cell_drift(x, a, b, re);
    coaExcess = max((x(4,:) - opts.targetCoA) ./ opts.targetCoA, 0);
    bonus = opts.w_coaPush .* coaExcess .* (a(5,:)./opts.aHigh(5) + a(6,:)./opts.aHigh(6));
    v = sum(p .* dxdt, 1) - bonus;
end

function g = control_grid(lo, hi, n)
%   act = a/(Ka+a) は a が小さいほど感度が高いので、0付近を細かく取る
    g = unique([lo, lo + logspace(-2, log10(hi - lo), n - 1)]);
end
