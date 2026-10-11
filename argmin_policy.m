function a = argmin_policy(valueNet, t, x, b, re, opts)
%ARGMIN_POLICY  学習済みの価値関数 V から、各細胞の最適制御
%               a*(t,x) = argmin_a H(x, a, ∇V(t,x))  を計算する（FP側で使う）。
%
%   a = argmin_policy(valueNet, t, x, b, re, opts)
%
%   入力
%     valueNet : hjb_liver_pinn.m で学習した価値ネット
%     t        : スカラー  時刻[sec]
%     x        : 6×N  細胞内状態
%     b        : 4×1  血中状態
%     re       : 反応速度パラメータ
%     opts     : hjb_liver_pinn.m の opts（hjbInfo.opts をそのまま渡す）
%   出力
%     a : 6×N  最適制御
%
%   run_mfg_picard.m では  @(t,x,b) argmin_policy(valueNet, t, x, b, re, hjbInfo.opts)
%   として fp_liver_particles.m に渡す。

    N = size(x, 2);
    cfg = struct('hard', opts.hardTerminal, 'T', opts.T, 'scaleX', opts.scaleX, ...
        'targetCoA', opts.targetCoA, 'targetX05', opts.targetX05, ...
        'w_coa', opts.w_coa, 'w_x05', opts.w_x05);
    X = dlarray([x; repmat(t, 1, N)], 'CB');
    gradV = dlfeval(@value_gradient, valueNet, X, cfg);
    p = double(extractdata(gradV(1:6,:)));
    a = optimal_control(x, b, p, re, opts);
end

% =====================================================================
function gradV = value_gradient(valueNet, X, cfg)
    V = value_eval(valueNet, X, cfg);
    gradV = dlgradient(sum(V, 'all'), X);
end
