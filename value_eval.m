function V = value_eval(valueNet, X, cfg)
%VALUE_EVAL  生の [x;t] (7×B) から V(t,x) を評価する。
%   hjb_liver_pinn.m（学習）と argmin_policy.m（FPでの制御計算）で共通に使う。
%   cfg.hard=true のとき  V(t,x) = g(x) + (T-t) * NN(x/scaleX, t/T)
%   とし、終端条件 V(T,x)=g(x) を構造的に満たす。
%   ランニングコストは [1/sec] 単位でT=86400秒ぶん積分されるため、V(0,x) は
%   g(x) より何桁も大きくなる。NN にはその「1秒あたりの平均コスト」(O(1)~O(100))
%   だけを学習させ、桁の大きさは (T-t) が受け持つ。
    Xn = [X(1:6,:) ./ cfg.scaleX; X(7,:) ./ cfg.T];
    NN = forward(valueNet, Xn);
    if cfg.hard
        V = terminal_cost(X(1:6,:), cfg) + (cfg.T - X(7,:)) .* NN;
    else
        V = NN;
    end
end

% =====================================================================
function g = terminal_cost(xRaw, cfg)
    g = cfg.w_coa .* (xRaw(4,:)./cfg.targetCoA - 1).^2 ...
      + cfg.w_x05 .* (xRaw(1,:)./cfg.targetX05 - 1).^2;
end
