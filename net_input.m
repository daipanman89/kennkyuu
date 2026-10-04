function Xn = net_input(x, t, scaleX, T)
%NET_INPUT  Value網/Actor網に入力する前に (x,t) を正規化する共通ヘルパー。
%
%   Xn = net_input(x, t, scaleX, T)
%
%   入力
%     x      : 6×N  細胞内状態（生スケール）
%     t      : 1×N  時刻[sec]（生スケール、0..T）。スカラーでもよい（自動展開）
%     scaleX : 6×1  x05..x10 の典型スケール
%     T      : スカラー  ホライズン[sec]
%
%   出力
%     Xn : 7×N  正規化済み入力（ネットワークにそのまま渡す）
%
%   hjb_liver_pinn.m と fp_liver_particles.m の両方から呼び、
%   学習時と推論（母集団シミュレーション）時で正規化がずれないようにする。

    N = size(x,2);
    if isscalar(t) || size(t,2) == 1
        t = repmat(t, 1, N);
    end
    Xn = [x ./ scaleX; t ./ T];
end
