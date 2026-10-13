function [dxdt, bflux] = liver_cell_drift_limited(x, a, b, re, tauFast)
%LIVER_CELL_DRIFT_LIMITED  liver_cell_drift.m に、速い反応 r13 の上限を付けたもの。
%
%   [dxdt, bflux] = liver_cell_drift_limited(x, a, b, re, tauFast)
%
%   r13 = re(47)*x07*x09*act_Ins*10 は2物質の積に比例する反応で、re(47)=0.01 だと
%   インスリンが少しでも入ると 1e5〜1e6 /秒 という、他の反応(約0.1/秒)より桁違いに
%   速い反応になる。陽的Euler法(dt=1秒)でそのまま進めると、1ステップで x07, x09 を
%   何百倍も引き過ぎてしまい、max(x,0) で0に切り詰めた分だけ x10 が「無から」増える
%   （質量保存が壊れる。例: x07=2000, x09=7600 から x10 が 138万 増える）。
%
%   そこで r13 を「tauFast 秒で手元の x07, x09 を使い切る速さ」で頭打ちにする:
%       r13_eff = min(r13, min(x07, x09/c3) / tauFast)
%   x07 + x09*... → x10 の量論（x07 1, x09 c3 → x10 1）は保ったまま差し替えるので、
%   質量は保存される。r13 は血中フラックス bflux には入らないので bflux は変わらない。
%   tauFast は FP の時間刻み dt 以上にすること（既定 1秒 = dt）。
%   HJB・FP・最適制御の計算すべてでこの関数を使い、同じダイナミクスを共有する。

    [dxdt, bflux, rr] = liver_cell_drift(x, a, b, re);
    c3 = 3; % liver_cell_drift.m と同じ量論係数

    r13cap = min(max(x(3,:), 0), max(x(5,:), 0) ./ c3) ./ tauFast;
    delta  = min(rr.r13, r13cap) - rr.r13;   % <= 0（上限を超えた分を差し引く）

    dxdt(3,:) = dxdt(3,:) - delta;           % dx07 = ... - r13
    dxdt(5,:) = dxdt(5,:) - c3 .* delta;     % dx09 = ... - c3*r13
    dxdt(6,:) = dxdt(6,:) + delta;           % dx10 = ... + r13
end
