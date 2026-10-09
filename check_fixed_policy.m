%CHECK_FIXED_POLICY  制御 a を一定値に固定して、1細胞＋血中のモデルを
%                    ノイズなし（決定論的）で24時間シミュレーションする診断スクリプト。
%
%   学習した方策とは無関係に「この制御なら x05 と coa を保てるか」を確かめるためのもの。
%   ダイナミクスは liver_cell_drift.m をそのまま使い、血中側は fp_liver_particles.m と
%   同じ式（r45, r46）で更新する。N=1細胞・ノイズなしなので、平均場極限の挙動に対応する。

clearvars; clc;

S = load('re_params.mat', 're');
re = S.re;

T  = 24*3600;
dt = 1;
nSteps = round(T/dt);

x0 = [5000; 200000; 1000; 1000; 1000; 1000]; % fp_liver_particles.m の初期分布の中心(scaleX)
b0 = [5000; 1000; 1000; 100] + 1000;         % 目標値+1000（run_mfg_picard.m と同じ）

% 比較する固定制御 [a1;a2;a3;a4;a5;a6]（各列が1ケース）
aCases = [ 0  10  0  0  0  0;   % グルカゴン最大、coaの流入・排出なし
           3  10  5  0  0  0;   % やや グルカゴン寄り + グルコース由来のcoa流入
          10  10  2  0  0  0;   % 中立（act_Ins = act_Gca = 0.5）
           0  10  0  0  0 30 ]';% 学習した方策の序盤に近い（a6=最大でcoaを排出）
caseNames = {'a1=0 (グルカゴン最大)', 'a1=3, a3=5', 'a1=10 (中立), a3=2', 'a1=0, a6=30 (排出最大)'};

c3 = 3;
r45fun = @(x02) re(45) .* x02 ./ (re(46) + x02);
r46fun = @(x01) re(9)  .* x01 ./ (re(10) + x01);

logEvery = 60;
nLog = floor(nSteps/logEvery) + 1;
nCase = size(aCases, 2);
tLog = (0:nLog-1) * logEvery * dt;
xLog = zeros(6, nLog, nCase);
bLog = zeros(4, nLog, nCase);

for c = 1:nCase
    a = aCases(:,c);
    x = x0; b = b0;
    xLog(:,1,c) = x; bLog(:,1,c) = b;
    li = 1;
    for step = 1:nSteps
        [dxdt, bflux] = liver_cell_drift(x, a, b, re);
        dGLU  = bflux(1) - r46fun(b(1));
        dVLDL = bflux(2) - r45fun(b(2));
        dFFA  = bflux(3) + c3*r45fun(b(2));
        dGLR  = bflux(4) + r45fun(b(2));
        x = max(x + dxdt .* dt, 0);
        b = max(b + [dGLU; dVLDL; dFFA; dGLR] .* dt, 0);
        if mod(step, logEvery) == 0
            li = li + 1;
            xLog(:,li,c) = x; bLog(:,li,c) = b;
        end
    end
    fprintf('%-24s : x05 %6.0f -> %6.0f,  x06 %7.0f -> %7.0f,  coa %5.0f -> %5.0f,  GLU %5.0f -> %5.0f\n', ...
        caseNames{c}, xLog(1,2,c), xLog(1,end,c), xLog(2,1,c), xLog(2,end,c), ...
        xLog(4,1,c), xLog(4,end,c), bLog(1,2,c), bLog(1,end,c));
end

%% ---- 図：x05, x06(グリコーゲン), coa, 血中GLU ----
targetX05 = 5000 * (5/1.5);
targetCoA = 1000;
th = tLog/3600;

figure('Name','固定制御での24時間シミュレーション（ノイズなし）');
subplot(2,2,1);
plot(th, squeeze(xLog(1,:,:)), 'LineWidth',1.3); hold on;
yline(targetX05, 'k--'); grid on;
xlabel('time [hour]'); ylabel('x05'); title('x05（肝臓内グルコース）');

subplot(2,2,2);
plot(th, squeeze(xLog(2,:,:)), 'LineWidth',1.3); grid on;
xlabel('time [hour]'); ylabel('x06'); title('x06（肝グリコーゲン）');

subplot(2,2,3);
plot(th, squeeze(xLog(4,:,:)), 'LineWidth',1.3); hold on;
yline(targetCoA, 'k--'); grid on;
xlabel('time [hour]'); ylabel('x08'); title('coa (x08)');

subplot(2,2,4);
plot(th, squeeze(bLog(1,:,:)), 'LineWidth',1.3); grid on;
xlabel('time [hour]'); ylabel('GLU'); title('血中GLU');
legend(caseNames, 'Location','best');
