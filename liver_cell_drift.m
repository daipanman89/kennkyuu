function [dxdt, bflux, rr] = liver_cell_drift(x, a, b, re)
%LIVER_CELL_DRIFT  代表肝細胞1個の細胞内ダイナミクス（ドリフト項）。
%
%   [dxdt, bflux, rr] = liver_cell_drift(x, a, b, re)
%
%   model_single.m の肝臓ループ（num_L=1, m=1 相当）を、HJBソルバと
%   FPパーティクルシミュレーションの両方から再利用できる形に切り出したもの。
%   バッチ計算（dlarray, 6×N や 1×N など列方向がサンプル）にそのまま対応する。
%
%   入力
%     x  : 6×N   細胞内状態  [x05;x06;x07;x08;x09;x10]
%     a  : 6×N   制御（ゲイン） [aL1..aL6]  （model_single.m の aL と同じ並び）
%     b  : 4×N または 4×1  血中状態 [x01;x02;x03;x04] = [GLU;VLDL;FFA;GLR]
%          （4×1 を渡した場合は全サンプルで共通の血中値としてブロードキャストする）
%     re : 反応速度パラメータベクトル（model_single.m の re と同一のものを使うこと）
%
%   出力
%     dxdt  : 6×N   細胞内状態の時間微分（VoB/VoLスケーリング込み、model_single.mと同一）
%     bflux : 4×N   この1細胞をフルVoLの代表エージェント（num_L=1相当）として
%                    計算した場合の、血中側の式 f(1)..f(4) への寄与
%                    （model_single.m の "-r01 + r02*(VoL/VoB)" などと同一、
%                     VoL/VoBスケーリングは内蔵済み）。
%                    単一代表エージェント近似として使う場合はそのまま使えばよい。
%                    N細胞（100細胞など）で集団平均する場合は fp_liver_particles.m
%                    を参照：model_single.m 冒頭コメント「細胞数を増やすことは
%                    VoL → VoL/num_L にすること」を踏まえ、本関数の出力（フルVoL
%                    で計算したbflux）をN細胞で単純平均することで、1細胞あたり
%                    VoL/N を負担させて総和するのと等価な集計を行う。
%     rr    : struct  反応速度 r01..r17（デバッグ・可視化用）

    if size(b,2) == 1 && size(x,2) > 1
        b = repmat(b, 1, size(x,2));
    end

    VoB = 5; VoL = 1.5;
    c1 = 1; c2 = 1/8; c3 = 3; Ka = 10; % c1: 8から1に変更（ユーザー調整済み）

    x05 = x(1,:); x06 = x(2,:); x07 = x(3,:);
    x08 = x(4,:); x09 = x(5,:); x10 = x(6,:);

    x01 = b(1,:); x03 = b(3,:); x04 = b(4,:); % x02(VLDL) は肝臓細胞内式には未使用

    a1 = a(1,:); a2 = a(2,:); a3 = a(3,:);
    a4 = a(4,:); a5 = a(5,:); a6 = a(6,:);

    act_Ins = a1 ./ (Ka + a1);
    act_Gca = Ka ./ (Ka + a1);
    act_MTP = a2 ./ (Ka + a2);
    gain_r07 = a3;
    gain_r15 = a4;
    gain_r16 = a5;
    gain_r17 = a6;

    r01 = re(1)  .* x01                                 .* 10;
    r02 = re(1)  .* x05                                 .* 10;
    r03 = re(17) .* x05 ./ (re(18) + x05)                .* act_Ins .* 10;
    r04 = re(19) .* x06 ./ (re(20) + x06)                .* act_Gca .* 10;
    r05 = re(73) .* x05 ./ (re(74) + x05)                .* act_Ins .* 10;
    r06 = re(75) .* x07 ./ (re(76) + x07)                .* act_Gca .* 10;
    r07 = re(21) .* x05 ./ (re(22) + x05)                .* act_Ins .* gain_r07;

    r08 = re(33) .* x03                                 .* 10;
    r09 = re(33) .* x09                                 .* 10;
    r10 = re(71) .* x10 ./ (re(72) + x10)                .* act_MTP .* 10;

    r11 = re(59) .* x04                                 .* 10;
    r12 = re(59) .* x07                                 .* 10;

    r13 = re(47) .* x07 .* x09                           .* act_Ins .* 10;
    r14 = re(48) .* x10 ./ (re(49) + x10)                .* act_Gca .* 10;
    r15 = re(77) .* x09 ./ (re(78) + x09)                .* act_Gca .* gain_r15;
    r16 = re(79) .* x08 ./ (re(80) + x08)                .* act_Ins .* gain_r16;
    r17 = re(81) .* x08 ./ (re(82) + x08)                .* gain_r17;

    dx05 = r01.*(VoB/VoL) - r02 - r03 + r04 - r05 + r06 - r07;
    dx06 = r03 - r04;
    dx07 = r05 - r06 + r11.*(VoB/VoL) - r12 - r13 + r14;
    dx08 = r07 + c1.*r15 - r16 - r17;
    dx09 = r08.*(VoB/VoL) - r09 - c3.*r13 + c3.*r14 - r15 + c2.*r16;
    dx10 = -r10 + r13 - r14;

    dxdt = [dx05; dx06; dx07; dx08; dx09; dx10];

    bflux = [ -r01 + r02.*(VoL/VoB);
               r10.*(VoL/VoB);
              -r08 + r09.*(VoL/VoB);
              -r11 + r12.*(VoL/VoB) ];

    if nargout > 2
        rr = struct('r01',r01,'r02',r02,'r03',r03,'r04',r04,'r05',r05,'r06',r06, ...
                     'r07',r07,'r08',r08,'r09',r09,'r10',r10,'r11',r11,'r12',r12, ...
                     'r13',r13,'r14',r14,'r15',r15,'r16',r16,'r17',r17);
    end
end
