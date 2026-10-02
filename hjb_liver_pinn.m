function [valueNet, policyNet, info] = hjb_liver_pinn(bTraj, opts, refTraj)
    if nargin < 2, opts = struct(); end
    opts = parse_opts(opts);
    if nargin < 3, refTraj = []; end

    if nargin < 1 || isempty(bTraj)
        bTraj = struct('t', [0, opts.T], 'val', repmat(opts.bInit, 1, 2));
    end

    if isfile('re_params.mat')
        S = load('re_params.mat','re');
        re = S.re;
    else
        error('re_params.mat が見つかりません。');
    end

    %% ---- 問題設定 ----
    scaleX   = opts.scaleX;
    sigma    = opts.sigmaFrac .* scaleX;
    aLow     = opts.aLow;
    aHigh    = opts.aHigh;
    T        = opts.T;
    targetCoA= opts.targetCoA;
    targetX05= opts.targetX05;
    w_coa    = opts.w_coa;
    w_x05    = opts.w_x05;
    w_coaPush= opts.w_coaPush;
    w_ctrl   = opts.w_ctrl;

    domLow  = opts.domLowFrac  .* scaleX;
    domHigh = opts.domHighFrac .* scaleX;

    %% ---- ネットワーク構築 ----
    valueNet  = build_value_net(7, opts.hiddenUnits);
    policyNet = build_policy_net(7, 6, aLow, aHigh, opts.hiddenUnits);

    avgV=[]; avgSqV=[]; avgP=[]; avgSqP=[];
    lossHistory = zeros(opts.numIters,1);
    lossPDEHistory = zeros(opts.numIters,1);
    lossTermHistory = zeros(opts.numIters,1);
    lossHamHistory = zeros(opts.numIters,1);
    h = opts.fdStep .* scaleX; 

    bt = bTraj.t; bv = bTraj.val; 

    % --- refTrajの準備 ---
    useRef = ~isempty(refTraj) && isfield(refTraj,'xAll') && ~isempty(refTraj.xAll);
    if useRef
        refT = refTraj.t(:)';                 
        [~, refN, refL] = size(refTraj.xAll);  
        refXFlat = reshape(refTraj.xAll, 6, refN*refL); 
        nBatchRef = round(opts.batchSize * opts.refFrac);
        nBatchUni = opts.batchSize - nBatchRef;
        jitter = opts.refJitterFrac .* scaleX; 
        fprintf('[HJB] refTraj使用: 1バッチあたり一様%d点 + 実軌道近傍%d点\n', nBatchUni, nBatchRef);
    else
        nBatchUni = opts.batchSize;
        nBatchRef = 0;
    end

    for iter = 1:opts.numIters
        windowFrac = min(1, iter / max(1, opts.numIters * opts.causalWarmupFrac));
        tMin = T * (1 - windowFrac);

        % --- 動的重み付け（Dynamic Weighting）---
        % 学習の前半(50%)をかけて、終端Lossの重みを初期値から最終値へ徐々に引き上げる
        weightProgress = min(1, iter / (opts.numIters * 0.5));
        currentTermWeight = opts.termWeightInit + weightProgress * (opts.termWeightFinal - opts.termWeightInit);

        u  = rand(6, nBatchUni);
        XbUni = domLow + u .* (domHigh - domLow);
        tbUni = tMin + rand(1, nBatchUni) .* (T - tMin);

        if nBatchRef > 0
            validIdx = find(refT >= tMin);
            if isempty(validIdx), validIdx = 1:refL; end 
            timeIdx = validIdx(randi(numel(validIdx), 1, nBatchRef));
            cellIdx = randi(refN, 1, nBatchRef);
            idx = (timeIdx-1)*refN + cellIdx;
            XbRef = refXFlat(:, idx) + jitter .* randn(6, nBatchRef);
            XbRef = max(XbRef, 0); 
            tbRef = refT(timeIdx);
            Xb = [XbUni, XbRef];
            tb = [tbUni, tbRef];
        else
            Xb = XbUni;
            tb = tbUni;
        end

        uT  = rand(6, opts.batchSizeTerminal);
        XbT = domLow + uT .* (domHigh - domLow);

        X   = dlarray([Xb; tb], 'CB');            
        XT  = dlarray([XbT; T*ones(1,opts.batchSizeTerminal)], 'CB');

        bAtT = interp1(bt, bv', tb, 'linear', 'extrap')'; 

        [gradTh, gradPh, lossVal, lPDE, lTerm, lHam] = dlfeval(@modelLoss, valueNet, policyNet, X, XT, ...
            re, sigma, h, scaleX, T, bAtT, targetCoA, targetX05, w_coa, w_x05, w_coaPush, w_ctrl, ...
            currentTermWeight, aHigh, aLow, opts.w_satPenalty);

        % --- Valueネットワークの事前学習（Pre-training）と学習率減衰 ---
        decaySteps = floor(iter / opts.lrDecayEvery);
        lrV = opts.lrValue  * (opts.lrDecayFactor ^ decaySteps);
        lrP = opts.lrPolicy * (opts.lrDecayFactor ^ decaySteps);

        % Value側は常に更新
        [valueNet, avgV, avgSqV] = adamupdate(valueNet, gradTh, avgV, avgSqV, iter, lrV);

        % Policy側は pretrainIters を超えてから更新を開始する（Adamの内部カウンタもリセット）
        if iter > opts.pretrainIters
            [policyNet, avgP, avgSqP] = adamupdate(policyNet, gradPh, avgP, avgSqP, iter - opts.pretrainIters, lrP);
        else
            lrP = 0; % 表示用
        end

        lossHistory(iter) = double(lossVal);
        lossPDEHistory(iter) = double(lPDE);
        lossTermHistory(iter) = double(lTerm);
        lossHamHistory(iter) = double(lHam);

        if mod(iter, opts.printEvery) == 0 || iter == 1
            fprintf('[HJB] iter %6d / %6d   V-loss = %.6g (PDE=%.6g, 終端=%.6g)   H(policy) = %.6g   termW=%.2g   lrP=%.2g\n', ...
                iter, opts.numIters, lossHistory(iter), lossPDEHistory(iter), lossTermHistory(iter), ...
                lossHamHistory(iter), currentTermWeight, lrP);
        end
    end

    info = struct('lossHistory', lossHistory, 'lossPDEHistory', lossPDEHistory, ...
        'lossTermHistory', lossTermHistory, 'lossHamHistory', lossHamHistory, 'opts', opts, 'bTraj', bTraj);
    save('hjb_solution.mat', 'valueNet', 'policyNet', 'info', 'bTraj', 'scaleX', 'T');
    fprintf('[HJB] 学習完了。hjb_solution.mat に保存しました。\n');
end

% =====================================================================
function [gradTh, gradPh, lossVal, lossPDE, lossTerm, lossHam] = modelLoss(valueNet, policyNet, X, XT, ...
    re, sigma, h, scaleX, T, bAtT, targetCoA, targetX05, w_coa, w_x05, w_coaPush, w_ctrl, currentTermWeight, aHigh, aLow, w_satPenalty)

    xRaw = X(1:6,:);
    Xn   = [xRaw ./ scaleX; X(7,:) ./ T];
    x05  = xRaw(1,:);
    x08  = xRaw(4,:);

    aRaw   = forward(policyNet, Xn);              
    aFixed = dlarray(extractdata(aRaw));          

    V0 = forward(valueNet, Xn);                   

    gradAll = dlgradient(sum(V0,'all'), X, 'EnableHigherDerivatives', true); 
    gradX = gradAll(1:6,:);
    gradT = gradAll(7,:);
    gradXFixed = dlarray(extractdata(gradX));     

    dxdt_a = liver_cell_drift(xRaw, aRaw, bAtT, re); 

    a5norm = aRaw(5,:) ./ aHigh(5);
    a6norm = aRaw(6,:) ./ aHigh(6);
    coaExcess = max((x08 - targetCoA) ./ targetCoA, 0);
    coaPushBonus = w_coaPush .* coaExcess .* (a5norm + a6norm); 
                                                                  
    runCostA = w_coa .* (x08./targetCoA - 1).^2 ...
             + w_x05 .* (x05./targetX05 - 1).^2 ...
             + w_ctrl .* sum(aRaw.^2, 1) ...
             - coaPushBonus;

    tanhOut = (2.*aRaw - aHigh - aLow) ./ (aHigh - aLow); 
    satPenalty = w_satPenalty .* sum(-log(max(1 - tanhOut.^2, 1e-8)), 1);

    Hamiltonian = runCostA + satPenalty + sum(gradXFixed .* dxdt_a, 1); 
    lossHam = mean(Hamiltonian, 'all');

    dxdt_v = liver_cell_drift(xRaw, aFixed, bAtT, re); 

    runCostV = w_coa .* (x08./targetCoA - 1).^2 ...
             + w_x05 .* (x05./targetX05 - 1).^2 ...
             + w_ctrl .* sum(aFixed.^2, 1); 

    driftTermV = sum(gradX .* dxdt_v, 1);

    lap = dlarray(zeros(1, size(X,2), 'like', extractdata(V0)));
    for i = 1:6
        Xp = X; Xp(i,:) = Xp(i,:) + h(i);
        Xm = X; Xm(i,:) = Xm(i,:) - h(i);
        Vp = forward(valueNet, [Xp(1:6,:)./scaleX; Xp(7,:)./T]);
        Vm = forward(valueNet, [Xm(1:6,:)./scaleX; Xm(7,:)./T]);
        d2 = (Vp - 2*V0 + Vm) ./ (h(i)^2);
        lap = lap + (sigma(i)^2) .* d2;
    end

    residual = gradT + runCostV + driftTermV + 0.5 .* lap;
    lossPDE = mean(residual.^2, 'all');

    xRawT = XT(1:6,:);
    XnT   = [xRawT ./ scaleX; XT(7,:) ./ T];
    VT    = forward(valueNet, XnT);
    gT    = w_coa .* (xRawT(4,:)./targetCoA - 1).^2 ...
          + w_x05 .* (xRawT(1,:)./targetX05 - 1).^2;
    lossTerm = mean((VT - gT).^2, 'all');

    % 固定重み(terminalWeight)から動的重み(currentTermWeight)へ変更
    lossVal = lossPDE + currentTermWeight .* lossTerm; 

    gradTh = dlgradient(lossVal, valueNet.Learnables);  
    gradPh = dlgradient(lossHam, policyNet.Learnables); 
end

function net = build_value_net(inDim, hidden)
    layers = [
        featureInputLayer(inDim, 'Normalization','none', 'Name','xt')
        fullyConnectedLayer(hidden, 'Name','fc1')
        tanhLayer('Name','act1')
        fullyConnectedLayer(hidden, 'Name','fc2')
        tanhLayer('Name','act2')
        fullyConnectedLayer(1, 'Name','V')
    ];
    net = dlnetwork(layers);
end

function net = build_policy_net(inDim, outDim, aLow, aHigh, hidden)
    scale = (aHigh - aLow) / 2;
    bias  = (aHigh + aLow) / 2;
    layers = [
        featureInputLayer(inDim, 'Normalization','none', 'Name','xt')
        fullyConnectedLayer(hidden, 'Name','fc1')
        reluLayer('Name','act1')
        fullyConnectedLayer(hidden, 'Name','fc2')
        reluLayer('Name','act2')
        fullyConnectedLayer(outDim, 'Name','fc3')
        tanhLayer('Name','tanh_out')
        scalingLayer('Name','scale_out', 'Scale', scale, 'Bias', bias)
    ];
    net = dlnetwork(layers);
end

function opts = parse_opts(opts)
    targetB = [5000; 1000; 1000; 100]; 
    defaults = struct( ...
        'scaleX',        [5000;200000;1000;1000;1000;1000], ...
        'sigmaFrac',    0.05 / sqrt(24*3600), ...
        'aLow',          zeros(6,1), ...
        'aHigh',         [100;100;10;10;30;30], ... 
        'T',             24*3600, ...
        'bInit',         targetB + 1000, ...
        'targetX05',     5000 * (5/1.5), ...
        'targetCoA',     1000, ...
        'w_coa',         3.0 * (5/1.5), ... 
        'w_x05',         9.0 * (5/1.5), ...
        'w_coaPush',     8.0, ... 
        'w_ctrl',        1e-4, ...
        'w_satPenalty',  1.0, ... 
        'refFrac',       0.5, ...   
        'refJitterFrac', 0.02, ...  
        'domLowFrac',    0.2, ...
        'domHighFrac',   [4.0; 4.0; 4.0; 10.0; 4.0; 4.0], ...
        'hiddenUnits',   64, ... 
        'batchSize',     256, ...
        'batchSizeTerminal', 64, ...
        'pretrainIters', 15000, ...  % ★新規: 最初の15,000イテレーションはValueのみ学習
        'termWeightInit', 0.1, ...   % ★新規: 学習序盤は終端Lossの重みを0.1にしてPDEに集中させる
        'termWeightFinal', 10.0, ... % ★新規: 学習後半に向けて重みを10.0まで引き上げる
        'numIters',      60000, ...
        'lrValue',       1e-3, ...
        'lrPolicy',      1e-4, ...
        'lrDecayEvery',  10000, ... 
        'lrDecayFactor', 0.5, ...   
        'causalWarmupFrac', 0.5, ... 
        'fdStep',        0.01, ...
        'printEvery',    500 ...
    );
    fn = fieldnames(defaults);
    for k = 1:numel(fn)
        if ~isfield(opts, fn{k}) || isempty(opts.(fn{k}))
            opts.(fn{k}) = defaults.(fn{k});
        end
    end
end