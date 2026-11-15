%% CL_MAC_ThreeProtocol_Improved.m
% Improved CL-MAC packet-level evaluation against the two manuscript baselines.
% Protocols compared: UAV-TDMA, UAV-EATDMA, and the proposed improved CL-MAC.
%
% Proposed CL-MAC uses a queue-aware cross-layer access rule:
%   p_i = pMax*(alpha*E_ratio + beta*LQ + delta*Q_norm),
%   alpha+beta+delta=1.
% It adds queue-age admission, collision-adaptive contention windows, and
% multi-stage mini-slot reservation resolution. Reservation collisions are
% resolved before data transmission; unresolved reservation collisions
% consume retry budget but do not incur a fictitious data-packet transmission.
% UAV-TDMA and UAV-EATDMA benchmark implementations remain unchanged.
%
% This is an extension of the manuscript's original two-factor probability
% equation. Update the manuscript equations and describe this change if these
% simulation results are reported.
%
% UAV-TDMA and UAV-EATDMA benchmark rules are kept unchanged.
%
% Metrics: PDR, throughput, packet loss, mean delay, delay CDF,
% generated/delivered/dropped/pending packets, energy, energy/delivered packet,
% residual energy and surviving nodes.
%
% IMPORTANT:
% This packet-level model extends the manuscript's energy model to assess
% delivery/QoS. The reservation/queue-age mechanism is an explicit enhancement
% and must be described in the manuscript if these results are reported.
% No performance values are forced. All protocols use common traffic/channel
% traces, and every scenario is retained, including cases where CL-MAC loses.
% The packet-level abstraction is not a full two-tier cluster-head simulator.
%
% MATLAB R2021b+ recommended. No toolboxes required.

clear; close all; clc;
rng(13112026,'twister');

%% 1. Common simulation configuration
G.N = 20; % 20 cluster-head senders (inter-cluster packet-level abstraction)
G.slots = 5000;
G.slot_s = 0.010;
G.packetBits = 500; % manuscript maximum packet size (bits)
G.maxRetries = 4;
G.maxQueue = 50;
G.initialEnergyJ = 2.0;
G.txEnergyJ = 6.0e-5;      % 6 mW * 10 ms (manuscript Table 1)
G.rxEnergyJ = 5.2e-5;      % 5.2 mW * 10 ms (manuscript Table 1)
G.idleEnergyJ = 1.5e-5;    % 1.5 mW * 10 ms (manuscript Table 1)
G.reservationOverhead_s = 2.0e-4; % added reservation mini-contention latency
G.requestEnergyJ = 6.0e-3*G.reservationOverhead_s;
G.grantRxEnergyJ = 5.2e-3*G.reservationOverhead_s;
G.deploymentLengthM = 100; % manuscript deployment length A (m)
G.uavAltitudeM = 50; % manuscript UAV altitude (m)
% Queue-aware cross-layer weights for this proposed algorithm revision.
% These values are a testable design choice, not calibrated optima.
G.alpha = 0.30; G.beta = 0.20; G.delta = 0.50;
G.pAdaptiveMax = 0.85;
G.replicates = 30;

% Protocol parameters. Keep baselines fixed and documented.
G.eatdmaAttemptProb = 0.10;
G.tdmaSchedule = 'round-robin';

% Proposed CL-MAC adaptive access bounds and coefficients.
C.CWmin = 4;
C.CWmax = 128;
C.ema = 0.08;
C.queueWeight = 0.40;
C.ageWeight = 0.30;
C.linkWeight = 0.20;
C.energyWeight = 0.10;
C.ageThreshold_s = 0.50;
C.resolutionRounds = 3;

protocols = {'UAV-TDMA','UAV-EATDMA','CL-MAC (Proposed)'};
nP = numel(protocols);

%% 2. Predeclared scenarios
% Arrival probability is per node per slot, before burst modulation.
S(1) = struct('name','Sparse_Bursty','arrivalProb',0.012, ...
    'bursty',true,'mobility',0.15,'channelLoss',0.04,'batteryHet',0.10);
S(2) = struct('name','Moderate_Load','arrivalProb',0.035, ...
    'bursty',false,'mobility',0.20,'channelLoss',0.06,'batteryHet',0.15);
S(3) = struct('name','High_Congestion','arrivalProb',0.075, ...
    'bursty',true,'mobility',0.20,'channelLoss',0.08,'batteryHet',0.15);
S(4) = struct('name','High_UAV_Mobility','arrivalProb',0.040, ...
    'bursty',false,'mobility',0.85,'channelLoss',0.10,'batteryHet',0.15);
S(5) = struct('name','Heterogeneous_Battery','arrivalProb',0.040, ...
    'bursty',false,'mobility',0.25,'channelLoss',0.06,'batteryHet',0.65);
S(6) = struct('name','Unreliable_Channel','arrivalProb',0.045, ...
    'bursty',false,'mobility',0.35,'channelLoss',0.25,'batteryHet',0.15);
S(7) = struct('name','Combined_Stress','arrivalProb',0.060, ...
    'bursty',true,'mobility',0.75,'channelLoss',0.18,'batteryHet',0.55);
nS = numel(S);

%% 3. Run all scenario experiments using common random numbers
rawRows = cell(0,15);
delayCDFsamples = cell(nS,nP);
for s = 1:nS
    fprintf('Scenario %d/%d: %s\n',s,nS,S(s).name);
    for rep = 1:G.replicates
        seed = 900000 + 1000*s + rep;
        T = makeTraces(G,S(s),seed);
        for p = 1:nP
            M = runProtocol(G,C,S(s),T,protocols{p});
            rawRows(end+1,:) = {S(s).name,protocols{p},rep, ...
                M.generated,M.delivered,M.dropped,M.pending,M.PDR, ...
                M.throughputMbps,M.lossRatio,M.meanDelay_s,M.energyJ, ...
                M.energyPerDeliveredJ,M.meanResidualJ,M.aliveNodes}; %#ok<SAGROW>
            if ~isempty(M.delaySamples)
                delayCDFsamples{s,p} = [delayCDFsamples{s,p},M.delaySamples]; %#ok<AGROW>
            end
        end
    end
end

rawNames = {'Scenario','Protocol','Replicate','Generated','Delivered', ...
    'Dropped','Pending','PDR','Throughput_Mbps','LossRatio','MeanDelay_s', ...
    'Energy_J','EnergyPerDelivered_J','MeanResidual_J','AliveNodes'};
Raw = cell2table(rawRows,'VariableNames',rawNames);

%% 4. Mean and standard deviation summary
metrics = {'Generated','Delivered','Dropped','Pending','PDR', ...
    'Throughput_Mbps','LossRatio','MeanDelay_s','Energy_J', ...
    'EnergyPerDelivered_J','MeanResidual_J','AliveNodes'};
sumRows = cell(0,2+2*numel(metrics));
for s=1:nS
    for p=1:nP
        ix=strcmp(Raw.Scenario,S(s).name) & strcmp(Raw.Protocol,protocols{p});
        rr=Raw(ix,:);
        row=cell(1,2+2*numel(metrics));
        row{1}=S(s).name; row{2}=protocols{p};
        for k=1:numel(metrics)
            v=rr.(metrics{k});
            row{2+2*k-1}=mean(v,'omitnan');
            row{2+2*k}=std(v,'omitnan');
        end
        sumRows(end+1,:)=row; %#ok<SAGROW>
    end
end
sumNames={'Scenario','Protocol'};
for k=1:numel(metrics)
    sumNames{end+1}=[metrics{k} '_Mean']; %#ok<SAGROW>
    sumNames{end+1}=[metrics{k} '_SD']; %#ok<SAGROW>
end
Summary=cell2table(sumRows,'VariableNames',sumNames);

disp(' ');
disp('================= THREE-PROTOCOL SCENARIO RESULTS (MEAN) =================');
disp(Summary(:,{'Scenario','Protocol','PDR_Mean','Throughput_Mbps_Mean', ...
    'LossRatio_Mean','MeanDelay_s_Mean','Generated_Mean','Delivered_Mean', ...
    'Dropped_Mean','Pending_Mean','EnergyPerDelivered_J_Mean','AliveNodes_Mean'}));

%% 5. Save tabular and native MATLAB results
outDir='CL_MAC_ThreeProtocol_OptimizedAccess_Results';
if ~exist(outDir,'dir'), mkdir(outDir); end
writetable(Raw,fullfile(outDir,'CLMAC_raw_replications.csv'));
writetable(Summary,fullfile(outDir,'CLMAC_scenario_summary_mean_sd.csv'));
save(fullfile(outDir,'CLMAC_three_protocol_results.mat'),'G','C','S','protocols','Raw', ...
    'Summary','delayCDFsamples');

%% 6. Publication-style scenario trend plots (line plots with SD)
plotScenarioLines(Summary,S,protocols,'PDR','Packet delivery ratio',outDir);
plotScenarioLines(Summary,S,protocols,'Throughput_Mbps','Throughput (Mbps)',outDir);
plotScenarioLines(Summary,S,protocols,'LossRatio','Packet loss ratio',outDir);
plotScenarioLines(Summary,S,protocols,'MeanDelay_s','Mean end-to-end delay (s)',outDir);
plotScenarioLines(Summary,S,protocols,'EnergyPerDelivered_J','Energy per delivered packet (J)',outDir);
plotScenarioLines(Summary,S,protocols,'AliveNodes','Number of surviving nodes',outDir);

%% 7. Delay distribution: empirical CDF, not just mean bars
f=figure('Color','w','Position',[100 100 1050 620]); hold on; grid on; box on;
for p=1:nP
    d=[];
    for s=1:nS, d=[d delayCDFsamples{s,p}]; end %#ok<AGROW>
    if ~isempty(d)
        d=sort(d);
        yy=(1:numel(d))/numel(d);
        plot(d,yy,'LineWidth',2,'DisplayName',protocols{p});
    end
end
xlabel('End-to-end delay (s)'); ylabel('Empirical cumulative probability');
title('End-to-end packet-delay distribution across all scenarios');
legend('Location','southeast'); set(gca,'FontSize',11);
exportgraphics(f,fullfile(outDir,'Delay_Empirical_CDF.png'),'Resolution',400);

%% 8. Generated, delivered, dropped and pending packet profiles
% Mean packet accounting for each scenario and protocol.
for s=1:nS
    vals=zeros(nP,4);
    for p=1:nP
        ix=strcmp(Summary.Scenario,S(s).name) & strcmp(Summary.Protocol,protocols{p});
        vals(p,1)=Summary.Generated_Mean(ix);
        vals(p,2)=Summary.Delivered_Mean(ix);
        vals(p,3)=Summary.Dropped_Mean(ix);
        vals(p,4)=Summary.Pending_Mean(ix);
    end
    f=figure('Color','w','Position',[100 100 1050 560]);
    plot(1:nP,vals(:,1),'-o','LineWidth',2,'MarkerSize',8); hold on;
    plot(1:nP,vals(:,2),'-s','LineWidth',2,'MarkerSize',8);
    plot(1:nP,vals(:,3),'-^','LineWidth',2,'MarkerSize',8);
    plot(1:nP,vals(:,4),'-d','LineWidth',2,'MarkerSize',8);
    grid on; box on; xlim([0.8 nP+0.2]);
    set(gca,'XTick',1:nP,'XTickLabel',protocols,'FontSize',10);
    ylabel('Mean packet count');
    title(['Packet accounting: ' strrep(S(s).name,'_',' ')]);
    legend('Generated','Delivered','Dropped','Pending','Location','best');
    exportgraphics(f,fullfile(outDir,['Packet_Accounting_' S(s).name '.png']), ...
        'Resolution',350);
end

%% 9. Load sweep: PDR, throughput, delay versus offered load
% Uses the same channel/mobility setting for all protocols at each load.
loadLevels=[0.008 0.015 0.025 0.040 0.055 0.070 0.085];
loadMobility=0.30; loadChannelLoss=0.08;
LoadPDR=nan(numel(loadLevels),nP);
LoadThr=LoadPDR; LoadDelay=LoadPDR; LoadLoss=LoadPDR;
loadRep=max(12,round(G.replicates/2));
for a=1:numel(loadLevels)
    sc=struct('name','LoadSweep','arrivalProb',loadLevels(a), ...
        'bursty',false,'mobility',loadMobility, ...
        'channelLoss',loadChannelLoss,'batteryHet',0.15);
    mm=cell(nP,1);
    for p=1:nP, mm{p}=zeros(loadRep,4); end
    for rep=1:loadRep
        T=makeTraces(G,sc,700000+a*100+rep);
        for p=1:nP
            M=runProtocol(G,C,sc,T,protocols{p});
            mm{p}(rep,:)=[M.PDR M.throughputMbps M.meanDelay_s M.lossRatio];
        end
    end
    for p=1:nP
        LoadPDR(a,p)=mean(mm{p}(:,1),'omitnan');
        LoadThr(a,p)=mean(mm{p}(:,2),'omitnan');
        LoadDelay(a,p)=mean(mm{p}(:,3),'omitnan');
        LoadLoss(a,p)=mean(mm{p}(:,4),'omitnan');
    end
end
f=figure('Color','w','Position',[100 100 1150 720]);
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
plotMetricLines(loadLevels,LoadPDR,protocols,'Offered packet-arrival probability','PDR','PDR versus offered load');
plotMetricLines(loadLevels,LoadThr,protocols,'Offered packet-arrival probability','Throughput (Mbps)','Throughput versus offered load');
plotMetricLines(loadLevels,LoadDelay,protocols,'Offered packet-arrival probability','Mean delay (s)','Delay versus offered load');
plotMetricLines(loadLevels,LoadLoss,protocols,'Offered packet-arrival probability','Packet loss ratio','Packet loss versus offered load');
sgtitle('Proposed CL-MAC and Manuscript Baseline Performance under Increasing Offered Load');
exportgraphics(f,fullfile(outDir,'Load_Sweep_Performance.png'),'Resolution',400);
loadNames={'ArrivalProbability'};
for p=1:nP
    tag=matlab.lang.makeValidName(strrep(protocols{p},'-','_'));
    loadNames{end+1}=['PDR_' tag]; %#ok<SAGROW>
end
for p=1:nP
    tag=matlab.lang.makeValidName(strrep(protocols{p},'-','_'));
    loadNames{end+1}=['Throughput_' tag]; %#ok<SAGROW>
end
for p=1:nP
    tag=matlab.lang.makeValidName(strrep(protocols{p},'-','_'));
    loadNames{end+1}=['Delay_' tag]; %#ok<SAGROW>
end
for p=1:nP
    tag=matlab.lang.makeValidName(strrep(protocols{p},'-','_'));
    loadNames{end+1}=['Loss_' tag]; %#ok<SAGROW>
end
LoadTable=array2table([loadLevels(:),LoadPDR,LoadThr,LoadDelay,LoadLoss], ...
    'VariableNames',loadNames);
writetable(LoadTable,fullfile(outDir,'Load_Sweep_Results.csv'));

%% 10. 3D surfaces: PDR, throughput, delay versus load and mobility
% These are actual simulation grid results, not interpolated target values.
surfaceLoads=[0.012 0.025 0.040 0.055 0.070];
surfaceMob=[0.10 0.30 0.50 0.70 0.90];
surfRep=max(10,round(G.replicates/3));
SurfPDR=nan(numel(surfaceMob),numel(surfaceLoads),nP);
SurfThr=SurfPDR; SurfDelay=SurfPDR; SurfLoss=SurfPDR;

for im=1:numel(surfaceMob)
    for il=1:numel(surfaceLoads)
        sc=struct('name','LoadMobilityGrid','arrivalProb',surfaceLoads(il), ...
            'bursty',false,'mobility',surfaceMob(im), ...
            'channelLoss',0.10,'batteryHet',0.20);
        accum=zeros(nP,4);
        for rep=1:surfRep
            T=makeTraces(G,sc,810000+im*10000+il*100+rep);
            for p=1:nP
                M=runProtocol(G,C,sc,T,protocols{p});
                accum(p,:)=accum(p,:)+[M.PDR M.throughputMbps M.meanDelay_s M.lossRatio];
            end
        end
        accum=accum/surfRep;
        for p=1:nP
            SurfPDR(im,il,p)=accum(p,1);
            SurfThr(im,il,p)=accum(p,2);
            SurfDelay(im,il,p)=accum(p,3);
            SurfLoss(im,il,p)=accum(p,4);
        end
    end
end
save(fullfile(outDir,'CLMAC_3D_surface_data.mat'),'surfaceLoads','surfaceMob', ...
    'SurfPDR','SurfThr','SurfDelay','SurfLoss');

plot3DProtocolSurfaces(surfaceLoads,surfaceMob,SurfPDR,protocols, ...
    'Packet delivery ratio','PDR','Surface_PDR_3D.png',outDir);
plot3DProtocolSurfaces(surfaceLoads,surfaceMob,SurfThr,protocols, ...
    'Throughput','Throughput (Mbps)','Surface_Throughput_3D.png',outDir);
plot3DProtocolSurfaces(surfaceLoads,surfaceMob,SurfDelay,protocols, ...
    'Mean end-to-end delay','Delay (s)','Surface_Delay_3D.png',outDir);
plot3DProtocolSurfaces(surfaceLoads,surfaceMob,SurfLoss,protocols, ...
    'Packet loss ratio','Loss ratio','Surface_PacketLoss_3D.png',outDir);

%% 11. 3D relative-performance surfaces (percentage difference)
% Positive values mean the named proposed CL-MAC is better for PDR/throughput
% and lower delay/loss than the named baseline. Negative values mean worse.
for variant=3
    variantName=protocols{variant};
    variantTag=matlab.lang.makeValidName(strrep(variantName,' ','_'));
    for baseline=1:2
        baseName=protocols{baseline};
        baseTag=matlab.lang.makeValidName(strrep(baseName,'-','_'));
        relPDR=100*(SurfPDR(:,:,variant)-SurfPDR(:,:,baseline))./ ...
            max(SurfPDR(:,:,baseline),eps);
        relThr=100*(SurfThr(:,:,variant)-SurfThr(:,:,baseline))./ ...
            max(SurfThr(:,:,baseline),eps);
        relDelay=100*(SurfDelay(:,:,baseline)-SurfDelay(:,:,variant))./ ...
            max(SurfDelay(:,:,baseline),eps);
        relLoss=100*(SurfLoss(:,:,baseline)-SurfLoss(:,:,variant))./ ...
            max(SurfLoss(:,:,baseline),eps);
        plotOneSurface(surfaceLoads,surfaceMob,relPDR, ...
            [variantName ' PDR advantage over ' baseName], ...
            'Relative PDR improvement (%)', ...
            ['Relative_PDR_' variantTag '_vs_' baseTag '.png'],outDir);
        plotOneSurface(surfaceLoads,surfaceMob,relThr, ...
            [variantName ' throughput advantage over ' baseName], ...
            'Relative throughput improvement (%)', ...
            ['Relative_Throughput_' variantTag '_vs_' baseTag '.png'],outDir);
        plotOneSurface(surfaceLoads,surfaceMob,relDelay, ...
            [variantName ' delay advantage over ' baseName], ...
            'Delay reduction (%)', ...
            ['Relative_Delay_' variantTag '_vs_' baseTag '.png'],outDir);
        plotOneSurface(surfaceLoads,surfaceMob,relLoss, ...
            [variantName ' packet-loss advantage over ' baseName], ...
            'Packet-loss reduction (%)', ...
            ['Relative_Loss_' variantTag '_vs_' baseTag '.png'],outDir);
    end
end

%% 12. Relative improvement of proposed CL-MAC over manuscript baselines
advRows=cell(0,10);
for s=1:nS
    ixCL=strcmp(Summary.Scenario,S(s).name) & ...
          strcmp(Summary.Protocol,'CL-MAC (Proposed)');
    clPDR=Summary.PDR_Mean(ixCL);
    clThr=Summary.Throughput_Mbps_Mean(ixCL);
    clDelay=Summary.MeanDelay_s_Mean(ixCL);
    clLoss=Summary.LossRatio_Mean(ixCL);
    clDelivered=Summary.Delivered_Mean(ixCL);
    for b=1:2
        ixB=strcmp(Summary.Scenario,S(s).name) & ...
             strcmp(Summary.Protocol,protocols{b});
        bPDR=Summary.PDR_Mean(ixB);
        bThr=Summary.Throughput_Mbps_Mean(ixB);
        bDelay=Summary.MeanDelay_s_Mean(ixB);
        bLoss=Summary.LossRatio_Mean(ixB);
        pdrGain=100*(clPDR-bPDR)/max(abs(bPDR),eps);
        thrGain=100*(clThr-bThr)/max(abs(bThr),eps);
        delayReduction=100*(bDelay-clDelay)/max(abs(bDelay),eps);
        if bLoss>eps
            lossReduction=100*(bLoss-clLoss)/bLoss;
        else
            lossReduction=NaN;
        end
        advRows(end+1,:)={S(s).name,protocols{b},pdrGain,thrGain, ...
            delayReduction,lossReduction,clDelivered, ...
            Summary.Delivered_Mean(ixB),Summary.Dropped_Mean(ixCL), ...
            Summary.Pending_Mean(ixCL)}; %#ok<SAGROW>
    end
end
AdvantageTable=cell2table(advRows,'VariableNames', ...
    {'Scenario','Baseline','PDR_Improvement_pct','Throughput_Improvement_pct', ...
     'Delay_Reduction_pct','PacketLoss_Reduction_pct', ...
     'CLMAC_Delivered_Mean','Baseline_Delivered_Mean', ...
     'CLMAC_Dropped_Mean','CLMAC_Pending_Mean'});
writetable(AdvantageTable,fullfile(outDir,'CLMAC_Improvement_vs_Baselines.csv'));

%% 13. Save complete sweep data and report diagnostics
versionInfo='Three-protocol evaluation: unchanged UAV-TDMA/UAV-EATDMA and CL-MAC with queue-aware multi-stage reservation';
metricList={'Generated','Delivered','Dropped','Pending','PDR','Throughput_Mbps', ...
    'LossRatio','MeanDelay_s','Energy_J','EnergyPerDelivered_J','MeanResidual_J','AliveNodes'};
save(fullfile(outDir,'CLMAC_complete_results.mat'),'G','C','S','protocols', ...
    'Raw','Summary','LoadTable','AdvantageTable','SurfPDR','SurfThr','SurfDelay','SurfLoss', ...
    'versionInfo','metricList');

fprintf('\n================ CL-MAC PROPOSED-METHOD DIAGNOSTICS ================\n');
for s=1:nS
    fprintf('\n%s\n',S(s).name);
    for p=1:nP
        ix=strcmp(Summary.Scenario,S(s).name) & strcmp(Summary.Protocol,protocols{p});
        fprintf('%-12s PDR=%6.2f%%  Throughput=%7.4f Mbps  Delay=%7.4f s  Loss=%6.2f%%\n', ...
            protocols{p},100*Summary.PDR_Mean(ix), ...
            Summary.Throughput_Mbps_Mean(ix),Summary.MeanDelay_s_Mean(ix), ...
            100*Summary.LossRatio_Mean(ix));
    end
end
fprintf('\nAll scenario and sweep outputs saved in: %s\n',outDir);
fprintf('Retain all scenarios, including any in which the proposed CL-MAC performs worse.\n');

%% ============================= FUNCTIONS ================================

function T=makeTraces(G,sc,seed)
% Common random traces: arrivals, link quality, access random numbers,
% and packet reception random numbers. All protocols see identical traces.
rng(seed,'twister');
N=G.N; K=G.slots;
T.arrivals=false(N,K);
T.linkQ=zeros(N,K);
T.mobilityFactor=zeros(N,K);
T.uAccess=rand(N,K);
T.uSuccess=rand(N,K);
T.uBackoff=rand(N,K); % retained for compatibility with saved trace conventions
T.uBackoffRound=rand(N,K,3);

burst=false;
for t=1:K
    if sc.bursty
        if burst
            if rand<0.05, burst=false; end
        else
            if rand<0.08, burst=true; end
        end
        burstFactor=1+2.0*double(burst);
    else
        burstFactor=1;
    end
    pa=min(0.95,sc.arrivalProb*burstFactor);
    T.arrivals(:,t)=rand(N,1)<pa;
end

% Mobility changes distance; link quality also includes slow fading.
horizontalDist=25+45*rand(N,1);
for t=1:K
    horizontalDist=horizontalDist+sc.mobility*2.2*randn(N,1);
    horizontalDist=min(sqrt(2)*G.deploymentLengthM,max(5,horizontalDist));
    dist3D=sqrt(horizontalDist.^2+G.uavAltitudeM^2);
    mobilityFactor=dist3D/G.deploymentLengthM;
    path=exp(-mobilityFactor); % manuscript exponential link-quality form
    fading=0.80+0.20*rand(N,1);
    impairment=max(0.05,1-sc.channelLoss*(0.5+rand(N,1)));
    T.linkQ(:,t)=min(0.995,max(0.05,path.*fading.*impairment));
    T.mobilityFactor(:,t)=mobilityFactor;
end
% Common initial battery factors.
T.initialBatteryFactor=0.65+0.35*(1-sc.batteryHet*rand(N,1));
end

function M=runProtocol(G,C,sc,T,protocol)
N=G.N; K=G.slots;
qTime=cell(N,1); qRetry=cell(N,1);
for n=1:N, qTime{n}=[]; qRetry{n}=[]; end
residual=G.initialEnergyJ*T.initialBatteryFactor;
generated=0; delivered=0; dropped=0; delays=[];
collisionEMA=0.10; activeEMA=1;
for t=1:K
    % Enqueue common packet arrivals.
    new=find(T.arrivals(:,t));
    for z=1:numel(new)
        n=new(z); generated=generated+1;
        if numel(qTime{n})<G.maxQueue
            qTime{n}(end+1)=t; %#ok<AGROW>
            qRetry{n}(end+1)=0; %#ok<AGROW>
        else
            dropped=dropped+1;
        end
    end

    qLen=cellfun(@numel,qTime);
    active=find(qLen>0 & residual>0);
    attempts=[];

    switch protocol
        case 'UAV-TDMA'
            % Fixed, collision-free, one scheduled opportunity per node cycle.
            n=mod(t-1,N)+1;
            if qLen(n)>0 && residual(n)>0, attempts=n; end

        case 'UAV-EATDMA'
            % Fixed-probability event-triggered random access.
            if ~isempty(active)
                pick=T.uAccess(active,t)<G.eatdmaAttemptProb;
                attempts=active(pick);
            end

        case 'CL-MAC (Proposed)'
            % Queue-aware cross-layer access probability:
            % p_i = pMax*(alpha*ER_i + beta*LQ_i + delta*Q_i).
            % ER, LQ and Q are bounded to [0,1]; alpha+beta+delta=1.
            reservationFailed=false;
            reservationRoundsUsed=0;

            if ~isempty(active)
                observed=numel(active);
                activeEMA=(1-C.ema)*activeEMA+C.ema*observed;
                congestion=min(1,activeEMA/N);
                meanM=mean(T.mobilityFactor(active,t));

                % Collision/load-adaptive contention window.
                cwLoad=min(1,0.60*collisionEMA+0.30*congestion+0.10*meanM);
                CWnow=round(C.CWmin+(C.CWmax-C.CWmin)*cwLoad);
                CWnow=max(C.CWmin,min(C.CWmax,CWnow));

                pAdaptive=zeros(numel(active),1);
                score=zeros(numel(active),1);
                ageNow=zeros(numel(active),1);

                for a=1:numel(active)
                    n=active(a);
                    eRatio=max(0,min(1,residual(n)/G.initialEnergyJ));
                    linkQuality=max(0,min(1,T.linkQ(n,t)));
                    queuePressure=min(1,qLen(n)/max(1,G.maxQueue));
                    headAge_s=(t-qTime{n}(1))*G.slot_s;
                    ageNow(a)=headAge_s;
                    ageUrgency=min(1,headAge_s/max(C.ageThreshold_s,eps));

                    pAdaptive(a)=G.pAdaptiveMax*( ...
                        G.alpha*eRatio + G.beta*linkQuality + ...
                        G.delta*queuePressure);
                    pAdaptive(a)=min(G.pAdaptiveMax,max(0,pAdaptive(a)));

                    % Priority is used for bounded contention backoff only.
                    score(a)=C.queueWeight*queuePressure + ...
                             C.ageWeight*ageUrgency + ...
                             C.linkWeight*linkQuality + ...
                             C.energyWeight*sqrt(eRatio);
                end

                % Probabilistic admission according to the cross-layer rule.
                eligibleMask=T.uAccess(active,t)<pAdaptive;
                candidates=active(eligibleMask);

                % Aging safeguard: the highest-priority aged node joins
                % contention if probabilistic admission selected nobody.
                aged=find(ageNow>=C.ageThreshold_s);
                if isempty(candidates) && ~isempty(aged)
                    [~,bestAged]=max(score(aged));
                    candidates=active(aged(bestAged));
                end

                if ~isempty(candidates)
                    % Resolve the earliest reservation contenders over up to
                    % three mini-slot stages. Only tied earliest contenders
                    % proceed to the next stage.
                    contenders=candidates(:);
                    reservationWinner=[];
                    for rr=1:C.resolutionRounds
                        reservationRoundsUsed=rr;

                        % Priority-adjusted, bounded random backoff.
                        contenderScore=zeros(numel(contenders),1);
                        for z=1:numel(contenders)
                            loc=find(active==contenders(z),1);
                            contenderScore(z)=score(loc);
                        end
                        effectiveCW=CWnow.*(1.20-0.45*contenderScore);
                        effectiveCW=max(C.CWmin, ...
                            min(C.CWmax,round(effectiveCW)));

                        backoff=floor(T.uBackoffRound(contenders,t,rr).*effectiveCW);

                        % Only nodes tied in the earliest mini-slot transmit
                        % reservation requests and incur request energy.
                        earliest=min(backoff);
                        tied=contenders(backoff==earliest);
                        residual(tied)=max(0,residual(tied)-G.requestEnergyJ);

                        if numel(tied)==1
                            reservationWinner=tied;
                            break;
                        end
                        contenders=tied;
                    end

                    if ~isempty(reservationWinner) && residual(reservationWinner)>0
                        % Charge the successful grant reception to the winner.
                        residual(reservationWinner)=max(0, ...
                            residual(reservationWinner)-G.grantRxEnergyJ);
                        if residual(reservationWinner)>0
                            attempts=reservationWinner;
                        else
                            reservationFailed=true;
                            attempts=reservationWinner;
                        end
                    else
                        % A reservation collision is not a data transmission.
                        % It consumes retry budget and reservation energy only.
                        reservationFailed=true;
                        attempts=contenders;
                    end
                end
            end

        otherwise
            error('Unknown protocol');
    end

    % Idle energy for all operational nodes.
    residual=max(0,residual-G.idleEnergyJ);
    nAtt=numel(attempts);

    % Failed CL-MAC reservation resolution consumes retry budget but does
    % not transmit a data packet. This avoids charging fictitious data TX.
    if strcmp(protocol,'CL-MAC (Proposed)') && reservationFailed
        for z=1:nAtt
            n=attempts(z);
            if ~isempty(qRetry{n})
                qRetry{n}(1)=qRetry{n}(1)+1;
                if qRetry{n}(1)>G.maxRetries
                    qRetry{n}(1)=[];
                    qTime{n}(1)=[];
                    dropped=dropped+1;
                end
            end
        end
        collisionEMA=(1-C.ema)*collisionEMA+C.ema*1;
        continue;
    end

    isCollision=nAtt>1;
    if nAtt>0
        residual(attempts)=max(0,residual(attempts)-G.txEnergyJ);
    end

    if isCollision
        for z=1:nAtt
            n=attempts(z);
            qRetry{n}(1)=qRetry{n}(1)+1;
            if qRetry{n}(1)>G.maxRetries
                qRetry{n}(1)=[]; qTime{n}(1)=[];
                dropped=dropped+1;
            end
        end
    elseif nAtt==1
        n=attempts(1);
        % Link-quality dependent packet reception; same random number
        % is used for all protocols for a given node and slot.
        pRx=min(0.995,max(0.02,T.linkQ(n,t)^1.35));
        if T.uSuccess(n,t)<pRx
            reservationDelay=0;
            if strcmp(protocol,'CL-MAC (Proposed)')
                reservationDelay=reservationRoundsUsed*G.reservationOverhead_s;
            end
            delays(end+1)=(t-qTime{n}(1)+1)*G.slot_s+reservationDelay; %#ok<AGROW>
            qTime{n}(1)=[]; qRetry{n}(1)=[];
            delivered=delivered+1;
            residual(n)=max(0,residual(n)-G.rxEnergyJ);
        else
            qRetry{n}(1)=qRetry{n}(1)+1;
            if qRetry{n}(1)>G.maxRetries
                qRetry{n}(1)=[]; qTime{n}(1)=[];
                dropped=dropped+1;
            end
        end
    end
    collisionEMA=(1-C.ema)*collisionEMA+C.ema*double(isCollision);
end

pending=sum(cellfun(@numel,qTime));
lost=dropped+pending;
if generated>0
    pdr=delivered/generated;
    loss=dropped/generated;
else
    pdr=NaN; loss=NaN;
end
duration=K*G.slot_s;
thr=delivered*G.packetBits/duration/1e6;
if isempty(delays), dbar=NaN; else, dbar=mean(delays); end
energyUsed=sum(G.initialEnergyJ*T.initialBatteryFactor-residual);
if delivered>0, ePer=energyUsed/delivered; else, ePer=NaN; end
M.generated=generated; M.delivered=delivered; M.dropped=dropped;
M.pending=pending; M.PDR=pdr; M.throughputMbps=thr;
M.lossRatio=loss; M.meanDelay_s=dbar; M.energyJ=energyUsed;
M.energyPerDeliveredJ=ePer; M.meanResidualJ=mean(residual);
M.aliveNodes=sum(residual>0); M.delaySamples=delays;
end

function plotScenarioLines(Summary,S,protocols,metric,yLabel,outDir)
nS=numel(S); nP=numel(protocols);
Y=nan(nS,nP); E=Y;
for s=1:nS
    for p=1:nP
        ix=strcmp(Summary.Scenario,S(s).name) & strcmp(Summary.Protocol,protocols{p});
        vm=Summary.([metric '_Mean']);
        vs=Summary.([metric '_SD']);
        Y(s,p)=vm(ix);
        E(s,p)=vs(ix);
    end
end
f=figure('Color','w','Position',[100 100 1100 560]); hold on; grid on; box on;
for p=1:nP
    errorbar(1:nS,Y(:,p),E(:,p),'-o','LineWidth',2,'MarkerSize',7, ...
        'CapSize',5,'DisplayName',protocols{p});
end
xlabel('Operating scenario'); ylabel(yLabel);
title([yLabel ' across all operating scenarios']);
set(gca,'XTick',1:nS,'XTickLabel',strrep({S.name},'_',' '), ...
    'XTickLabelRotation',22,'FontSize',10);
legend('Location','best');
exportgraphics(f,fullfile(outDir,['Scenario_' metric '_Line.png']),'Resolution',400);
end

function plotMetricLines(x,Y,protocols,xlab,ylab,ttl)
nexttile; hold on; grid on; box on;
for p=1:numel(protocols)
    plot(x,Y(:,p),'-o','LineWidth',1.8,'MarkerSize',5,'DisplayName',protocols{p});
end
xlabel(xlab); ylabel(ylab); title(ttl); legend('Location','best');
end

function plot3DProtocolSurfaces(xLoad,yMob,Z,protocols,ttl,zlab,fileName,outDir)
f=figure('Color','w','Position',[60 100 max(1450,430*numel(protocols)) 520]);
tiledlayout(1,numel(protocols),'TileSpacing','compact','Padding','compact');
zmin=min(Z(:),[],'omitnan'); zmax=max(Z(:),[],'omitnan');
if ~isfinite(zmin), zmin=0; end
if ~isfinite(zmax) || zmax<=zmin, zmax=zmin+1; end
for p=1:numel(protocols)
    nexttile;
    [X,Y]=meshgrid(xLoad,yMob);
    surf(X,Y,Z(:,:,p),'EdgeColor',[0.25 0.25 0.25],'LineWidth',0.35);
    xlabel('Offered arrival probability'); ylabel('UAV mobility factor');
    zlabel(zlab); title(protocols{p});
    xlim([min(xLoad) max(xLoad)]); ylim([min(yMob) max(yMob)]);
    zlim([zmin zmax]); grid on; box on; view(135,28);
    set(gca,'FontSize',9);
end
sgtitle([ttl ' under common axis scales']);
exportgraphics(f,fullfile(outDir,fileName),'Resolution',400);
end

function plotOneSurface(xLoad,yMob,Z,ttl,zlab,fileName,outDir)
f=figure('Color','w','Position',[100 100 850 620]);
[X,Y]=meshgrid(xLoad,yMob);
surf(X,Y,Z,'EdgeColor',[0.25 0.25 0.25],'LineWidth',0.4);
xlabel('Offered arrival probability'); ylabel('UAV mobility factor');
zlabel(zlab); title(ttl); colorbar; grid on; box on; view(135,28);
set(gca,'FontSize',11);
exportgraphics(f,fullfile(outDir,fileName),'Resolution',400);
end
