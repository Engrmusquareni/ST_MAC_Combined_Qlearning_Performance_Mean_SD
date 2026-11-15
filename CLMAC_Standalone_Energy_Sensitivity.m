%% CLMAC_Standalone_Energy_Sensitivity.m
% Standalone energy sensitivity experiments for the latest OptimizedAccess model.
% No edits to, or execution of, the main simulation script are required.
% MATLAB R2021b+ recommended.
clear; close all; clc; rng(13112026,'twister');

%% Model parameters copied from CL_MAC_ThreeProtocol_OptimizedAccess.m
G.N=20; G.slots=5000; G.slot_s=0.010; G.packetBits=500;
G.maxRetries=4; G.maxQueue=50; G.initialEnergyJ=2.0;
G.txEnergyJ=6.0e-5; G.rxEnergyJ=5.2e-5; G.idleEnergyJ=1.5e-5;
G.reservationOverhead_s=2.0e-4;
G.requestEnergyJ=6.0e-3*G.reservationOverhead_s;
G.grantRxEnergyJ=5.2e-3*G.reservationOverhead_s;
G.deploymentLengthM=100; G.uavAltitudeM=50;
G.alpha=0.30; G.beta=0.20; G.delta=0.50; G.pAdaptiveMax=0.85;
G.replicates=30; G.eatdmaAttemptProb=0.10; G.tdmaSchedule='round-robin';
C.CWmin=4; C.CWmax=128; C.ema=0.08;
C.queueWeight=0.40; C.ageWeight=0.30; C.linkWeight=0.20; C.energyWeight=0.10;
C.ageThreshold_s=0.50; C.resolutionRounds=3;
protocols={'UAV-TDMA','UAV-EATDMA','CL-MAC (Proposed)'}; nP=numel(protocols);
baseScenario=struct('name','Energy_Sweep','arrivalProb',0.035, ...
    'bursty',false,'mobility',0.20,'channelLoss',0.06,'batteryHet',0.15);
outDir='CLMAC_Standalone_Energy_Results';
if ~exist(outDir,'dir'), mkdir(outDir); end

%% A. Energy versus sessions per cycle
sessionsValues=10:10:100;
[ES,ESsd]=energySweep(G,C,protocols,baseScenario,sessionsValues,'sessions',G.replicates);
plotEnergy(sessionsValues,ES,ESsd,protocols,'Sessions per cycle', ...
    fullfile(outDir,'Energy_vs_Sessions_Per_Cycle.png'));

%% B. Energy versus event-generation probability
probValues=0.1:0.1:1.0;
[EP,EPsd]=energySweep(G,C,protocols,baseScenario,probValues,'probability',G.replicates);
plotEnergy(probValues,EP,EPsd,protocols,'Event-generation probability', ...
    fullfile(outDir,'Energy_vs_Event_Generation_Probability.png'));

%% C. Energy versus packet size (TX/RX energy scales from the 500-bit reference)
packetValues=50:50:500;
[EK,EKsd]=energySweep(G,C,protocols,baseScenario,packetValues,'packetSize',G.replicates);
plotEnergy(packetValues,EK,EKsd,protocols,'Packet size (bits)', ...
    fullfile(outDir,'Energy_vs_Packet_Size.png'));

%% D. Energy versus number of simulated sensor senders / cluster-heads
nodeValues=10:10:100;
[EN,ENsd]=energySweep(G,C,protocols,baseScenario,nodeValues,'nodes',G.replicates);
plotEnergy(nodeValues,EN,ENsd,protocols,'Number of simulated senders', ...
    fullfile(outDir,'Energy_vs_Number_of_Sensor_Nodes.png'));

%% E(a). Energy versus UAV altitude
altitudeValues=10:10:100;
[EA,EAsd]=energySweep(G,C,protocols,baseScenario,altitudeValues,'altitude',G.replicates);
plotEnergy(altitudeValues,EA,EAsd,protocols,'UAV altitude (m)', ...
    fullfile(outDir,'Fig6a_Energy_vs_UAV_Altitude.png'));

%% E(b). 3D surface: UAV altitude, sessions per cycle and energy
surfaceSessions=10:10:100; surfaceAltitudes=10:10:100;
EnergySurfaceMean=nan(numel(surfaceAltitudes),numel(surfaceSessions),nP);
EnergySurfaceSD=EnergySurfaceMean;
for ia=1:numel(surfaceAltitudes)
    fprintf('3D surface altitude %d/%d: %g m\n',ia,numel(surfaceAltitudes),surfaceAltitudes(ia));
    for is=1:numel(surfaceSessions)
        vals=nan(G.replicates,nP);
        for rep=1:G.replicates
            seed=950000+ia*10000+is*100+rep;
            Gt=G; Gt.uavAltitudeM=surfaceAltitudes(ia);
            T=makeTraces(Gt,baseScenario,seed);
            T=imposeSessions(T,Gt,surfaceSessions(is),seed+500000);
            for p=1:nP
                M=runProtocol(Gt,C,baseScenario,T,protocols{p});
                vals(rep,p)=1000*M.energyJ;
            end
        end
        EnergySurfaceMean(ia,is,:)=mean(vals,1,'omitnan');
        EnergySurfaceSD(ia,is,:)=std(vals,0,1,'omitnan');
    end
end
f=figure('Color','w','Position',[50 80 1500 560]);
tiledlayout(1,nP,'TileSpacing','compact','Padding','compact');
[X,Y]=meshgrid(surfaceSessions,surfaceAltitudes);
for p=1:nP
    nexttile; surf(X,Y,EnergySurfaceMean(:,:,p),'EdgeColor',[.25 .25 .25],'LineWidth',.35);
    xlabel('Sessions per cycle'); ylabel('UAV altitude (m)'); zlabel('Energy consumption (mJ)');
    title(protocols{p}); grid on; box on; view(135,30); set(gca,'FontSize',10);
end
sgtitle('Energy Consumption versus UAV Altitude and Sessions per Cycle');
exportgraphics(f,fullfile(outDir,'Fig6b_3D_Energy_Surface.png'),'Resolution',400);

%% Export CSV and MAT data
writeEnergyCSV(sessionsValues,ES,ESsd,fullfile(outDir,'Energy_vs_Sessions_Per_Cycle.csv'));
writeEnergyCSV(probValues,EP,EPsd,fullfile(outDir,'Energy_vs_Event_Generation_Probability.csv'));
writeEnergyCSV(packetValues,EK,EKsd,fullfile(outDir,'Energy_vs_Packet_Size.csv'));
writeEnergyCSV(nodeValues,EN,ENsd,fullfile(outDir,'Energy_vs_Number_of_Sensor_Nodes.csv'));
writeEnergyCSV(altitudeValues,EA,EAsd,fullfile(outDir,'Energy_vs_UAV_Altitude.csv'));
save(fullfile(outDir,'CLMAC_Standalone_Energy_Results.mat'),'G','C','protocols', ...
    'sessionsValues','probValues','packetValues','nodeValues','altitudeValues', ...
    'ES','ESsd','EP','EPsd','EK','EKsd','EN','ENsd','EA','EAsd', ...
    'surfaceSessions','surfaceAltitudes','EnergySurfaceMean','EnergySurfaceSD');
fprintf('\nCompleted. Results saved in %s\n',outDir);

%% Local helper functions
function [Ym,Ys]=energySweep(G,C,protocols,sc,xv,mode,nRep)
nP=numel(protocols); Y=nan(numel(xv),nRep,nP);
for ix=1:numel(xv)
    fprintf('%s sweep %g (%d/%d)\n',mode,xv(ix),ix,numel(xv));
    for r=1:nRep
        seed=920000+ix*10000+r; Gt=G; st=sc;
        switch mode
            case 'sessions'
            case 'probability', st.arrivalProb=xv(ix);
            case 'packetSize'
                scale=xv(ix)/500; Gt.packetBits=xv(ix);
                Gt.txEnergyJ=G.txEnergyJ*scale; Gt.rxEnergyJ=G.rxEnergyJ*scale;
            case 'nodes', Gt.N=xv(ix);
            case 'altitude', Gt.uavAltitudeM=xv(ix);
            otherwise, error('Unknown sweep mode.');
        end
        T=makeTraces(Gt,st,seed);
        if strcmp(mode,'sessions'), T=imposeSessions(T,Gt,xv(ix),seed+500000); end
        for p=1:nP
            M=runProtocol(Gt,C,st,T,protocols{p}); Y(ix,r,p)=1000*M.energyJ;
        end
    end
end
Ym=squeeze(mean(Y,2,'omitnan')); Ys=squeeze(std(Y,0,2,'omitnan'));
end

function T=imposeSessions(T,G,nSessions,seed)
rng(seed,'twister'); total=G.N*G.slots;
if nSessions>total, error('Sessions exceed available node-slot positions.'); end
T.arrivals=false(G.N,G.slots); ix=randperm(total,nSessions); T.arrivals(ix)=true;
end

function plotEnergy(x,Y,S,protocols,xlab,file)
f=figure('Color','w','Position',[100 100 1000 620]); hold on; grid on; box on;
for p=1:numel(protocols)
    errorbar(x,Y(:,p),S(:,p),'-o','LineWidth',2,'MarkerSize',7, ...
        'CapSize',5,'DisplayName',protocols{p});
end
xlabel(xlab); ylabel('Energy consumption (mJ)');
title(['Energy Consumption versus ' xlab]); legend('Location','best'); set(gca,'FontSize',11);
exportgraphics(f,file,'Resolution',400);
end

function writeEnergyCSV(x,Y,S,file)
tags={'UAV_TDMA','UAV_EATDMA','CL_MAC_Proposed'};
names={'IndependentVariable'};
for p=1:3, names{end+1}=[tags{p} '_Mean_mJ']; end
for p=1:3, names{end+1}=[tags{p} '_SD_mJ']; end
T=array2table([x(:),Y,S],'VariableNames',names); writetable(T,file);
end
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
