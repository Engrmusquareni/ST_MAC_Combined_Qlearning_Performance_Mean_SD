%% QLEARNING_FINAL_CONVERGENCE_AND_ENERGY_MODEL
% MATLAB R2021a compatible. Self-contained; no input CSV files required.
%
% This script revises the previous version by:
% 1) Using alpha(s,a)=0.20/N(s,a)^0.51.
% 2) Increasing the training budget to 20,000,000 updates.
% 3) Retaining epsilon-greedy exploration: epsilon=max(0.05,0.80*0.9999^t).
% 4) Balancing exploratory action visits within each state.
% 5) Reporting FINAL checkpoint variation separately from the largest
%    historical variation.
% 6) Requiring Q-reference accuracy, checkpoint stability, correct greedy
%    policy, adequate visits, and five consecutive qualifying checkpoints.
%
% SCIENTIFIC SCOPE:
% Training uses a stationary five-state self-transition MDP (s_next=s)
% so its exact Bellman-optimal Q-table can be calculated. This is a
% controlled MATLAB convergence test, not a reproduction of a changing
% battery-state ns-3 simulation. The 3000-s radio energy evaluation is
% performed separately using the learned greedy policy.
%
% Radio currents and transition parameters are illustrative placeholders.
% Replace them with calibrated device/ns-3 parameters before publication.
%
% Save as: Qlearning_Final_Convergence.m

clear; clc; close all;
rng(11,'twister');

% All outputs from both stages are collected in one directory.
outputFolder = 'ST_MAC_Combined_Results';
if ~exist(outputFolder,'dir'), mkdir(outputFolder); end

%% 1. Q-LEARNING CONFIGURATION
alpha0 = 0.20;
alphaExponent = 0.51;
gamma = 0.90;

epsilon0 = 0.80;
epsilonMin = 0.05;
epsilonDecay = 0.9999;

nStates = 5;
nActions = 5;
sleepActions = [0.05 0.10 0.20 0.35 0.50]; % seconds

maxUpdates = 20000000;
checkpointEvery = 1000;

qChangeTolerance = 1e-3;
qReferenceTolerance = 1e-2;
stableCheckpointsRequired = 5;
minimumVisitsPerPair = 1000;

%% 2. RADIO ENERGY PARAMETERS (PLACEHOLDERS)
Vradio = 3.0;          % volts
I_TX = 0.0174;         % amperes
I_RX = 0.0197;         % amperes
I_IDLE = 0.0197;       % amperes
I_SLEEP = 20e-6;       % amperes

activeDuration = 0.10; % seconds
fractionTX = 0.20;
fractionRX = 0.20;
fractionIdle = 0.60;

if abs(fractionTX+fractionRX+fractionIdle-1) > 1e-12
    error('TX, RX and idle fractions must sum to one.');
end

% Incremental transition energy, separate from state-duration energy.
I_transition_incremental = 0.015; % A, placeholder
t_transition = 100e-6;             % s, placeholder
nTransitionsPerCycle = 4;

%% 3. DEFINE STATIONARY TRAINING REWARD MATRIX
% Battery-state labels:
% S0: 0-20%, S1: 20-40%, S2: 40-60%,
% S3: 60-80%, S4: 80-100%.
%
% Target sleep durations are specified by the model:
% low battery -> longer sleep; high battery -> shorter sleep.
idealSleep = [0.50 0.35 0.20 0.10 0.05];

tTX = activeDuration*fractionTX;
tRX = activeDuration*fractionRX;
tIdle = activeDuration*fractionIdle;

EtxAction = Vradio*I_TX*tTX*ones(1,nActions);
ErxAction = Vradio*I_RX*tRX*ones(1,nActions);
EidleAction = Vradio*I_IDLE*tIdle*ones(1,nActions);
EsleepAction = Vradio*I_SLEEP*sleepActions;
EtransitionAction = nTransitionsPerCycle*Vradio* ...
    I_transition_incremental*t_transition*ones(1,nActions);

EcycleAction = EtxAction+ErxAction+EidleAction+ ...
    EsleepAction+EtransitionAction;

R_sleep = zeros(nStates,nActions);
R_energy = zeros(nStates,nActions);
R = zeros(nStates,nActions);

for s = 1:nStates
    for a = 1:nActions
        R_sleep(s,a) = max(0, ...
            1-abs(sleepActions(a)-idealSleep(s))/max(sleepActions));
        R_energy(s,a) = 1/(1+EcycleAction(a));
        R(s,a) = 0.70*R_sleep(s,a)+0.30*R_energy(s,a);
    end
end

%% 4. EXACT BELLMAN-OPTIMAL REFERENCE
% For the stationary self-transition MDP:
% V*(s)=max_a R(s,a)/(1-gamma)
% Q*(s,a)=R(s,a)+gamma*V*(s).
Vstar = max(R,[],2)/(1-gamma);
Qstar = R+gamma*repmat(Vstar,1,nActions);
[~,optimalAction] = max(Qstar,[],2);
optimalPolicy = optimalAction(:)';
optimalMeanMaxQ = mean(max(Qstar,[],2));

%% 5. INITIALIZE LEARNING AND CHECKPOINT STORAGE
Q = zeros(nStates,nActions);
visitCount = zeros(nStates,nActions);
explorationPointer = zeros(nStates,1);

epsilon = epsilon0;
stableCount = 0;
converged = false;
convergenceUpdate = NaN;

nMaxSnapshots = floor(maxUpdates/checkpointEvery);
Qhistory = zeros(nMaxSnapshots,nStates,nActions);
snapshotUpdate = zeros(nMaxSnapshots,1);
snapshotEpsilon = zeros(nMaxSnapshots,1);
meanMaxQ = zeros(nMaxSnapshots,1);
maxDeltaQ = nan(nMaxSnapshots,1);
maxQError = zeros(nMaxSnapshots,1);
policyMatches = false(nMaxSnapshots,1);
minimumVisits = zeros(nMaxSnapshots,1);
snapshotCount = 0;

% Compact logs: one record per checkpoint, not per individual update.
meanRewardCheckpoint = zeros(nMaxSnapshots,1);
meanAlphaCheckpoint = zeros(nMaxSnapshots,1);
meanAbsTDCheckpoint = zeros(nMaxSnapshots,1);
rewardAccumulator = 0;
alphaAccumulator = 0;
tdAccumulator = 0;
blockCount = 0;

%% 6. Q-LEARNING TRAINING
fprintf('\nStarting Q-learning training: maximum %d updates.\n',maxUpdates);

for update = 1:maxUpdates

    % Deterministic uniform state sampling: every state receives updates.
    s = mod(update-1,nStates)+1;

    % Epsilon-greedy action selection.
    if rand < epsilon
        % Round-robin exploration within each state prevents action starvation.
        a = mod(explorationPointer(s),nActions)+1;
        explorationPointer(s) = explorationPointer(s)+1;
    else
        % Greedy action with random tie-breaking.
        qrow = Q(s,:);
        tied = find(abs(qrow-max(qrow)) < 1e-12);
        a = tied(randi(numel(tied)));
    end

    % Stationary self-transition.
    sNext = s;
    reward = R(s,a);

    % Diminishing per-state/action learning rate.
    visitCount(s,a) = visitCount(s,a)+1;
    alphaSA = alpha0/(visitCount(s,a)^alphaExponent);

    tdTarget = reward+gamma*max(Q(sNext,:));
    tdError = tdTarget-Q(s,a);
    Q(s,a) = Q(s,a)+alphaSA*tdError;

    % Epsilon decay.
    epsilon = max(epsilonMin,epsilon*epsilonDecay);

    rewardAccumulator = rewardAccumulator+reward;
    alphaAccumulator = alphaAccumulator+alphaSA;
    tdAccumulator = tdAccumulator+abs(tdError);
    blockCount = blockCount+1;

    %% Checkpoint and convergence tests
    if mod(update,checkpointEvery)==0
        snapshotCount = snapshotCount+1;

        Qhistory(snapshotCount,:,:) = Q;
        snapshotUpdate(snapshotCount) = update;
        snapshotEpsilon(snapshotCount) = epsilon;
        meanMaxQ(snapshotCount) = mean(max(Q,[],2));
        maxQError(snapshotCount) = max(abs(Q(:)-Qstar(:)));
        minimumVisits(snapshotCount) = min(visitCount(:));

        meanRewardCheckpoint(snapshotCount) = ...
            rewardAccumulator/max(blockCount,1);
        meanAlphaCheckpoint(snapshotCount) = ...
            alphaAccumulator/max(blockCount,1);
        meanAbsTDCheckpoint(snapshotCount) = ...
            tdAccumulator/max(blockCount,1);

        rewardAccumulator = 0;
        alphaAccumulator = 0;
        tdAccumulator = 0;
        blockCount = 0;

        [~,currentPolicy] = max(Q,[],2);
        policyMatches(snapshotCount) = ...
            isequal(currentPolicy(:)',optimalPolicy);

        if snapshotCount > 1
            Qprevious = squeeze(Qhistory(snapshotCount-1,:,:));
            maxDeltaQ(snapshotCount) = max(abs(Q(:)-Qprevious(:)));
        end

        enoughVisits = all(visitCount(:)>=minimumVisitsPerPair);
        policyCorrect = policyMatches(snapshotCount);
        referenceAccurate = ...
            maxQError(snapshotCount)<=qReferenceTolerance;

        if snapshotCount > 1
            checkpointStable = ...
                maxDeltaQ(snapshotCount)<qChangeTolerance;
        else
            checkpointStable = false;
        end

        if enoughVisits && policyCorrect && ...
                referenceAccurate && checkpointStable
            stableCount = stableCount+1;
        else
            stableCount = 0;
        end

        if stableCount>=stableCheckpointsRequired
            converged = true;
            convergenceUpdate = update;
            fprintf('All convergence criteria met at update %d.\n',update);
            break;
        end
    end

    % Progress message every one million updates.
    if mod(update,1000000)==0
        fprintf(['Update %d / %d | mean max Q = %.6f | ' ...
            'max Q error = %.6f | epsilon = %.4f\n'], ...
            update,maxUpdates,mean(max(Q,[],2)), ...
            max(abs(Q(:)-Qstar(:))),epsilon);
    end
end

updatesDone = update;

%% 7. TRIM CHECKPOINT ARRAYS
Qhistory = Qhistory(1:snapshotCount,:,:);
snapshotUpdate = snapshotUpdate(1:snapshotCount);
snapshotEpsilon = snapshotEpsilon(1:snapshotCount);
meanMaxQ = meanMaxQ(1:snapshotCount);
maxDeltaQ = maxDeltaQ(1:snapshotCount);
maxQError = maxQError(1:snapshotCount);
policyMatches = policyMatches(1:snapshotCount);
minimumVisits = minimumVisits(1:snapshotCount);
meanRewardCheckpoint = meanRewardCheckpoint(1:snapshotCount);
meanAlphaCheckpoint = meanAlphaCheckpoint(1:snapshotCount);
meanAbsTDCheckpoint = meanAbsTDCheckpoint(1:snapshotCount);

Qfinal = Q;
[finalMaxQ,finalAction] = max(Qfinal,[],2);
finalPolicy = finalAction(:)';
finalSleep = sleepActions(finalAction);

validDelta = maxDeltaQ(~isnan(maxDeltaQ));
if isempty(validDelta)
    finalCheckpointChange = NaN;
    largestHistoricalChange = NaN;
else
    finalCheckpointChange = validDelta(end);
    largestHistoricalChange = max(validDelta);
end

finalQError = max(abs(Qfinal(:)-Qstar(:)));
finalPolicyCorrect = isequal(finalPolicy,optimalPolicy);
finalVisitsOK = all(visitCount(:)>=minimumVisitsPerPair);

%% 8. FIGURE 1: MEAN MAXIMUM Q-VALUE
figure('Color','w');
plot(snapshotUpdate,meanMaxQ,'LineWidth',2); hold on;
yline(optimalMeanMaxQ,'--r','Bellman-optimal reference');
grid on;
xlabel('Learning update index');
ylabel('Mean maximum Q-value across states');
title('Q-table learning progression');
if converged
    xline(convergenceUpdate,':k','Convergence accepted');
end
saveas(gcf,fullfile(outputFolder,'Fig1_QTable_Convergence.png'));

%% 9. FIGURE 2: CHECKPOINT Q-TABLE VARIATION
figure('Color','w');
valid = ~isnan(maxDeltaQ);
semilogy(snapshotUpdate(valid),max(maxDeltaQ(valid),eps), ...
    'LineWidth',1.6); hold on;
yline(qChangeTolerance,'--r','Tolerance = 0.001');
grid on;
xlabel('Learning update index');
ylabel('Maximum absolute Q-table change');
title('Q-table checkpoint stability');
saveas(gcf,fullfile(outputFolder,'Fig2_QTable_Stability.png'));

%% 10. FIGURE 3: ERROR AGAINST BELLMAN-OPTIMAL Q-TABLE
figure('Color','w');
semilogy(snapshotUpdate,max(maxQError,eps),'LineWidth',1.8);
hold on;
yline(qReferenceTolerance,'--r','Reference error tolerance = 0.01');
grid on;
xlabel('Learning update index');
ylabel('Maximum absolute error |Q-Q^*|');
title('Q-table accuracy against Bellman reference');
saveas(gcf,fullfile(outputFolder,'Fig3_QTable_Reference_Error.png'));

%% 11. FIGURE 4: EPSILON DECAY AT CHECKPOINTS
figure('Color','w');
plot(snapshotUpdate,snapshotEpsilon,'LineWidth',1.8);
hold on;
yline(epsilonMin,'--r','Minimum epsilon');
grid on;
xlabel('Learning update index');
ylabel('Exploration probability');
title('Epsilon-greedy exploration schedule');
saveas(gcf,fullfile(outputFolder,'Fig4_Epsilon_Decay.png'));

%% 12. FIGURE 5: MEAN REWARD PER CHECKPOINT
figure('Color','w');
plot(snapshotUpdate,meanRewardCheckpoint,'LineWidth',1.8);
grid on;
xlabel('Learning update index');
ylabel('Mean reward per checkpoint');
title('Reward progression during Q-learning');
saveas(gcf,fullfile(outputFolder,'Fig5_Reward_Progression.png'));

%% 13. FIGURE 6: FINAL Q-TABLE HEATMAP
figure('Color','w');
imagesc(Qfinal); colorbar;
xlabel('Sleep action');
ylabel('Battery state');
title('Final learned Q-table');
xticks(1:5);
xticklabels({'0.05','0.10','0.20','0.35','0.50'});
yticks(1:5);
yticklabels({'S0','S1','S2','S3','S4'});
saveas(gcf,fullfile(outputFolder,'Fig6_Final_QTable.png'));

%% 14. FIGURE 7: STATE-SPECIFIC MAXIMUM Q-VALUES
stateMaxHistory = zeros(snapshotCount,nStates);
for k = 1:snapshotCount
    Qk = squeeze(Qhistory(k,:,:));
    stateMaxHistory(k,:) = max(Qk,[],2)';
end

figure('Color','w');
plot(snapshotUpdate,stateMaxHistory,'LineWidth',1.6);
hold on;
for s = 1:nStates
    yline(max(Qstar(s,:)),'--');
end
grid on;
xlabel('Learning update index');
ylabel('Maximum Q-value');
title('State-specific Q-value convergence');
legend({'S0','S1','S2','S3','S4', ...
    'S0 reference','S1 reference','S2 reference', ...
    'S3 reference','S4 reference'},'Location','eastoutside');
saveas(gcf,fullfile(outputFolder,'Fig7_StateSpecific_Convergence.png'));

%% 15. FIGURE 8: STATE-ACTION VISIT COUNTS
figure('Color','w');
imagesc(visitCount); colorbar;
xlabel('Sleep action');
ylabel('Battery state');
title('State-action visitation counts');
xticks(1:5);
xticklabels({'0.05','0.10','0.20','0.35','0.50'});
yticks(1:5);
yticklabels({'S0','S1','S2','S3','S4'});
saveas(gcf,fullfile(outputFolder,'Fig8_StateAction_Visits.png'));

%% 16. SEPARATE 3000-s ENERGY EVALUATION USING GREEDY POLICY
evaluationDuration = 3000;
initialBattery = 1200;
battery = initialBattery;
elapsed = 0;

evalTime = 0;
evalBattery = battery;
evalTX = [];
evalRX = [];
evalIdle = [];
evalSleep = [];
evalTransition = [];
evalTotal = [];
evalState = [];
evalAction = [];

while elapsed < evaluationDuration && battery > 0

    batteryRatio = max(0,min(1,battery/initialBattery));
    s = min(nStates,floor(batteryRatio*nStates)+1);

    a = finalAction(s);
    sleepT = sleepActions(a);

    ttx = activeDuration*fractionTX;
    trx = activeDuration*fractionRX;
    tidle = activeDuration*fractionIdle;

    eTX = Vradio*I_TX*ttx;
    eRX = Vradio*I_RX*trx;
    eIdle = Vradio*I_IDLE*tidle;
    eSleep = Vradio*I_SLEEP*sleepT;
    eTransition = nTransitionsPerCycle*Vradio* ...
        I_transition_incremental*t_transition;

    cycleDuration = activeDuration+sleepT;
    cycleEnergy = eTX+eRX+eIdle+eSleep+eTransition;

    remainingTime = evaluationDuration-elapsed;
    if cycleDuration > remainingTime
        scale = remainingTime/cycleDuration;
        eTX=eTX*scale;
        eRX=eRX*scale;
        eIdle=eIdle*scale;
        eSleep=eSleep*scale;
        eTransition=eTransition*scale;
        cycleEnergy=cycleEnergy*scale;
        cycleDuration=remainingTime;
    end

    cycleEnergy = min(cycleEnergy,battery);
    battery = max(0,battery-cycleEnergy);
    elapsed = elapsed+cycleDuration;

    evalTX(end+1,1)=eTX; %#ok<SAGROW>
    evalRX(end+1,1)=eRX; %#ok<SAGROW>
    evalIdle(end+1,1)=eIdle; %#ok<SAGROW>
    evalSleep(end+1,1)=eSleep; %#ok<SAGROW>
    evalTransition(end+1,1)=eTransition; %#ok<SAGROW>
    evalTotal(end+1,1)=cycleEnergy; %#ok<SAGROW>
    evalState(end+1,1)=s-1; %#ok<SAGROW>
    evalAction(end+1,1)=a-1; %#ok<SAGROW>
    evalTime(end+1,1)=elapsed; %#ok<SAGROW>
    evalBattery(end+1,1)=battery; %#ok<SAGROW>
end

%% 17. FIGURE 9: ENERGY BREAKDOWN
energyTotals = [sum(evalTX),sum(evalRX),sum(evalIdle), ...
    sum(evalSleep),sum(evalTransition)];

figure('Color','w');
bar(energyTotals);
grid on;
xticklabels({'Transmission','Reception','Idle','Sleep','Transitions'});
ylabel('Energy consumption (J)');
title('Radio energy consumption breakdown');
saveas(gcf,fullfile(outputFolder,'Fig9_Energy_Breakdown.png'));

%% 18. FIGURE 10: RESIDUAL BATTERY
figure('Color','w');
plot(evalTime,evalBattery,'LineWidth',1.8);
grid on;
xlabel('Evaluation time (s)');
ylabel('Residual battery energy (J)');
title('Residual battery energy under learned greedy policy');
saveas(gcf,fullfile(outputFolder,'Fig10_Residual_Battery.png'));

%% 19. EXPORT Q-TABLE HISTORY
stateLabels = {'S0_0_20','S1_20_40','S2_40_60', ...
    'S3_60_80','S4_80_100'};
actionLabels = {'A0_005s','A1_010s','A2_020s','A3_035s','A4_050s'};

QhistTable = table(snapshotUpdate,snapshotEpsilon,meanMaxQ, ...
    maxDeltaQ,maxQError,policyMatches,minimumVisits, ...
    meanRewardCheckpoint,meanAlphaCheckpoint,meanAbsTDCheckpoint, ...
    'VariableNames',{'update','epsilon','mean_max_Q', ...
    'max_delta_Q','max_abs_Q_error','policy_matches','min_visits', ...
    'mean_reward','mean_alpha','mean_abs_TD_error'});

for s = 0:4
    for a = 0:4
        QhistTable.(sprintf('Q_S%d_A%d',s,a)) = ...
            squeeze(Qhistory(:,s+1,a+1));
    end
end
writetable(QhistTable,fullfile(outputFolder,'qtable_history.csv'));

FinalQTable = array2table(Qfinal,'VariableNames',actionLabels, ...
    'RowNames',stateLabels);
writetable(FinalQTable,fullfile(outputFolder,'Final_QTable.csv'),'WriteRowNames',true);

BellmanQTable = array2table(Qstar,'VariableNames',actionLabels, ...
    'RowNames',stateLabels);
writetable(BellmanQTable,fullfile(outputFolder,'Bellman_Optimal_QTable.csv'),'WriteRowNames',true);

PolicyTable = table((0:4)',optimalAction-1,finalSleep(:), ...
    finalMaxQ(:),max(visitCount,[],2), ...
    'VariableNames',{'StateIndex','ActionIndex','Sleep_s', ...
    'MaximumQ','MaximumActionVisits'});
writetable(PolicyTable,fullfile(outputFolder,'Final_Optimal_Policy.csv'));

%% 20. EXPORT LEARNING LOG AT CHECKPOINTS
LearningLog = table(snapshotUpdate,snapshotEpsilon,meanRewardCheckpoint, ...
    meanAlphaCheckpoint,meanAbsTDCheckpoint,meanMaxQ,maxQError, ...
    maxDeltaQ,'VariableNames',{'update','epsilon','mean_reward', ...
    'mean_alpha','mean_abs_TD_error','mean_max_Q', ...
    'max_abs_Q_error','max_delta_Q'});
writetable(LearningLog,fullfile(outputFolder,'Qlearning_Learning_Log.csv'));

%% 21. EXPORT ENERGY LOGS AND SUMMARY
EnergyEvents = table(evalTime(2:end),evalTX,evalRX,evalIdle, ...
    evalSleep,evalTransition,evalTotal,evalState,evalAction, ...
    'VariableNames',{'time_s','TxEnergy_J','RxEnergy_J', ...
    'IdleEnergy_J','SleepEnergy_J','TransitionEnergy_J', ...
    'TotalEnergy_J','BatteryState','Action'});
writetable(EnergyEvents,fullfile(outputFolder,'Energy_Events.csv'));

EnergySummary = table(energyTotals(1),energyTotals(2), ...
    energyTotals(3),energyTotals(4),energyTotals(5), ...
    sum(energyTotals),'VariableNames',{'Transmission_J', ...
    'Reception_J','IdleListening_J','Sleep_J', ...
    'Transitions_J','TotalEnergy_J'});
writetable(EnergySummary,fullfile(outputFolder,'Energy_Summary.csv'));

%% 22. SAVE MATLAB WORKSPACE
save(fullfile(outputFolder,'Qlearning_Final_Convergence_Results.mat'));

%% 23. FINAL REPORT
fprintf('\n========== FINAL Q-LEARNING CONVERGENCE REPORT ==========\n');
fprintf('Training updates performed: %d\n',updatesDone);
fprintf('Final epsilon: %.6f\n',epsilon);
fprintf('Learning rate: alpha0/N(s,a)^%.2f\n',alphaExponent);
fprintf('Final mean maximum Q: %.8f\n',mean(max(Qfinal,[],2)));
fprintf('Bellman-optimal mean maximum Q: %.8f\n',optimalMeanMaxQ);
fprintf('Final maximum absolute Q error: %.8f\n',finalQError);
fprintf('Final checkpoint Q change: %.8f\n',finalCheckpointChange);
fprintf('Largest historical checkpoint change: %.8f\n', ...
    largestHistoricalChange);
fprintf('Minimum state-action visits: %d\n',min(visitCount(:)));
fprintf('Greedy policy matches Bellman-optimal policy: %d\n', ...
    finalPolicyCorrect);
fprintf('All state-action visit requirements met: %d\n',finalVisitsOK);

if converged
    fprintf('CONVERGENCE ACCEPTED at update %d.\n',convergenceUpdate);
else
    fprintf('CONVERGENCE NOT ACCEPTED within the update budget.\n');
    fprintf('Do not describe this run as fully converged.\n');
end

fprintf('\nFinal greedy policy:\n');
disp(PolicyTable);

fprintf('\n========== RADIO ENERGY REPORT ==========\n');
fprintf('Transmission: %.9f J\n',energyTotals(1));
fprintf('Reception: %.9f J\n',energyTotals(2));
fprintf('Idle listening: %.9f J\n',energyTotals(3));
fprintf('Sleep: %.9f J\n',energyTotals(4));
fprintf('Transitions: %.9f J\n',energyTotals(5));
fprintf('Total energy: %.9f J\n',sum(energyTotals));
fprintf('Residual battery: %.9f J\n',battery);

fprintf('\nQ-learning stage complete. Continuing to network comparison...\n');
fprintf(['RESEARCH NOTE: training uses a stationary self-transition MDP; ' ...
    'energy evaluation is separate. These are MATLAB model results, ' ...
    'not ns-3.45 measurements.\n']);
fprintf(['Radio currents and transition parameters are illustrative ' ...
    'placeholders and must be calibrated before publication.\n']);

%% ST-MAC PERFORMANCE EVALUATION: ST-MAC vs TDMA vs EATDMA
% MATLAB R2021a compatible; no Statistics and Machine Learning Toolbox required.
%
% Generates six publication-quality figures:
% 14 Average energy per step vs number of nodes
% 15 Packet delivery ratio vs number of nodes
% 16 Estimated first-node network lifetime vs number of nodes
% 17 Network throughput vs number of nodes
% 18 Collision rate vs node density
% 19 End-to-end latency vs number of nodes
%
% MODEL SCOPE AND ASSUMPTIONS:
% This is a configurable discrete-time comparative MAC abstraction, NOT a
% standards-complete IEEE 802.15.4 implementation or a replacement for ns-3.
% TDMA: fixed round-robin slots, one scheduled node per time step.
% EATDMA: simplified event-aware round-robin service of queued nodes.
% ST-MAC: learned greedy sleep policy [0.50 0.35 0.20 0.10 0.05] s for
% battery states S0 (lowest) through S4 (highest).
% ST-MAC collisions occur when multiple awake nodes contend in one step.
% TDMA/EATDMA are collision-free by construction in this abstraction.
% Network lifetime is EXTRAPOLATED from measured average node energy drain;
% it is not an observed first-node-death time.
% Replace the scheduling/PHY rules if your manuscript defines the protocols
% differently. State all assumptions when reporting the results.

% Q-learning variables and figures are retained; no second clear/close.

%% 1. FAIR, COMMON NETWORK AND RADIO CONFIGURATION
% The same traffic arrivals, PHY random variates, radio powers, packet size,
% battery capacity, simulation duration and energy accounting are used for
% all three MAC protocols.
%
% Protocol scheduling remains distinct:
%   ST-MAC  : learned adaptive sleep; awake nodes contend (collisions possible)
%   TDMA    : fixed round-robin slots (collision-free by slot assignment)
%   EATDMA  : event-triggered round-robin reservation (collision-free by
%             centralized reservation assumption)
%
% These are explicit MATLAB MAC abstractions, not complete IEEE 802.15.4
% implementations. The scheduling assumptions must be described in the paper.

cfg.areaSide_m          = 100;
cfg.initialEnergy_J     = 1200;
cfg.simTime_s           = 3000;
cfg.dt_s                = 0.05;       % resolves the 0.05-s minimum sleep action
cfg.eventRate_pkt_s     = 0.02;       % same Poisson event rate per node
cfg.packetSize_bytes    = 64;
cfg.queueCapacity       = 5;

% Use the SAME radio parameters as the Q-learning energy/reward model above.
cfg.Vradio              = Vradio;
cfg.I_TX                = I_TX;
cfg.I_RX                = I_RX;
cfg.I_IDLE              = I_IDLE;
cfg.I_SLEEP             = I_SLEEP;
cfg.Ptx_W               = cfg.Vradio*cfg.I_TX;
cfg.Prx_W               = cfg.Vradio*cfg.I_RX;
cfg.Pidle_W             = cfg.Vradio*cfg.I_IDLE;
cfg.Psleep_W            = cfg.Vradio*cfg.I_SLEEP;

% Use the same active-period composition as the Q-learning energy model.
cfg.activeDuration_s    = activeDuration;
cfg.fractionTX          = fractionTX;
cfg.fractionRX          = fractionRX;
cfg.fractionIdle        = fractionIdle;
cfg.txDuration_s        = activeDuration*fractionTX;
cfg.ackDuration_s       = activeDuration*fractionRX;

% One wake-up cycle includes the same four transition events used by the
% Q-learning energy model. This cost is charged whenever a node wakes,
% regardless of which MAC protocol caused the wake-up.
cfg.wakeupEnergy_J = nTransitionsPerCycle*Vradio* ...
    I_transition_incremental*t_transition;

% Shared, density-dependent PHY packet error model.
cfg.basePacketError     = 0.02;
cfg.densityPERfactor    = 0.00002;
cfg.maxPacketError      = 0.20;

% Learned greedy policy from the Q-learning stage, ordered S0 (lowest
% battery) through S4 (highest battery).
cfg.stateEdges          = [0 0.20 0.40 0.60 0.80 1.000001];
cfg.sleepActions_s      = finalSleep(:)';

% Experiment design
nodeCounts              = [10 20 40 60 80 100];
protocols               = {'ST-MAC','TDMA','EATDMA'};
numberOfRepeats         = 3;          % descriptive mean/SD; consider >=10 for final study
baseRandomSeed          = 20261101;
saveFigures             = true;

fprintf('\nLearned ST-MAC sleep policy S0...S4 (s): ');
fprintf('%.3f ',cfg.sleepActions_s);
fprintf('\n');
fprintf('Common radio powers: TX %.6f W, RX %.6f W, IDLE %.6f W, SLEEP %.8f W\n', ...
    cfg.Ptx_W,cfg.Prx_W,cfg.Pidle_W,cfg.Psleep_W);
fprintf('All protocols use the same event traffic and PHY random traces per repeat.\n');

%% 2. RUN PAIRED NODE-SCALING EXPERIMENTS
nN = numel(nodeCounts);
nP = numel(protocols);
nSteps = round(cfg.simTime_s/cfg.dt_s);

energyPerStep = zeros(nN,nP,numberOfRepeats);
pdr = zeros(nN,nP,numberOfRepeats);
lifetime_s = zeros(nN,nP,numberOfRepeats);
throughput_kbps = zeros(nN,nP,numberOfRepeats);
collisionRate = zeros(nN,nP,numberOfRepeats);
latency_ms = nan(nN,nP,numberOfRepeats);
generatedCount = zeros(nN,nP,numberOfRepeats);
deliveredCount = zeros(nN,nP,numberOfRepeats);
droppedCount = zeros(nN,nP,numberOfRepeats);
totalEnergy_J = zeros(nN,nP,numberOfRepeats);
residualEnergy_J = zeros(nN,nP,numberOfRepeats);

fprintf('\nRunning paired ST-MAC / TDMA / EATDMA simulations...\n');
for iN = 1:nN
    N = nodeCounts(iN);
    density_nodes_m2 = N/(cfg.areaSide_m^2);

    for r = 1:numberOfRepeats
        % Generate one common traffic trace and one common PHY random trace.
        % Every protocol receives exactly the same packet-arrival pattern
        % and per-time-step channel random variates for this N and repeat.
        rng(baseRandomSeed + 1000*iN + r,'twister');
        arrivalMask = rand(N,nSteps) < cfg.eventRate_pkt_s*cfg.dt_s;
        phyUniform = rand(1,nSteps);

        for iP = 1:nP
            result = simulateOneProtocol(protocols{iP},N, ...
                density_nodes_m2,cfg,arrivalMask,phyUniform);

            energyPerStep(iN,iP,r) = result.energyPerNodePerStep_J;
            pdr(iN,iP,r) = result.PDR;
            lifetime_s(iN,iP,r) = result.estimatedFirstNodeLifetime_s;
            throughput_kbps(iN,iP,r) = result.throughput_kbps;
            collisionRate(iN,iP,r) = result.collisionRate;
            latency_ms(iN,iP,r) = result.meanLatency_ms;
            generatedCount(iN,iP,r) = result.generatedPackets;
            deliveredCount(iN,iP,r) = result.deliveredPackets;
            droppedCount(iN,iP,r) = result.droppedPackets;
            totalEnergy_J(iN,iP,r) = result.totalEnergy_J;
            residualEnergy_J(iN,iP,r) = result.totalResidualEnergy_J;
        end
    end
    fprintf('Completed N=%d nodes (%d paired repeats).\n',N,numberOfRepeats);
end

%% 3. AVERAGE METRICS OVER PAIRED REPEATS
meanEnergy = mean(energyPerStep,3);
meanPDR = mean(pdr,3);
meanLifetime = mean(lifetime_s,3);
meanThroughput = mean(throughput_kbps,3);
meanCollision = mean(collisionRate,3);
meanLatency = mean(latency_ms,3,'omitnan');
meanTotalEnergy = mean(totalEnergy_J,3);
meanResidualEnergy = mean(residualEnergy_J,3);
meanGenerated = mean(generatedCount,3);
meanDelivered = mean(deliveredCount,3);
meanDropped = mean(droppedCount,3);

%% 3A. STATISTICAL DISPERSION ACROSS PAIRED SIMULATION REPEATS
% Sample standard deviation (N-1 normalization) across the repeat dimension.
% With numberOfRepeats=3, SD is descriptive and should be interpreted
% cautiously; increase repetitions for stronger uncertainty assessment.
stdEnergy = std(energyPerStep,0,3);
stdPDR = std(pdr,0,3);
stdLifetime = std(lifetime_s,0,3);
stdThroughput = std(throughput_kbps,0,3);
stdCollision = std(collisionRate,0,3);
stdLatency = std(latency_ms,0,3,'omitnan');
stdTotalEnergy = std(totalEnergy_J,0,3);
stdResidualEnergy = std(residualEnergy_J,0,3);

% Density is shown in nodes/km^2; 100 m x 100 m = 0.01 km^2.
nodeDensity_km2 = nodeCounts/(cfg.areaSide_m^2)*1e6;

%% 4. FIGURE 14: AVERAGE ENERGY PER NODE PER STEP (MEAN +/- SD)
fig14 = figure('Color','w','Name','Average energy per step');
hold on; grid on; box on;
for p = 1:nP
    errorbar(nodeCounts,meanEnergy(:,p),stdEnergy(:,p),'o-', ...
        'LineWidth',1.8,'MarkerSize',7,'CapSize',8);
end
xlabel('Number of sensor nodes');
ylabel('Average energy consumed (J/node/step)');
title('Average energy per step (Mean \pm SD)');
legend(protocols,'Location','best');
set(gca,'FontSize',11);
saveFigurePair(fig14,outputFolder,'Figure14_Average_Energy_Per_Step',saveFigures);

%% 5. FIGURE 15: PACKET DELIVERY RATIO (MEAN +/- SD)
fig15 = figure('Color','w','Name','Packet delivery ratio');
hold on; grid on; box on;
for p = 1:nP
    errorbar(nodeCounts,100*meanPDR(:,p),100*stdPDR(:,p),'s-', ...
        'LineWidth',1.8,'MarkerSize',7,'CapSize',8);
end
xlabel('Number of sensor nodes');
ylabel('Packet delivery ratio (%)');
title('Packet delivery ratio (Mean \pm SD)');
ylim([0 105]);
legend(protocols,'Location','best');
set(gca,'FontSize',11);
saveFigurePair(fig15,outputFolder,'Figure15_Packet_Delivery_Ratio',saveFigures);

%% 6. FIGURE 16: ESTIMATED FIRST-NODE LIFETIME (MEAN +/- SD)
fig16 = figure('Color','w','Name','Estimated network lifetime');
hold on; grid on; box on;
for p = 1:nP
    errorbar(nodeCounts,meanLifetime(:,p)/3600,stdLifetime(:,p)/3600,'d-', ...
        'LineWidth',1.8,'MarkerSize',7,'CapSize',8);
end
xlabel('Number of sensor nodes');
ylabel('Estimated first-node lifetime (hours)');
title('Estimated network lifetime (Mean \pm SD)');
legend(protocols,'Location','best');
set(gca,'FontSize',11);
saveFigurePair(fig16,outputFolder,'Figure16_Estimated_Network_Lifetime',saveFigures);

%% 7. FIGURE 17: AGGREGATE NETWORK THROUGHPUT (MEAN +/- SD)
fig17 = figure('Color','w','Name','Network throughput');
hold on; grid on; box on;
for p = 1:nP
    errorbar(nodeCounts,meanThroughput(:,p),stdThroughput(:,p),'^-', ...
        'LineWidth',1.8,'MarkerSize',7,'CapSize',8);
end
xlabel('Number of sensor nodes');
ylabel('Aggregate application throughput (kbps)');
title('Network throughput (Mean \pm SD)');
legend(protocols,'Location','best');
set(gca,'FontSize',11);
saveFigurePair(fig17,outputFolder,'Figure17_Network_Throughput',saveFigures);

%% 8. FIGURE 18: COLLISION RATE VS NODE DENSITY (MEAN +/- SD)
fig18 = figure('Color','w','Name','Collision rate vs density');
hold on; grid on; box on;
for p = 1:nP
    errorbar(nodeDensity_km2,100*meanCollision(:,p),100*stdCollision(:,p),'v-', ...
        'LineWidth',1.8,'MarkerSize',7,'CapSize',8);
end
xlabel('Node density (nodes/km^2)');
ylabel('Collision rate (%)');
title('Collision rate (Mean \pm SD)');
legend(protocols,'Location','best');
set(gca,'FontSize',11);
saveFigurePair(fig18,outputFolder,'Figure18_Collision_Rate_vs_Density',saveFigures);

%% 9. FIGURE 19: END-TO-END LATENCY (MEAN +/- SD)
fig19 = figure('Color','w','Name','End-to-end latency comparison');
hold on; grid on; box on;
for p = 1:nP
    errorbar(nodeCounts,meanLatency(:,p),stdLatency(:,p),'o-', ...
        'LineWidth',1.8,'MarkerSize',7,'CapSize',8);
end
xlabel('Number of sensor nodes');
ylabel('Mean end-to-end latency (ms)');
title('End-to-end latency (Mean \pm SD)');
legend(protocols,'Location','best');
set(gca,'FontSize',11);
saveFigurePair(fig19,outputFolder,'Figure19_End_to_End_Latency',saveFigures);

%% 10. EXPORT FAIR-COMPARISON SUMMARY AND RAW REPEAT DATA
numRows = nN*nP;
NodeCount = zeros(numRows,1);
NodeDensity_nodes_km2 = zeros(numRows,1);
Protocol = cell(numRows,1);
AverageEnergy_J_per_node_step = zeros(numRows,1);
TotalNetworkEnergy_J = zeros(numRows,1);
ResidualNetworkEnergy_J = zeros(numRows,1);
PDR_percent = zeros(numRows,1);
EstimatedFirstNodeLifetime_hours = zeros(numRows,1);
Throughput_kbps = zeros(numRows,1);
CollisionRate_percent = zeros(numRows,1);
MeanLatency_ms = nan(numRows,1);
MeanGeneratedPackets = zeros(numRows,1);
MeanDeliveredPackets = zeros(numRows,1);
MeanDroppedPackets = zeros(numRows,1);
SD_Energy_J_per_node_step = zeros(numRows,1);
SD_PDR_percent = zeros(numRows,1);
SD_EstimatedFirstNodeLifetime_hours = zeros(numRows,1);
SD_Throughput_kbps = zeros(numRows,1);
SD_CollisionRate_percent = zeros(numRows,1);
SD_Latency_ms = nan(numRows,1);

row = 0;
for iN = 1:nN
    for iP = 1:nP
        row = row+1;
        NodeCount(row) = nodeCounts(iN);
        NodeDensity_nodes_km2(row) = nodeDensity_km2(iN);
        Protocol{row} = protocols{iP};
        AverageEnergy_J_per_node_step(row) = meanEnergy(iN,iP);
        TotalNetworkEnergy_J(row) = meanTotalEnergy(iN,iP);
        ResidualNetworkEnergy_J(row) = meanResidualEnergy(iN,iP);
        PDR_percent(row) = 100*meanPDR(iN,iP);
        EstimatedFirstNodeLifetime_hours(row) = meanLifetime(iN,iP)/3600;
        Throughput_kbps(row) = meanThroughput(iN,iP);
        CollisionRate_percent(row) = 100*meanCollision(iN,iP);
        MeanLatency_ms(row) = meanLatency(iN,iP);
        MeanGeneratedPackets(row) = meanGenerated(iN,iP);
        MeanDeliveredPackets(row) = meanDelivered(iN,iP);
        MeanDroppedPackets(row) = meanDropped(iN,iP);
        SD_Energy_J_per_node_step(row) = stdEnergy(iN,iP);
        SD_PDR_percent(row) = 100*stdPDR(iN,iP);
        SD_EstimatedFirstNodeLifetime_hours(row) = stdLifetime(iN,iP)/3600;
        SD_Throughput_kbps(row) = stdThroughput(iN,iP);
        SD_CollisionRate_percent(row) = 100*stdCollision(iN,iP);
        SD_Latency_ms(row) = stdLatency(iN,iP);
    end
end

ResultsTable = table(NodeCount,NodeDensity_nodes_km2,Protocol, ...
    AverageEnergy_J_per_node_step,TotalNetworkEnergy_J, ...
    ResidualNetworkEnergy_J,PDR_percent, ...
    EstimatedFirstNodeLifetime_hours,Throughput_kbps, ...
    CollisionRate_percent,MeanLatency_ms,MeanGeneratedPackets, ...
    MeanDeliveredPackets,MeanDroppedPackets, ...
    SD_Energy_J_per_node_step,SD_PDR_percent, ...
    SD_EstimatedFirstNodeLifetime_hours,SD_Throughput_kbps, ...
    SD_CollisionRate_percent,SD_Latency_ms);

writetable(ResultsTable,fullfile(outputFolder,'ST_MAC_Fair_Performance_Summary.csv'));

% Dedicated, compact mean +/- SD report for manuscript preparation.
MeanSDTable = ResultsTable(:,{'NodeCount','NodeDensity_nodes_km2','Protocol', ...
    'AverageEnergy_J_per_node_step','SD_Energy_J_per_node_step', ...
    'PDR_percent','SD_PDR_percent', ...
    'EstimatedFirstNodeLifetime_hours','SD_EstimatedFirstNodeLifetime_hours', ...
    'Throughput_kbps','SD_Throughput_kbps', ...
    'CollisionRate_percent','SD_CollisionRate_percent', ...
    'MeanLatency_ms','SD_Latency_ms'});
writetable(MeanSDTable,fullfile(outputFolder,'ST_MAC_Mean_SD_Results.csv'));
save(fullfile(outputFolder,'ST_MAC_Fair_Performance_All_Results.mat'), ...
    'cfg','nodeCounts','nodeDensity_km2','protocols','numberOfRepeats', ...
    'energyPerStep','pdr','lifetime_s','throughput_kbps','collisionRate', ...
    'latency_ms','generatedCount','deliveredCount','droppedCount', ...
    'totalEnergy_J','residualEnergy_J','meanEnergy','meanPDR', ...
    'meanLifetime','meanThroughput','meanCollision','meanLatency', ...
    'stdEnergy','stdPDR','stdLifetime','stdThroughput','stdCollision', ...
    'stdLatency','ResultsTable','MeanSDTable');

fprintf('\n========== FAIR NETWORK COMPARISON SUMMARY ==========\n');
disp(ResultsTable);
fprintf('Results saved in: %s\n',outputFolder);
fprintf('Statistical reporting: mean and sample SD across %d paired repeats.\n', ...
    numberOfRepeats);
fprintf('Mean +/- SD results exported to ST_MAC_Mean_SD_Results.csv.\n');
fprintf(['NOTE: first-node lifetime is extrapolated from the observed ' ...
    'per-node energy drain, not measured to actual battery exhaustion.\n']);
fprintf(['NOTE: TDMA and EATDMA are collision-free by their slot/reservation ' ...
    'assumptions; ST-MAC contention collisions are explicitly modelled.\n']);
fprintf('\nBOTH Q-LEARNING AND NETWORK STAGES COMPLETED.\n');

%% ========================== LOCAL FUNCTIONS ===========================

function out = simulateOneProtocol(protocol,N,density,cfg,arrivalMask,phyUniform)
% Common per-node battery/radio model for all protocols.
% Each node's energy is accounted for in every time step as sleep, idle,
% transmit, receive/ACK, and wake-up transition energy.
%
% ST-MAC: learned adaptive active/sleep cycle; all awake queued nodes
% contend in the same channel opportunity.
% TDMA: one node is assigned a fixed round-robin slot per active period.
% EATDMA: one queued node is selected by event-aware round-robin reservation.
% TDMA/EATDMA collision-free behaviour is a scheduling assumption.

dt = cfg.dt_s;
nSteps = size(arrivalMask,2);
tEnd = nSteps*dt;
packetBits = cfg.packetSize_bytes*8;
slotSteps = max(1,round(cfg.activeDuration_s/dt));

remainingEnergy = cfg.initialEnergy_J*ones(N,1);
energyConsumed = zeros(N,1);
queues = cell(N,1);
for node = 1:N
    queues{node} = zeros(1,0);
end

% ST-MAC active-period state
activeUntil = -inf(N,1);
nextWake = zeros(N,1);
stMacSentThisCycle = false(N,1);
tdmaSentThisSlot = false;
eatdmaSentThisReservation = false;

generatedPackets = 0;
deliveredPackets = 0;
droppedPackets = 0;
transmissionAttempts = 0;
collidedAttempts = 0;
latencySum_s = 0;
latencyCount = 0;
roundRobinPointer = 1;
eatdmaActiveNode = 0;
eatdmaActiveUntil = -inf;

phyPER = min(cfg.maxPacketError, ...
    cfg.basePacketError + cfg.densityPERfactor*density);

for k = 1:nSteps
    now = (k-1)*dt;

    %% A. Apply the SAME exogenous traffic trace to every protocol.
    newArrivals = find(arrivalMask(:,k));
    for q = 1:numel(newArrivals)
        node = newArrivals(q);
        generatedPackets = generatedPackets+1;
        if numel(queues{node}) < cfg.queueCapacity
            queues{node}(end+1) = now; %#ok<AGROW>
        else
            droppedPackets = droppedPackets+1;
        end
    end

    %% B. Select protocol-specific awake and transmitting nodes.
    awakeNodes = [];
    txNodes = [];
    newlyAwake = [];

    switch upper(protocol)
        case 'ST-MAC'
            % Start a new active period when the learned sleep timer expires.
            for node = 1:N
                if remainingEnergy(node)>0 && now>=nextWake(node) && ...
                        now>=activeUntil(node)
                    batteryRatio = max(0,min(1, ...
                        remainingEnergy(node)/cfg.initialEnergy_J));
                    state = find(batteryRatio>=cfg.stateEdges(1:end-1) & ...
                        batteryRatio<cfg.stateEdges(2:end),1,'first');
                    if isempty(state), state=5; end
                    sleepDuration = cfg.sleepActions_s(state);
                    activeUntil(node) = now+cfg.activeDuration_s;
                    nextWake(node) = activeUntil(node)+sleepDuration;
                    stMacSentThisCycle(node) = false;
                    newlyAwake(end+1) = node; %#ok<AGROW>
                end
            end
            awakeNodes = find(now<activeUntil & remainingEnergy>0);
            if ~isempty(awakeNodes)
                hasPacket = false(size(awakeNodes));
                for q = 1:numel(awakeNodes)
                    node = awakeNodes(q);
                    hasPacket(q) = ~isempty(queues{node}) && ...
                        ~stMacSentThisCycle(node);
                end
                txNodes = awakeNodes(hasPacket);
            end

        case 'TDMA'
            % Fixed round-robin slot allocation. Each node receives one
            % active period every N active periods; all others sleep.
            slotNumber = floor((k-1)/slotSteps);
            scheduledNode = mod(slotNumber,N)+1;
            if remainingEnergy(scheduledNode)>0
                awakeNodes = scheduledNode;
                if mod(k-1,slotSteps)==0
                    newlyAwake = scheduledNode;
                    tdmaSentThisSlot = false;
                end
                if ~isempty(queues{scheduledNode}) && ~tdmaSentThisSlot
                    txNodes = scheduledNode;
                end
            end

        case 'EATDMA'
            % Event-aware centralized reservation: one queued node receives
            % an active period of the same configured duration as ST-MAC.
            % The node remains awake for that reservation; other nodes sleep.
            if eatdmaActiveNode==0 || now>=eatdmaActiveUntil || ...
                    remainingEnergy(eatdmaActiveNode)<=0 || ...
                    isempty(queues{eatdmaActiveNode})
                eatdmaActiveNode = 0;
                queueNonempty = false(N,1);
                for node = 1:N
                    queueNonempty(node) = ~isempty(queues{node}) && ...
                        remainingEnergy(node)>0;
                end
                queuedNodes = find(queueNonempty);
                if ~isempty(queuedNodes)
                    idx = find(queuedNodes>=roundRobinPointer,1,'first');
                    if isempty(idx), idx=1; end
                    eatdmaActiveNode = queuedNodes(idx);
                    eatdmaActiveUntil = now+cfg.activeDuration_s;
                    newlyAwake = eatdmaActiveNode;
                    eatdmaSentThisReservation = false;
                    roundRobinPointer = mod(eatdmaActiveNode,N)+1;
                end
            end
            if eatdmaActiveNode>0 && now<eatdmaActiveUntil && ...
                    remainingEnergy(eatdmaActiveNode)>0
                awakeNodes = eatdmaActiveNode;
                if ~isempty(queues{eatdmaActiveNode}) && ...
                        ~eatdmaSentThisReservation
                    txNodes = eatdmaActiveNode;
                end
            end

        otherwise
            error('Unknown protocol: %s',protocol);
    end

    %% C. Account for every node's radio state consistently.
    % All nodes begin the step in SLEEP. Awake nodes are charged IDLE for
    % the full step. TX and ACK intervals replace the corresponding idle
    % intervals, rather than being added on top of them.
    stepEnergy = cfg.Psleep_W*dt*ones(N,1);
    if ~isempty(awakeNodes)
        stepEnergy(awakeNodes) = cfg.Pidle_W*dt;
    end

    % Same wake-transition energy is charged for each newly activated node,
    % regardless of protocol.
    if ~isempty(newlyAwake)
        stepEnergy(newlyAwake) = stepEnergy(newlyAwake)+cfg.wakeupEnergy_J;
    end

    %% D. Resolve channel access and packet outcomes.
    successfulNode = [];
    if ~isempty(txNodes)
        % At most one packet attempt is permitted per active cycle/slot/
        % reservation. This preserves the same TX/RX/idle time fractions
        % used by the Q-learning energy model.
        switch upper(protocol)
            case 'ST-MAC'
                stMacSentThisCycle(txNodes) = true;
            case 'TDMA'
                tdmaSentThisSlot = true;
            case 'EATDMA'
                eatdmaSentThisReservation = true;
        end
        transmissionAttempts = transmissionAttempts+numel(txNodes);

        if numel(txNodes)>1
            % Simultaneous ST-MAC contenders collide. TDMA/EATDMA schedulers
            % select at most one transmitter by construction.
            collidedAttempts = collidedAttempts+numel(txNodes);
        else
            node = txNodes(1);
            if phyUniform(k)>=phyPER
                successfulNode = node;
            end
        end

        % Each attempt consumes TX energy even if it collides or suffers
        % a PHY error. Replace idle energy for the TX and (if successful)
        % ACK-reception durations.
        for q = 1:numel(txNodes)
            node = txNodes(q);
            txT = min(cfg.txDuration_s,dt);
            stepEnergy(node) = stepEnergy(node) + ...
                (cfg.Ptx_W-cfg.Pidle_W)*txT;
        end

        if ~isempty(successfulNode)
            node = successfulNode(1);
            ackT = min(cfg.ackDuration_s,max(0,dt-cfg.txDuration_s));
            stepEnergy(node) = stepEnergy(node) + ...
                (cfg.Prx_W-cfg.Pidle_W)*ackT;

            generationTime = queues{node}(1);
            queues{node}(1) = [];
            deliveredPackets = deliveredPackets+1;
            latencySum_s = latencySum_s+(now-generationTime)+ ...
                cfg.txDuration_s+cfg.ackDuration_s;
            latencyCount = latencyCount+1;
        end
    end

    %% E. Apply battery depletion without allowing negative energy.
    stepEnergy = max(stepEnergy,0);
    stepEnergy = min(stepEnergy,remainingEnergy);
    energyConsumed = energyConsumed+stepEnergy;
    remainingEnergy = max(0,remainingEnergy-stepEnergy);
end

%% F. Calculate metrics from common definitions.
totalEnergy = sum(energyConsumed);
totalResidualEnergy = sum(remainingEnergy);
energyPerNodePerStep = totalEnergy/(N*nSteps);

if generatedPackets>0
    packetDeliveryRatio = deliveredPackets/generatedPackets;
else
    packetDeliveryRatio = 0;
end

throughput = deliveredPackets*packetBits/tEnd/1000;

if transmissionAttempts>0
    collisionFraction = collidedAttempts/transmissionAttempts;
else
    collisionFraction = 0;
end

if latencyCount>0
    meanLatency = 1000*latencySum_s/latencyCount;
else
    meanLatency = NaN;
end

% Estimated first-node-death lifetime based on each node's measured average
% energy drain over the evaluation interval. This is an extrapolation.
perNodePower = energyConsumed/tEnd;
positiveDrain = perNodePower(perNodePower>0);
if isempty(positiveDrain)
    estimatedLifetime = Inf;
else
    estimatedLifetime = cfg.initialEnergy_J/max(positiveDrain);
end

out.energyPerNodePerStep_J = energyPerNodePerStep;
out.PDR = packetDeliveryRatio;
out.estimatedFirstNodeLifetime_s = estimatedLifetime;
out.throughput_kbps = throughput;
out.collisionRate = collisionFraction;
out.meanLatency_ms = meanLatency;
out.generatedPackets = generatedPackets;
out.deliveredPackets = deliveredPackets;
out.droppedPackets = droppedPackets;
out.transmissionAttempts = transmissionAttempts;
out.collidedAttempts = collidedAttempts;
out.totalEnergy_J = totalEnergy;
out.totalResidualEnergy_J = totalResidualEnergy;
out.remainingEnergy_J = remainingEnergy;
out.energyConsumedPerNode_J = energyConsumed;
end

function saveFigurePair(fig,folder,baseName,doSave)
if ~doSave, return; end
pngPath = fullfile(folder,[baseName '.png']);
pdfPath = fullfile(folder,[baseName '.pdf']);
try
    exportgraphics(fig,pngPath,'Resolution',300);
    exportgraphics(fig,pdfPath,'ContentType','vector');
catch
    print(fig,pngPath,'-dpng','-r300');
    print(fig,pdfPath,'-dpdf','-painters');
end
end
