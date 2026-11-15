%% CLMAC_Statistical_Comparison.m
% Post-process the raw 30-replication output from
% CL_MAC_ThreeProtocol_OptimizedAccess.m.
%
% Input:
%   CL_MAC_ThreeProtocol_OptimizedAccess_Results/CLMAC_raw_replications.csv
%
% Outputs:
%   CLMAC_Statistical_Comparison.csv
%   CLMAC_Statistical_Comparison.mat
%
% Uses paired comparisons because all protocols share the same replication
% seed / traffic and channel traces within each scenario.
%
% No toolbox is required: t critical values and paired t-test p-values are
% calculated from the regularized incomplete beta function (betainc).
%
% IMPORTANT:
% This script calculates statistics from the actual replication-level data.
% It does not reconstruct SD, confidence intervals, or p-values from means.

clear; clc;

inDir = 'CL_MAC_ThreeProtocol_OptimizedAccess_Results';
rawFile = fullfile(inDir,'CLMAC_raw_replications.csv');
if ~isfile(rawFile)
    error(['Cannot find %s. Run the simulation first and place this ' ...
           'post-processing script in the same working directory.'],rawFile);
end

Raw = readtable(rawFile);
required = {'Scenario','Protocol','Replicate','PDR','Throughput_Mbps', ...
            'LossRatio','MeanDelay_s','EnergyPerDelivered_J'};
for k=1:numel(required)
    if ~ismember(required{k},Raw.Properties.VariableNames)
        error('Required column "%s" is missing from the raw CSV.',required{k});
    end
end

scenarioOrder = {'Sparse_Bursty','Moderate_Load','High_Congestion', ...
    'High_UAV_Mobility','Heterogeneous_Battery', ...
    'Unreliable_Channel','Combined_Stress'};
protocolOrder = {'UAV-TDMA','UAV-EATDMA','CL-MAC (Proposed)'};

% Higher is better for PDR and throughput. Lower is better for all others.
metricNames = {'PDR','Throughput_Mbps','LossRatio', ...
               'MeanDelay_s','EnergyPerDelivered_J'};
metricLabels = {'PDR','Throughput_Mbps','LossRatio', ...
                'MeanDelay_s','EnergyPerDelivered_J'};

% Long-form group summary and paired-comparison result tables.
groupRows = cell(0,8);
testRows = cell(0,20);

for s=1:numel(scenarioOrder)
    sc = scenarioOrder{s};

    % Validate replication pairing and completeness for all three protocols.
    repSets = cell(1,3);
    for p=1:3
        ix = strcmp(string(Raw.Scenario),sc) & ...
             strcmp(string(Raw.Protocol),protocolOrder{p});
        rr = Raw(ix,:);
        repSets{p} = sort(rr.Replicate(:));
        if numel(repSets{p}) ~= 30
            error('%s / %s has %d rows; expected 30 replications.', ...
                sc,protocolOrder{p},numel(repSets{p}));
        end
        if numel(unique(repSets{p})) ~= 30
            error('%s / %s contains duplicate replication IDs.', ...
                sc,protocolOrder{p});
        end
    end
    if ~isequal(repSets{1},repSets{2}) || ~isequal(repSets{1},repSets{3})
        error('Replication IDs are not paired consistently in scenario %s.',sc);
    end

    % Group means, sample SDs, and two-sided 95%% t confidence intervals.
    for p=1:3
        ix = strcmp(string(Raw.Scenario),sc) & ...
             strcmp(string(Raw.Protocol),protocolOrder{p});
        rr = Raw(ix,:);
        for m=1:numel(metricNames)
            x = rr.(metricNames{m});
            x = x(isfinite(x));
            n = numel(x);
            mu = mean(x);
            sd = std(x,0);
            if n>1
                tc = localTCrit(0.05,n-1);
                half = tc*sd/sqrt(n);
            else
                half = NaN;
            end
            groupRows(end+1,:) = {sc,protocolOrder{p},metricLabels{m}, ...
                n,mu,sd,mu-half,mu+half}; %#ok<SAGROW>
        end
    end

    % Paired CL-MAC minus baseline comparisons, retaining every scenario.
    for b=1:2
        ixCL = strcmp(string(Raw.Scenario),sc) & ...
               strcmp(string(Raw.Protocol),protocolOrder{3});
        ixB  = strcmp(string(Raw.Scenario),sc) & ...
               strcmp(string(Raw.Protocol),protocolOrder{b});
        CL = sortrows(Raw(ixCL,:),'Replicate');
        BL = sortrows(Raw(ixB,:),'Replicate');

        for m=1:numel(metricNames)
            col = metricNames{m};
            x = CL.(col);
            y = BL.(col);
            valid = isfinite(x) & isfinite(y);
            x=x(valid); y=y(valid);
            n=numel(x);
            d=x-y; % signed raw difference: CL-MAC minus baseline
            md=mean(d);
            sdD=std(d,0);

            if n>1
                tc=localTCrit(0.05,n-1);
                half=tc*sdD/sqrt(n);
                ciLo=md-half; ciHi=md+half;
                tstat=md/(sdD/sqrt(n));
                pT=localTPValue(tstat,n-1);
            else
                ciLo=NaN; ciHi=NaN; tstat=NaN; pT=NaN;
            end

            % Relative improvement is oriented so positive always favours
            % CL-MAC: higher PDR/throughput; lower loss/delay/energy.
            baseMean=mean(y);
            clMean=mean(x);
            if abs(baseMean)>eps
                if ismember(col,{'PDR','Throughput_Mbps'})
                    rel=100*(clMean-baseMean)/abs(baseMean);
                else
                    rel=100*(baseMean-clMean)/abs(baseMean);
                end
            else
                rel=NaN;
            end

            % Approximate two-sided Wilcoxon signed-rank p-value, with
            % tie correction and continuity correction. No toolbox required.
            pW=localWilcoxonApprox(d);

            if ismember(col,{'PDR','Throughput_Mbps'})
                direction='Positive CL-Baseline favours CL-MAC';
            else
                direction='Negative CL-Baseline favours CL-MAC';
            end

            testRows(end+1,:)={sc,protocolOrder{3},protocolOrder{b}, ...
                col,n,mean(x),std(x,0),mean(y),std(y,0), ...
                md,sdD,ciLo,ciHi,rel,tstat,pT,pW, ...
                direction,mean(d>0),mean(d<0)}; %#ok<SAGROW>
        end
    end
end

GroupStats=cell2table(groupRows,'VariableNames', ...
    {'Scenario','Protocol','Metric','N','Mean','SD','CI95_Lower','CI95_Upper'});

PairedTests=cell2table(testRows,'VariableNames', ...
    {'Scenario','Proposed','Baseline','Metric','N_Paired', ...
     'CL_Mean','CL_SD','Baseline_Mean','Baseline_SD', ...
     'Mean_Difference_CL_minus_Baseline','SD_Paired_Difference', ...
     'PairedDiff_CI95_Lower','PairedDiff_CI95_Upper', ...
     'Relative_Improvement_pct','Paired_t_statistic','Paired_t_pvalue', ...
     'Wilcoxon_approx_pvalue','Difference_Direction', ...
     'Fraction_CL_higher','Fraction_CL_lower'});

outCSV1=fullfile(inDir,'CLMAC_Group_Mean_SD_95CI.csv');
outCSV2=fullfile(inDir,'CLMAC_Paired_Statistical_Tests.csv');
outMAT=fullfile(inDir,'CLMAC_Statistical_Comparison.mat');
writetable(GroupStats,outCSV1);
writetable(PairedTests,outCSV2);
save(outMAT,'Raw','GroupStats','PairedTests','scenarioOrder', ...
    'protocolOrder','metricNames');

fprintf('\nGROUP SUMMARY: MEAN, SD AND 95%% CI\n');
disp(GroupStats);
fprintf('\nPAIRED CL-MAC COMPARISONS\n');
disp(PairedTests);
fprintf('\nSaved:\n%s\n%s\n%s\n',outCSV1,outCSV2,outMAT);

%% Local functions: base-MATLAB statistical calculations
function tcrit=localTCrit(alpha,df)
% Two-sided Student-t critical value using inverse regularized beta.
x=betaincinv(alpha,df/2,0.5);
tcrit=sqrt(df*(1/x-1));
end

function p=localTPValue(t,df)
% Two-sided Student-t p-value.
x=df/(df+t^2);
p=betainc(x,df/2,0.5);
p=max(0,min(1,p));
end

function p=localWilcoxonApprox(d)
% Two-sided normal approximation to Wilcoxon signed-rank test.
% Zero differences are removed; tied absolute ranks receive average ranks.
d=d(isfinite(d) & d~=0);
n=numel(d);
if n==0
    p=1; return;
end
a=abs(d);
[as,ord]=sort(a);
r=zeros(n,1);
i=1;
tieSizes=[];
while i<=n
    j=i;
    while j<n && as(j+1)==as(i)
        j=j+1;
    end
    r(ord(i:j))=mean(i:j);
    if j>i
        tieSizes(end+1)=j-i+1; %#ok<AGROW>
    end
    i=j+1;
end
Wplus=sum(r(d>0));
mu=n*(n+1)/4;
tieTerm=sum(tieSizes.^3-tieSizes);
variance=n*(n+1)*(2*n+1)/24-tieTerm/48;
if variance<=0
    p=NaN; return;
end
z=(abs(Wplus-mu)-0.5)/sqrt(variance);
z=max(0,z);
p=erfc(z/sqrt(2));
p=max(0,min(1,p));
end
