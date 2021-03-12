function [Sequence,DefectPercentage] = datasplit_empirical(data_filename,S,P1,N1)
%DATASPLIT_EMPIRICAL(data_filename,S,P1,N1)
% Function splits the input data into two data subsets, first subset with
% P1 defect percentage, and N1 apples.
%
% This method is a simple heuristic search method to empirically solve the
% data splitting problem. Function runs for S number of samples and finds
% the most successful sequence for the first subset and the corresponding
% defect percentage in the subset.
%
% INPUT:
%   data_filename   : location of file where the apple defect pixel
%                   : information is saved.
%   S               : Number of sample runs.
%   P1              : Target percentage for the defect spread.
%   N1              : Number of apples to split for P1%.
%
% OUTPUT:
%   Sequence        : Structure that includes Subset1 and Subset2 sequences.
%   DefectPercentage: Corresponding defect percentages of the resulting
%                     Subset1 and Subset2 sequences.
% DEFAULTS:
%   S  = 10000;
%   P1 = 0.2;
%   N1 = 20;
%
% Copyright (c) 2020 Sophia Bethany Coban
% Centrum Wiskunde & Informatica, Amsterdam, the Netherlands.
%
% Code is available via AppleCT Dataset Project;
% www.github.com/cicwi/applect-dataset-project
%
% Referenced paper: S.B. Coban, V. Andriiashen, P.S. Ganguly, et al.
% Parallel-beam X-ray CT datasets of apples with internal defects and label
% balancing for machine learning. 2020. www.arxiv.org/abs/2012.13346
%
% Dataset available via Zenodo; 10.5281/zenodo.4212301.

%% Checks:
% Check input arguments
if nargin<4
    N1 = 20;
elseif nargin<3
    P1 = 0.2;
elseif nargin<2
    S = 10000;
elseif nargin<1
    error('Not enough input arguments.\nA data file for defect pixel data is necessary, e.g. apple_defect_full.csv.')
end

% file format check:
if ~isfile(data_filename)
    error('No file with name "%s" exists.\nRemember to include the format in the filename.\nMake sure the file is included in the path, or enter the full path.',data_filename);
end

% Percentage value check:
if P1>1
    error('User-input percentage value needs to be between 0 and 1.');
end

%% Prepare for the algoritm:
A = csvread(data_filename); % Columns in the order of: | "Apple Number" | "bitterpit" | "holes" | "rot" | "browning" |
[totalN,~]=size(A); % number of apples and defects in table (latter should be 4!).
browning = sum(A(:,5));

%% Simple heuristic search with sample size S:
apple_seq = []; % initialise
j = 0;
t = tic;
for k = 1:S % S samples of shuffling + splitting
    
    randIndx = randperm(totalN);
    
    % Sum the first N1 apples in the random index:
    sum_browning = 0;
    for i = 1:94
        % we want to get to P1% browning with N1 apples:
        sum_browning = sum_browning + A(randIndx(i),5);
        
        % Is it over the P1% of the total browning pixels?
        if sum_browning >= browning*P1
            % Do we have N1 apples?
            if i == N1
                j = j+1;
                apple_seq(:,j) = randIndx';
            end
            % If we have browning over P1% but i < N1, we are way passed a
            % feasible solution, break here for the next sample run.
            break
        end
    end
    
end


%% Sort the successful sequences:
total_defect_sum = (sum(A(:,2:5)));

for k = 1:j
    A1(k,:) = sum(A(apple_seq(1:N1,k),2:5))./total_defect_sum;
    A2(k,:) = sum(A(apple_seq(N1+1:end,k),2:5))./total_defect_sum;
end

[~,sort_indx] = sort(sum(abs(A1-0.2),2));

Sequence.Subset1 = sort(apple_seq(1:N1,sort_indx(1,1)));
Sequence.Subset2 = sort(apple_seq(N1+1:end,sort_indx(1,1)));

DefectPercentage.Subset1 = A1(sort_indx(1,1),:);
DefectPercentage.Subset2 = A2(sort_indx(1,1),:);

t = toc(t)
%% Output results:

results_filename = 'empirical_split_results';

fid = fopen([results_filename '.txt'],'w+');
fprintf(fid,'Input data file: %s.\n',data_filename);
fprintf(fid,'Sample size: %d.\n',S);
fprintf(fid,'Successful runs: %d.\n',j);
fprintf(fid,'CPU time (s): %9.4f.\n\n',t);
fprintf(fid,'Defect Percentage for Subset 1 = [%.4f,%.4f,%.4f,%.4f]\n',DefectPercentage.Subset1(1),DefectPercentage.Subset1(2),DefectPercentage.Subset1(3),DefectPercentage.Subset1(4));
fprintf(fid,'Sequence for Subset 1: \n');for i = 1:N1;fprintf(fid,'%d\t',Sequence.Subset1(i));end;fprintf(fid,'\n\n');
fprintf(fid,'Defect Percentage for Subset 2 = [%.4f,%.4f,%.4f,%.4f]\n',DefectPercentage.Subset2(1),DefectPercentage.Subset2(2),DefectPercentage.Subset2(3),DefectPercentage.Subset2(4));
fprintf(fid,'Sequence for Subset 2: \n');for i = 1:(totalN-N1);fprintf(fid,'%d\t',Sequence.Subset2(i));end;fprintf(fid,'\n\n');
fclose(fid);

csvwrite([results_filename '_subset1.csv'],A(Sequence.Subset1,:));
csvwrite([results_filename '_subset2.csv'],A(Sequence.Subset2,:));
fprintf('Results for split percentage %.2f are saved in %s.\n',P1,[results_filename '_subset1.csv']);
fprintf('Results for split percentage %.2f are saved in %s.\n',1-P1,[results_filename '_subset2.csv']);

end
