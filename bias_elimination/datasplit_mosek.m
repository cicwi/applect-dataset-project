function [Sequence,DefectPercentage] = datasplit_mosek(data_filename,P1,N1)
%DATASPLIT_MOSEK(data_filename,cvx_folder_path,P1,N1)
% Function splits the input data into two data subsets, first subset with
% P1 defect percentage, and N1 apples.
%
% This method uses the Mixed-Integer Quadratic Programming formulation of
% the data splitting problem.
%
% The function uses an external toolbox, CVX with the solver MOSEK.
% Information on how this is setup detailed in "Usage Details" in the cited
% paper.
%
% INPUT:
%   data_filename   : location of file where the apple defect pixel
%                   : information is saved.
%   P1              : Target percentage for the defect spread.
%   N1              : Number of apples to split for P1%.
%
% OUTPUT:
%   Sequence        : Structure that includes Subset1 and Subset2 sequences.
%   DefectPercentage: Corresponding defect percentages of the resulting
%                     Subset1 and Subset2 sequences.
% DEFAULTS:
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
    if exist('cvx','dir')~=7
        error('Must add CVX folder to the current path.')
    else
        addpath(genpath('cvx')); %
    end 
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

%% Prepare for the solver:

% Read the apple defect label data:
A = csvread(data_filename); % Columns in the order of: | "Apple Number" | "bitterpit" | "holes" | "rot" | "browning" |
A_per = A(:,2:5)./sum(A(:,2:5)); % We only need the defects! Convert data from pixel numbers to percentages.

[totalN,totalDefect]=size(A_per); % number of apples and defects in table (latter should be 4!).

if N1 > totalN
    error('The requested number of apples for either of the subset cannot be bigger than the total number of apples available.\nUser-input subset size: %d;\nTotal number of apples in the data = %d.\n',N1,totalN);
end

% target percentage for each defect (subset 1):
b = P1*ones(totalDefect,1);

% constraint:
W = ones(1,totalN);

%% CVX with MOSEK solver

% Start the CVX environment
cvx_begin quiet % remove "quiet" if you want output of solver
cvx_solver mosek % choose MOSEK as the solver
cvx_precision default % precision set to default (see CVX manual on precision)

% Create and solve the problem within CVX environment:
variable x(totalN) binary % x is a totalN-by-1 binary.
minimize(sum_square(A_per'*x - b));
subject to
W * x == N1; % apple constraint (number of apples is exactly N1).
cvx_end % Finalize CVX environment.

fprintf('CVX status: %s.\n',cvx_status);
fprintf('CVX solver CPU time (s): %4.2f.\n',cvx_cputime);
fprintf('CVX Objective best solution: %e.\n',cvx_optval);

%% Output results:

results_filename = 'mosek_split_results';

Sequence.Subset1 = find(x==1); % Find the indices of apples that are in the TEST subset.
Sequence.Subset2 = find(x==0); % Everything else is in the TRAINING subset.

DefectPercentage.Subset1 = sum(A_per(x==1,:));
DefectPercentage.Subset2 = sum(A_per(x==0,:));

fid = fopen([results_filename '.txt'],'w+');
fprintf(fid,'Input data file: %s.\n',data_filename);
fprintf(fid,'CVX status: %s.\n',cvx_status);
fprintf(fid,'CVX Objective best solution: %e.\n',cvx_optval);
fprintf(fid,'CVX solver CPU time (s): %4.2f.\n\n',cvx_cputime);
fprintf(fid,'Defect Percentage for Subset 1 = [%.4f,%.4f,%.4f,%.4f]\n',DefectPercentage.Subset1(1),DefectPercentage.Subset1(2),DefectPercentage.Subset1(3),DefectPercentage.Subset1(4));
fprintf(fid,'Sequence for Subset 1: \n');for i = 1:N1;fprintf(fid,'%d\t',Sequence.Subset1(i));end;fprintf(fid,'\n\n');
fprintf(fid,'Defect Percentage for Subset 2 = [%.4f,%.4f,%.4f,%.4f]\n',DefectPercentage.Subset2(1),DefectPercentage.Subset2(2),DefectPercentage.Subset2(3),DefectPercentage.Subset2(4));
fprintf(fid,'Sequence for Subset 2: \n');for i = 1:(totalN-N1);fprintf(fid,'%d\t',Sequence.Subset2(i));end;fprintf(fid,'\n\n');
fclose(fid);

csvwrite([results_filename '_subset1.csv'],A(x==1,:));
csvwrite([results_filename '_subset2.csv'],A(x==0,:));
fprintf('Results for split percentage %.2f are saved in %s.\n',P1,[results_filename '_subset1.csv']);
fprintf('Results for split percentage %.2f are saved in %s.\n',1-P1,[results_filename '_subset2.csv']);
end
