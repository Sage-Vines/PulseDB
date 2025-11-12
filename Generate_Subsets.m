
clear; clc;
%% ==========================================================
%  Generate training, calibration, and testing subsets of PulseDB
%  ==========================================================
%  Note:
%  This version excludes all demographic fields (Age, Gender, Height, Weight, BMI)
%  for smaller file size and pure signal-based model training.
%  It includes both a visual waitbar and textual progress output.
%  ==========================================================

% ----------------------------------------------------------
% Locate segment files (Google Drive path)
% ----------------------------------------------------------
%MIMIC_Path = 'H:\.shortcut-targets-by-id\10mz4mfBo6NczPNbbjX0a9tAKQSMugBjV\PulseDB_v2_0\Segment_Files\PulseDB_MIMIC\';
Vital_Path = 'C:\Users\svines\Desktop\PulseDB\PulseDB\Segment_Files\PulseDB_Vital';

% ----------------------------------------------------------
% Locate info files (local path)
% ----------------------------------------------------------
Train_Info          = 'C:\Users\svines\Desktop\PulseDB\PulseDB\Info_Files\Train_Info.mat';
CalBased_Test_Info  = 'C:\Users\svines\Desktop\PulseDB\PulseDB\Info_Files\CalBased_Test_Info.mat';
CalFree_Test_Info   = 'C:\Users\svines\Desktop\PulseDB\PulseDB\Info_Files\CalFree_Test_Info.mat';
AAMI_Test_Info      = 'C:\Users\svines\Desktop\PulseDB\PulseDB\Info_Files\AAMI_Test_Info.mat';
AAMI_Cal_Info       = 'C:\Users\svines\Desktop\PulseDB\PulseDB\Info_Files\AAMI_Cal_Info.mat';

% ----------------------------------------------------------
% Output subset folder (local path)
% ----------------------------------------------------------
Subset_Save_Folder = 'C:\Users\svines\Desktop\PulseDB\PulseDB\Subset_Files';

% ----------------------------------------------------------
% Generate subsets
% ----------------------------------------------------------
Generate_Subset(Vital_Path, Vital_Path, Train_Info,         fullfile(Subset_Save_Folder,'Train_Subset_VitalOnly'))
Generate_Subset(Vital_Path, Vital_Path, CalBased_Test_Info, fullfile(Subset_Save_Folder,'CalBased_Test_Subset_VitalOnly'))
Generate_Subset(Vital_Path, Vital_Path, CalFree_Test_Info,  fullfile(Subset_Save_Folder,'CalFree_Test_Subset_VitalOnly'))
Generate_Subset(Vital_Path, Vital_Path, AAMI_Test_Info,     fullfile(Subset_Save_Folder,'AAMI_Test_Subset_VitalOnly'))
Generate_Subset(Vital_Path, Vital_Path, AAMI_Cal_Info,      fullfile(Subset_Save_Folder,'AAMI_Cal_Subset_VitalOnly'))

%% ==========================================================
%  Generate supplementary VitalDB-only subsets
%  ==========================================================
VitalDB_Train_Info         = 'C:\Users\svines\Desktop\PulseDB\PulseDB\Supplementary_Info_Files\VitalDB_Train_Info.mat';
VitalDB_CalBased_Test_Info = 'C:\Users\svines\Desktop\PulseDB\PulseDB\Supplementary_Info_Files\VitalDB_CalBased_Test_Info.mat';
VitalDB_CalFree_Test_Info  = 'C:\Users\svines\Desktop\PulseDB\PulseDB\Supplementary_Info_Files\VitalDB_CalFree_Test_Info.mat';
VitalDB_AAMI_Test_Info     = 'C:\Users\svines\Desktop\PulseDB\PulseDB\Supplementary_Info_Files\VitalDB_AAMI_Test_Info.mat';
VitalDB_AAMI_Cal_Info      = 'C:\Users\svines\Desktop\PulseDB\PulseDB\Supplementary_Info_Files\VitalDB_AAMI_Cal_Info.mat';

% Output folder for supplementary subsets
Supp_Save_Folder = 'C:\Users\svines\Desktop\PulseDB\PulseDB\Supplementary_Subset_Files';

Generate_Subset(Vital_Path, Vital_Path, VitalDB_Train_Info,         fullfile(Supp_Save_Folder,'VitalDB_Train_Subset_VitalOnly'))
Generate_Subset(Vital_Path, Vital_Path, VitalDB_CalBased_Test_Info, fullfile(Supp_Save_Folder,'VitalDB_CalBased_Test_Subset_VitalOnly'))
Generate_Subset(Vital_Path, Vital_Path, VitalDB_CalFree_Test_Info,  fullfile(Supp_Save_Folder,'VitalDB_CalFree_Test_Subset_VitalOnly'))
Generate_Subset(Vital_Path, Vital_Path, VitalDB_AAMI_Test_Info,     fullfile(Supp_Save_Folder,'VitalDB_AAMI_Test_Subset_VitalOnly'))
Generate_Subset(Vital_Path, Vital_Path, VitalDB_AAMI_Cal_Info,      fullfile(Supp_Save_Folder,'VitalDB_AAMI_Cal_Subset_VitalOnly'))


%% ==========================================================
%  Function: Generate_Subset (no demographics, Vital-only)
%  ==========================================================
function Generate_Subset(~, Vital_Path, Info_File_Path, Save_Name)
%% Retrieve segments from files using the Info file
Info = load(Info_File_Path);
Field = fieldnames(Info);
Info = Info.(Field{1});

% --- Keep only VitalDB subjects (ending in '1') ---
Info = Info(endsWith({Info.Subj_Name}, '1')); % Keep only VitalDB entries

Len = numel(Info);

% Pre-allocate memory (only required fields)
Subset.Subject = cell(Len,1);
Subset.Signals = zeros(Len,3,1250);
Subset.SBP = NaN(Len,1);
Subset.DBP = NaN(Len,1);

% ----------------------------------------------------------
% Locate unique subjects in the Info file
% ----------------------------------------------------------
Subjects = unique({Info.Subj_Name});
pos = 1;

% ----------------------------------------------------------
% Initialize both visual and textual progress indicators
% ----------------------------------------------------------
try
    title_str = sprintf('Gathering VitalDB Data For: %s', strrep(Save_Name, '\', '/'));
    f = waitbar(0, title_str);
    gui_mode = true;
    drawnow;
catch
    warning('GUI waitbar unavailable. Continuing with text-only progress updates.');
    gui_mode = false;
end

% ----------------------------------------------------------
% Main subject loop (VitalDB only)
% ----------------------------------------------------------
for i = 1:numel(Subjects)
    % Update waitbar visually if GUI mode is active
    if gui_mode
        try
            waitbar(i/numel(Subjects), f);
            drawnow;
        catch
            gui_mode = false;
            warning('Waitbar closed unexpectedly — switching to text-only mode.');
        end
    end

    % Textual update every 25 subjects
    if mod(i,25)==0
        fprintf('Processed %d of %d VitalDB subjects (%.1f%%)\n', i, numel(Subjects), 100*i/numel(Subjects));
    end

    % ------------------------------------------------------
    % Load and append subject data (Vital only)
    % ------------------------------------------------------
    Subj_Name = Subjects{i};
    Subj_ID = Subj_Name(1:7);

    % Force to VitalDB path (ignore MIMIC flag)
    Segment_Path = Vital_Path;

    Segments_File = load(fullfile(Segment_Path, Subj_ID));
    Subj_Segments = Segments_File.Subj_Wins;
    Selected_IDX = [Info(strcmp({Info.Subj_Name},Subj_Name)).Subj_SegIDX];
    
    for j = Selected_IDX
        % Skip indices that don't exist in this subject's segment file
        if j > numel(Subj_Segments)
            warning('⚠️ Skipping invalid index %d for subject %s (only %d segments available).', ...
                j, Subj_Name, numel(Subj_Segments));
            continue;
        end

        Segment = Subj_Segments(j);
        Subset.Subject{pos} = Subj_Name;
        Subset.Signals(pos,:,:) = [Segment.ECG_F, Segment.PPG_F, Segment.ABP_Raw]';
        Subset.SBP(pos) = Segment.SegSBP;
        Subset.DBP(pos) = Segment.SegDBP;
        pos = pos + 1;
    end
end

% ----------------------------------------------------------
% Finalize, save, and clean up
% ----------------------------------------------------------
if gui_mode
    waitbar(1,f,'Saving File...');
    drawnow;
end

fprintf('\nSaving VitalDB-only subset: %s\n', Save_Name);
save(Save_Name, 'Subset', '-v7.3');
fprintf('Saved successfully: %s\n\n', Save_Name);

if gui_mode
    delete(f);
end
end
