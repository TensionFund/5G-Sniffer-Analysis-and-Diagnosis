% one_click_diagnose_no_toolbox.m
% Toolbox-free combined diagnosis for sniffer
clearvars; close all; clc;

% ---------- User-configurable ----------
fs = 23040000;           % sampling rate (Hz) used by your sniffer config
Ltpl_expected = 27559;   % expected template length in samples
topN = 20;               % how many top events to attempt per-event derotation test
% ---------------------------------------

% helper: safe nanmedian if not present
if ~exist('nanmedian','builtin')
    nanmedian = @(x) median(x(~isnan(x)));
end

fprintf('Loading resampled IQ (resampled_complex_vec.mat expects variable x_resamp)...\n');
if exist('resampled_complex_vec.mat','file')
    s = load('resampled_complex_vec.mat');
    fn = fieldnames(s);
    x = s.(fn{1});
    fprintf('Loaded %s with length %d samples\n', fn{1}, length(x));
else
    error('resampled_complex_vec.mat not found in cwd. Put x_resamp in that file.');
end

% Spectrogram + global FFT peak
Nwant = min(length(x), 2^20);
x0 = x(1:Nwant);
fprintf('Computing spectrogram and global FFT peak (first %d samples)...\n', Nwant);
winsz=4096; noverlap = round(0.8*winsz);
figure('Name','Spectrogram'); spectrogram(x0,winsz,noverlap,winsz,fs,'yaxis'); title('Spectrogram (first block)');
drawnow;

% Global FFT peak
Nfft = 2^20;
X = fftshift(fft(x0, Nfft));
f = (-Nfft/2:Nfft/2-1)*(fs/Nfft);
[~,k] = max(abs(X));
fprintf('Strongest spectral peak at %.1f Hz offset from center\n', f(k));

% RMS quick ratio
rms_full = sqrt(mean(abs(x0).^2));
noise_chunk = x0(round(0.8*Nwant):round(0.95*Nwant));
rms_noise = sqrt(mean(abs(noise_chunk).^2));
snr_proxy_db = 20*log10(rms_full / rms_noise);
fprintf('RMS-based ratio (proxy SNR) = %.2f dB\n', snr_proxy_db);

% Load matched-filter CSV if available
csv_candidates = {'mf_with_derot_corr.csv','matched_filter_final_v2.csv','mf_refined_cfo.csv','mf_with_stacked_template.csv','mf_derotated_corrs.csv','mf_with_derot_corr.csv'};
csvfile = '';
for c=1:numel(csv_candidates)
    if exist(csv_candidates{c},'file')
        csvfile = csv_candidates{c};
        break;
    end
end

if isempty(csvfile)
    warning('No matched-filter CSV found in cwd. Skipping event stats and CFO histogram.');
else
    fprintf('Loading matched-filter CSV: %s\n', csvfile);
    T = readtable(csvfile);
    % find plausible CFO column
    cfo_cols = intersect({'cfo_hz','chosen_cfo','chosen_cfo_hz','refined_cfo_fft','refined_cfo_slope'}, T.Properties.VariableNames);
    if isempty(cfo_cols)
        % try any numeric column with "cfo" substring
        vc = T.Properties.VariableNames;
        for i=1:numel(vc)
            if contains(lower(vc{i}),'cfo'), cfo_cols{end+1} = vc{i}; end
        end
    end
    if ~isempty(cfo_cols)
        cfo = T.(cfo_cols{1});
        fprintf('CFO column used: %s\n', cfo_cols{1});
        fprintf('CFO stats (abs) median=%.1f Hz, frac>=10kHz=%.2f\n', median(abs(cfo),'omitnan'), mean(abs(cfo)>=1e4,'omitnan'));
        figure('Name','CFO histogram'); histogram(abs(cfo)/1e3,80); xlabel('abs(CFO) kHz'); title('CFO histogram (abs)'); xlim([0 max(50, max(abs(cfo)/1e3))]);
    else
        warning('No CFO-like column found in CSV.');
    end
    % correlation stats
    if ismember('corr_peak',T.Properties.VariableNames)
        fprintf('Median corr_peak = %.6f\n', nanmedian(T.corr_peak));
    end
end

% Try loading stacked template: pick the best candidate file and variable
tpl_loaded = false;
tpl_var = [];
cand_tmpl_files = {'stacked_template_refined.mat','stacked_template_v2.mat','stacked_template.mat','stacked_template.mat'};
for cf = cand_tmpl_files
    fn = cf{1};
    if exist(fn,'file')
        s = load(fn);
        vars = fieldnames(s);
        % choose any vector longer than 1000 samples (likely time template)
        for v=1:numel(vars)
            vv = s.(vars{v});
            if isvector(vv) && numel(vv) >= min(1000,Ltpl_expected)
                tpl = vv(:);
                fprintf('Loaded template from "%s" variable "%s" (len=%d)\n', fn, vars{v}, numel(tpl));
                tpl_loaded = true; tpl_var = vars{v}; break;
            end
        end
    end
    if tpl_loaded, break; end
end

if ~tpl_loaded
    warning('No full-length template found among candidate MATs. If you want a template created from top events, run build_template_from_events.m (provided).');
else
    % quick sanity: length match
    fprintf('Template length = %d (expected ~%d)\n', length(tpl), Ltpl_expected);
end

% If we have both matched CSV with start_sample and a template, try per-event derotation test on topN events
if exist('T','var') && exist('tpl','var')
    % find column naming for start sample
    start_col = '';
    if ismember('start_sample',T.Properties.VariableNames)
        start_col = 'start_sample';
    elseif ismember('start',T.Properties.VariableNames)
        start_col = 'start';
    elseif ismember('table_row',T.Properties.VariableNames) && ismember('start_sample',T.Properties.VariableNames)
        start_col = 'start_sample';
    end
    if isempty(start_col)
        warning('No start_sample column in matched CSV; cannot extract snippets automatically. You can supply start indices manually.');
    else
        % pick topN by corr_peak
        if ~ismember('corr_before',T.Properties.VariableNames) && ismember('corr_peak',T.Properties.VariableNames)
            [~, idx_sort] = sort(T.corr_peak,'descend');
        elseif ismember('corr_before',T.Properties.VariableNames)
            [~, idx_sort] = sort(T.corr_before,'descend');
        else
            idx_sort = (1:height(T))';
        end
        ntry = min(topN, height(T));
        fprintf('Performing per-event derotation test on top %d events (bounded FFT refine)\n', ntry);
        results = zeros(ntry,5); % beforeCorr, afterCorr, f_phase, f_fft_best, snr_db
        for k=1:ntry
            tidx = idx_sort(k);
            r = T.(start_col)(tidx);
            if r+length(tpl)-1 > length(x)
                fprintf('event %d: snippet out of bounds (r=%d). skipping\n', tidx, r);
                results(k,:) = NaN;
                continue;
            end
            snip = x(r:r+length(tpl)-1);
            [f_phase, f_fft_best, corr_before, corr_after, snr_db] = est_cfo_and_snr_for_event(snip, tpl, fs);
            results(k,:) = [corr_before, corr_after, f_phase, f_fft_best, snr_db];
            fprintf('event %d: corr_before=%.5f corr_after=%.5f f_phase=%.1f f_fft=%.1f snr_db=%.2f\n', tidx, corr_before, corr_after, f_phase, f_fft_best, snr_db);
        end
        save('one_click_derotation_results.mat','results','-v7.3');
    end
end

fprintf('Done. Files written: one_click_derotation_results.mat (if derotation ran).\n');
