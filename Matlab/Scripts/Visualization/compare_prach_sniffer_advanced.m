% compare_prach_sniffer_advanced.m
% Robust MATLAB script to compare gNB RX_PRACH events with a sniffer .fc32
% Designed for deterministic analysis (no trial-and-error).
% Edit PARAMETERS block below only.

clearvars; close all; clc;

%% ---------- PARAMETERS (Edit these) ----------
gnb_log_file = '/home/uob/work/log/gnb.log';                   % gNB log (contains RX_PRACH)
gnb_symbols_file = '/home/uob/work/log/phy_rx_symbols';         % gNB recorded symbol stream (binary) (optional)
sniffer_fc32_file = '/home/uob/Downloads/3rd sniffer file/prach_symbols_thread_2027986552778767634.fc32';         % sniffer capture (float32 I,Q)
output_csv = 'prach_sniffer_report_advanced.csv';

% Sample rates (Hz) -- set to match your environment
fs_gnb = 23.04e6;      % gNB symbols sample rate
fs_sniffer = 23.04e6;  % sniffer sample rate

% If you know absolute wall-clock time of first sniffer sample, set it.
% Otherwise leave NaT and script will try to align using symbols.
sniffer_start_datetime = NaT;

% Analysis parameters
energy_window_ms = 0.5;             % moving RMS window (ms)
detection_threshold_std = 5.0;      % used to compute coarse energy threshold (adaptive)
min_peak_prominence_db = 3;         % min peak prominence when calling findpeaks (dB)
max_coarse_align_seconds = 5;       % expected max offset between recordings (for decimated xcorr)
per_event_window_ms = [5 15];       % [pre_ms post_ms] around expected index for per-event inspection

% Safety / performance
max_log_size_bytes = 2e9;
progress_every = 200000;

% Optional PRACH preamble for matched-filtering (leave '' if not used)
prach_preamble_file = '';

%% ---------- Helper local function declarations ----------
% we place helper functions at the bottom to keep top-level flow clear

%% ---------- Parse gNB log ----------
fprintf('Parsing gNB log: %s\n', gnb_log_file);
finfo = dir(gnb_log_file);
if isempty(finfo), error('gNB log file not found: %s', gnb_log_file); end
if finfo.bytes > max_log_size_bytes
    fprintf('Warning: large gNB log (%.2f MB). Streaming parse...\n', finfo.bytes/1e6);
end

fid = fopen(gnb_log_file,'r');
if fid==-1, error('Cannot open gNB log: %s', gnb_log_file); end

ev_ts = {};
ev_offset = [];
ev_size = [];
det_ts = {};
det_rssi = [];
det_metric = [];

ln = 0;
while ~feof(fid)
    tline = fgetl(fid); ln = ln + 1;
    if ~ischar(tline), break; end
    if contains(tline,'RX_PRACH')
        try
            tokens = regexp(tline, '^(?<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3,6}).*RX_PRACH:.*offset=(?<offset>\d+)\s+size=(?<size>\d+)', 'names');
            if ~isempty(tokens)
                ev_ts{end+1,1} = try_parse_ts(tokens.ts);
                ev_offset(end+1,1) = double(str2double(tokens.offset));
                ev_size(end+1,1) = double(str2double(tokens.size));
            else
                tokens2 = regexp(tline,'^(?<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+).*RX_PRACH:.*offset=(?<offset>\d+).*size=(?<size>\d+)', 'names');
                if ~isempty(tokens2)
                    ev_ts{end+1,1} = try_parse_ts(tokens2.ts);
                    ev_offset(end+1,1) = double(str2double(tokens2.offset));
                    ev_size(end+1,1) = double(str2double(tokens2.size));
                end
            end
        catch
        end
    elseif contains(tline, 'PRACH:')
        try
            tokens = regexp(tline, '^(?<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+).*PRACH:.*rssi=(?<rssi>[-\d\.]+)dB.*detection_metric=(?<dmet>[-\d\.]+)', 'names');
            if ~isempty(tokens)
                det_ts{end+1,1} = try_parse_ts(tokens.ts);
                det_rssi(end+1,1) = str2double(tokens.rssi);
                det_metric(end+1,1) = str2double(tokens.dmet);
            end
        catch
        end
    end
    if mod(ln, progress_every)==0
        fprintf('  parsed %d lines ... found %d RX_PRACH events\n', ln, numel(ev_ts));
    end
end
fclose(fid);
fprintf('Finished parsing gNB log: %d lines, %d RX_PRACH events.\n', ln, numel(ev_ts));
if isempty(ev_ts), error('No RX_PRACH events found in gNB log.'); end

gnb_events = table();
gnb_events.timestamp = vertcat(ev_ts{:});
gnb_events.offset = ev_offset(:);
gnb_events.size = ev_size(:);

% attach detection metrics if present
if ~isempty(det_ts)
    det_ts_v = vertcat(det_ts{:});
    gnb_events.detection_metric = NaN(height(gnb_events),1);
    gnb_events.rssi = NaN(height(gnb_events),1);
    tol_sec = 0.01;
    for i=1:height(gnb_events)
        [d, idx] = min(abs(seconds(gnb_events.timestamp(i) - det_ts_v)));
        if d <= tol_sec
            gnb_events.detection_metric(i) = det_metric(idx);
            gnb_events.rssi(i) = det_rssi(idx);
        end
    end
else
    gnb_events.detection_metric = NaN(height(gnb_events),1);
    gnb_events.rssi = NaN(height(gnb_events),1);
end

fprintf('gNB events table ready: %d events.\n', height(gnb_events));

%% ---------- Load sniffer fc32 ----------
fprintf('Loading sniffer fc32: %s\n', sniffer_fc32_file);
sinfo = dir(sniffer_fc32_file);
if isempty(sinfo), error('Sniffer file not found: %s', sniffer_fc32_file); end

n_floats = floor(sinfo.bytes / 4);
if mod(n_floats,2)~=0
    n_floats = n_floats - 1;
    fprintf('Warning: odd number of floats in fc32 -> trimming last float\n');
end
Ns = n_floats/2;
fprintf('Sniffer file: %d complex samples (%.6f s @ %.2f MHz)\n', Ns, Ns/fs_sniffer, fs_sniffer/1e6);

use_memmap = sinfo.bytes > 200e6;
if use_memmap
    mm = memmapfile(sniffer_fc32_file, 'Format', {'single', [n_floats,1], 'data'});
    raw_vec = mm.Data.data;
    raw_vec = reshape(raw_vec,1,[]);
    sniffer_iq = double(raw_vec(1:2:end)) + 1j*double(raw_vec(2:2:end));
    clear raw_vec;
else
    fid = fopen(sniffer_fc32_file,'rb','l'); if fid==-1, error('Cannot open sniffer file'); end
    raw = fread(fid, n_floats, 'float32');
    fclose(fid);
    sniffer_iq = double(raw(1:2:end)) + 1j*double(raw(2:2:end));
    clear raw;
end
if isempty(sniffer_iq), error('Sniffer IQ empty after load.'); end

%% ---------- Optionally load gNB symbols ----------
use_gnb_symbols = false;
gnb_symbols_iq = [];
if exist(gnb_symbols_file,'file')
    try
        info_g = dir(gnb_symbols_file);
        n_floats_g = floor(info_g.bytes/4);
        if mod(n_floats_g,2)~=0, n_floats_g = n_floats_g - 1; end
        Ng = n_floats_g/2;
        fprintf('gNB symbols file: %d complex samples (%.6f s @ %.2f MHz)\n', Ng, Ng/fs_gnb, fs_gnb/1e6);
        if info_g.bytes > 200e6
            mmg = memmapfile(gnb_symbols_file, 'Format', {'single', [n_floats_g,1], 'data'});
            rawg = mmg.Data.data;
            gnb_symbols_iq = double(rawg(1:2:end)) + 1j*double(rawg(2:2:end));
            clear rawg;
        else
            fid = fopen(gnb_symbols_file,'rb','l'); rawg = fread(fid, n_floats_g, 'float32'); fclose(fid);
            gnb_symbols_iq = double(rawg(1:2:end)) + 1j*double(rawg(2:2:end));
            clear rawg;
        end
        use_gnb_symbols = true;
    catch ME
        warning('Could not read gNB symbols file: %s\nProceeding without it.\n', ME.message);
        use_gnb_symbols = false;
    end
else
    fprintf('gNB symbols file not present: %s\n', gnb_symbols_file);
end

%% ---------- Coarse alignment (if gNB symbols exist) ----------
delta_samples = NaN;
aligned_by = 'none';
if use_gnb_symbols
    fprintf('Computing decimated envelope coarse alignment...\n');
    wlen_env_sn = max(1, round((energy_window_ms/1000) * fs_sniffer));
    wlen_env_g = max(1, round((energy_window_ms/1000) * fs_gnb));
    env_sn = movmean(abs(sniffer_iq).^2, wlen_env_sn);
    env_g = movmean(abs(gnb_symbols_iq).^2, wlen_env_g);

    % decimate reasonably (target <= 1e6 samples for xcorr)
    target_max = 1e6;
    decim_sn = max(1, floor(length(env_sn)/target_max));
    decim_g = max(1, floor(length(env_g)/target_max));
    decim = max(decim_sn, decim_g);
    env_sn_d = env_sn(1:decim:end);
    env_g_d = env_g(1:decim:end);
    clear env_sn env_g;

    minLen = min(length(env_sn_d), length(env_g_d));
    if minLen < 512
        fprintf('Decimated envelopes too short (minLen=%d). Skipping coarse xcorr.\n', minLen);
    else
        env_sn_d = env_sn_d(1:minLen);
        env_g_d  = env_g_d(1:minLen);

        maxlag = min( floor(max_coarse_align_seconds * (fs_sniffer/decim)), minLen-1 );
        if maxlag < 1
            fprintf('Computed maxlag < 1 (maxlag=%d). Skipping coarse xcorr.\n', maxlag);
        else
            % normalized cross-correlation manually:
            a = env_sn_d - mean(env_sn_d);
            b = env_g_d - mean(env_g_d);
            c = xcorr(a, b, maxlag, 'none');
            denom = sqrt(sum(a.^2)*sum(b.^2));
            if denom > 0
                c = c / denom;
            end
            lags = -maxlag:maxlag;
            [~, imax] = max(abs(c));
            lag_decim = lags(imax);
            delta_samples = lag_decim * decim; % sniffer_index - gnb_index estimate
            aligned_by = 'symbols_xcorr';
            fprintf('Coarse alignment delta_samples = %d (%.6f s)\n', delta_samples, delta_samples/fs_sniffer);
        end
    end
else
    fprintf('No gNB symbols - cannot do symbols xcorr alignment. You must supply sniffer_start_datetime or symbol file.\n');
end

%% ---------- Map gNB offsets -> expected sniffer indices ----------
% create fields to store candidate indices and chosen index
gnb_events.sniffer_index_from_time = NaN(height(gnb_events),1);
gnb_events.sniffer_index_from_offset = NaN(height(gnb_events),1);
gnb_events.sniffer_index = NaN(height(gnb_events),1);

% time-based mapping if sniffer_start_datetime is known
if ~isnat(sniffer_start_datetime)
    gnb_events.sniffer_index_from_time = round( seconds(gnb_events.timestamp - sniffer_start_datetime) * fs_sniffer );
    fprintf('Computed time-based indices using provided sniffer_start_datetime.\n');
end

% offset-based mapping if delta available and gNB symbols present
if ~isnan(delta_samples) && use_gnb_symbols
    % gnb offset is sample index into gnb_symbols_iq
    gnb_events.sniffer_index_from_offset = gnb_events.offset + delta_samples;
    fprintf('Computed offset-based indices (using delta_samples).\n');
end

% choose best index (prefer offset, then time)
for i=1:height(gnb_events)
    idx_off = gnb_events.sniffer_index_from_offset(i);
    idx_time = gnb_events.sniffer_index_from_time(i);
    chosen = NaN;
    if ~isnan(idx_off) && idx_off>=1 && idx_off<=Ns
        chosen = round(idx_off);
    elseif ~isnan(idx_time) && idx_time>=1 && idx_time<=Ns
        chosen = round(idx_time);
    else
        chosen = NaN;
    end
    gnb_events.sniffer_index(i) = chosen;
end

n_in_range = sum(~isnan(gnb_events.sniffer_index));
fprintf('Events with valid sniffer_index inside sniffer file: %d of %d\n', n_in_range, height(gnb_events));

% If zero events inside sniffer, compute the nearest gap to help diagnosis
if n_in_range == 0
    fprintf('\n*** ALERT: zero gNB events map inside sniffer capture. Diagnostic below. ***\n');
    % If offset mapping exists, compute mapped sample range and compare
    if ~isnan(delta_samples) && use_gnb_symbols
        mapped_first = min(gnb_events.offset) + delta_samples;
        mapped_last = max(gnb_events.offset) + delta_samples;
        % distance to sniffer [1 Ns]
        dist_left = (1 - mapped_last);
        dist_right = (mapped_first - Ns);
        if mapped_first > Ns
            gap_samples = mapped_first - Ns;
            gap_sec = gap_samples / fs_sniffer;
            fprintf('Mapped gNB event region is after sniffer file by %.1f samples (%.6f s).\n', gap_samples, gap_sec);
        elseif mapped_last < 1
            gap_samples = 1 - mapped_last;
            gap_sec = gap_samples / fs_sniffer;
            fprintf('Mapped gNB region is before sniffer file by %.1f samples (%.6f s).\n', gap_samples, gap_sec);
        else
            fprintf('Mapped region overlaps partially but no individual event index fell inside - inspect per-event mapping.\n');
        end
    else
        fprintf('No delta_samples/time mapping available to diagnose overlap. Provide sniffer_start_datetime or gNB symbols file.\n');
    end
    fprintf('Suggest: verify sniffer capture window, sample rates, and sniffer_start_datetime (if available).\n\n');
    % still continue — maybe there are energy peaks we can correlate to a few gNB events
end

%% ---------- Sniffer power envelope + adaptive findpeaks ----------
winlen = max(1, round((energy_window_ms/1000)*fs_sniffer));
sniffer_power = movmean(abs(sniffer_iq).^2, winlen);
sniffer_db = 10*log10(sniffer_power + eps);

% compute an adaptive MinPeakHeight; if this ends up > max(db) reduce it
minPeakHeight = median(sniffer_db) + detection_threshold_std*std(sniffer_db);
if minPeakHeight >= max(sniffer_db)
    % fallback to a high percentile so we still find relative peaks in short captures
    minPeakHeight = prctile(sniffer_db, 95);
    fprintf('Adaptive MinPeakHeight pushed to 95th percentile: %.3f dB\n', minPeakHeight);
end

try
    [pk_vals, pk_locs] = findpeaks(sniffer_db, 'MinPeakHeight', minPeakHeight, 'MinPeakProminence', min_peak_prominence_db);
catch ME
    warning('findpeaks failed with MinPeakHeight - using relaxed detection. Message: %s', ME.message);
    [pk_vals, pk_locs] = findpeaks(sniffer_db, 'MinPeakProminence', min_peak_prominence_db);
end
fprintf('Sniffer automatic energy peak detector found %d peaks.\n', numel(pk_locs));

%% ---------- Per-event detailed check (REPLACEMENT LOOP with robust guards) ----------
Ns_total = length(sniffer_iq);
results = table((1:height(gnb_events))', gnb_events.timestamp, gnb_events.offset, gnb_events.size, ...
    gnb_events.rssi, gnb_events.detection_metric, gnb_events.sniffer_index, ...
    'VariableNames', {'idx','gnb_timestamp','gnb_offset','gnb_size','gnb_rssi','gnb_detection_metric','sniffer_index'});

% Add result columns
results.match = false(height(results),1);
results.reason = strings(height(results),1);
results.sniffer_peak_db = NaN(height(results),1);
results.sniffer_peak_idx = NaN(height(results),1);
results.time_shift_samples = NaN(height(results),1);
results.time_shift_ms = NaN(height(results),1);

% window sizes for per-event analysis
pre_s = round(per_event_window_ms(1)/1000 * fs_sniffer);
post_s = round(per_event_window_ms(2)/1000 * fs_sniffer);

% safety limits for image generation
max_saved_images = 200;   % <-- adjust as needed (reduce if MATLAB memory is tight)
saved_image_count = 0;

% error log file
errlog = 'compare_prach_errors.log';
ferr = fopen(errlog,'a');
if ferr~=-1
    fprintf(ferr, '=== Run started: %s ===\n', datestr(now,'yyyy-mm-dd HH:MM:SS'));
    fclose(ferr);
end

% Optional resume: if CSV exists, load and resume from next index
if exist(output_csv,'file')
    try
        prev = readtable(output_csv);
        if ismember('idx', prev.Properties.VariableNames)
            last_done = max(prev.idx);
            start_idx = last_done + 1;
            fprintf('Resuming from previous CSV: starting at event %d (last_done=%d)\n', start_idx, last_done);
        else
            start_idx = 1;
        end
    catch
        start_idx = 1;
    end
else
    start_idx = 1;
end

for i = start_idx:height(gnb_events)
    try
        sidx = gnb_events.sniffer_index(i);
        if isnan(sidx)
            if ~isnan(delta_samples) && use_gnb_symbols
                sidx = round(gnb_events.offset(i) + delta_samples);
            else
                results.reason(i) = "index_unknown";
                continue;
            end
        end

        start_s = max(1, sidx - pre_s);
        end_s = min(Ns_total, sidx + post_s);
        if start_s > end_s
            results.reason(i) = "index_out_of_sniffer_range";
            results.match(i) = false;
            continue;
        end

        % safe slice (force column vector)
        local_db = sniffer_db(start_s:end_s);
        local_db = local_db(:);
        if isempty(local_db)
            pmax = NaN; pidx = NaN; peak_global_idx = NaN;
        else
            % scalar-safe max
            [pmax, relidx] = max(local_db);  % pmax scalar, relidx scalar
            pidx = relidx;
            peak_global_idx = start_s + (pidx - 1);
        end

        % assign scalars
        results.sniffer_peak_db(i) = pmax;
        results.sniffer_peak_idx(i) = peak_global_idx;

        % coarse energy decision
        if ~isnan(pmax) && pmax >= minPeakHeight
            if ~isnan(peak_global_idx) && abs(peak_global_idx - sidx) <= max(pre_s, post_s)
                results.match(i) = true;
                if abs(peak_global_idx - sidx) <= round(0.002 * fs_sniffer)
                    results.reason(i) = "matched_by_energy";
                else
                    results.reason(i) = "matched_but_shifted_time";
                end
            else
                results.match(i) = true;
                results.reason(i) = "matched_but_far_shifted";
            end
        else
            raw_chunk = sniffer_iq(start_s:end_s);
            if isempty(raw_chunk) || all(abs(raw_chunk) < 1e-8)
                results.reason(i) = "zeros_or_truncated_data";
                results.match(i) = false;
            else
                results.reason(i) = "low_snr_or_filtered_out";
                results.match(i) = false;
            end
        end

        % precise local xcorr (if symbols provided)
        if use_gnb_symbols
            g_start = max(1, gnb_events.offset(i) - pre_s);
            g_end = min(length(gnb_symbols_iq), gnb_events.offset(i) + post_s);
            g_chunk = gnb_symbols_iq(g_start:g_end);
            s_chunk = sniffer_iq(start_s:end_s);

            if length(g_chunk) >= 32 && length(s_chunk) >= 32
                maxlag_local = min(round(0.01 * fs_sniffer), floor(min(length(s_chunk), length(g_chunk))/2));
                a = s_chunk - mean(s_chunk);
                b = g_chunk - mean(g_chunk);
                c_loc = xcorr(a, b, maxlag_local, 'none');
                denom = sqrt(sum(abs(a).^2) * sum(abs(b).^2));
                if denom > 0
                    c_loc = c_loc / denom;
                end
                lags_loc = -maxlag_local:maxlag_local;
                [cval, imaxloc] = max(abs(c_loc));
                lag_loc = lags_loc(imaxloc);

                if ~isnan(delta_samples)
                    expected_sniffer_idx = round(gnb_events.offset(i) + delta_samples);
                else
                    expected_sniffer_idx = sidx;
                end
                precise_shift = round((start_s - expected_sniffer_idx) + lag_loc);
                results.time_shift_samples(i) = precise_shift;
                results.time_shift_ms(i) = precise_shift / fs_sniffer * 1e3;

                if cval > 0.25
                    if ~results.match(i)
                        results.match(i) = true;
                        results.reason(i) = "xcorr_matched_low_energy";
                    else
                        if abs(precise_shift) > round(0.002 * fs_sniffer)
                            results.reason(i) = "matched_but_shifted_time_xcorr";
                        end
                    end
                end
            end
        else
            if ~isnan(sidx) && ~isnan(peak_global_idx)
                results.time_shift_samples(i) = peak_global_idx - sidx;
                results.time_shift_ms(i) = results.time_shift_samples(i) / fs_sniffer * 1e3;
            end
        end

        % Save spectrogram only for problems AND if we haven't exceeded the save limit
        do_save = (~results.match(i) || startsWith(results.reason(i), "matched_but_shifted")) && (saved_image_count < max_saved_images);
        if do_save
            % smaller spectrogram params -> much less memory
            window = 128; noverlap = round(window * 0.75); nfft = 256;
            chunk_iq = sniffer_iq(start_s:end_s);
            fname = sprintf('event_%04d_%s.png', i, datestr(gnb_events.timestamp(i),'yyyymmdd_HHMMSS'));
            fig = figure('Visible','off','Position',[100 100 900 400]);
            [S,F,T] = spectrogram(chunk_iq, window, noverlap, nfft, fs_sniffer);
            imagesc((T - per_event_window_ms(1)/1000), F/1e6, 20*log10(abs(S)+eps));
            axis xy; xlabel('Time (s) rel. to expected event'); ylabel('Freq (MHz)');
            title(sprintf('Event %d: %s (reason: %s, shift_ms=%.3f)', i, datestr(gnb_events.timestamp(i),'yyyy-mm-dd HH:MM:SS.FFF'), results.reason(i), results.time_shift_ms(i)));
            colorbar; hold on;
            xline((sidx - start_s)/fs_sniffer, 'r--', 'Expected');
            if ~isnan(peak_global_idx)
                xline((peak_global_idx - start_s)/fs_sniffer, 'g-', 'Peak');
            end
            saveas(fig, fname);
            close(fig);
            % clear large temp vars and force graphics flush
            clear S chunk_iq;
            drawnow limitrate;
            saved_image_count = saved_image_count + 1;
        end

        % periodically checkpoint results to CSV to avoid reprocessing in case of crash
        if mod(i,500) == 0
            writetable(results, output_csv);
            fprintf('Checkpoint saved at event %d\n', i);
        end

    catch ME
        % log the error and continue (do not crash)
        ferr = fopen(errlog,'a');
        if ferr~=-1
            fprintf(ferr, '%s - Event %d error: %s\n', datestr(now,'yyyy-mm-dd HH:MM:SS'), i, ME.message);
            fprintf(ferr, 'Stack:\n');
            for k=1:length(ME.stack)
                fprintf(ferr, '  %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
            end
            fclose(ferr);
        end
        fprintf('Warning: skipped event %d due to error: %s (see %s)\n', i, ME.message, errlog);
        % mark as errored for visibility
        results.reason(i) = "internal_error_skipped";
        results.match(i) = false;
        continue;
    end
end

% final save
results.gnb_rssi = gnb_events.rssi;
results.gnb_detection_metric = gnb_events.detection_metric;
writetable(results, output_csv);
fprintf('Saved detailed report to %s (final).\n', output_csv);


%% ---------- Save CSV and summary stats ----------
results.gnb_rssi = gnb_events.rssi;
results.gnb_detection_metric = gnb_events.detection_metric;
writetable(results, output_csv);
fprintf('Saved detailed report to %s\n', output_csv);

n_total = height(results);
n_matched = sum(results.match);
n_unmatched = n_total - n_matched;
fprintf('\nSummary: %d gNB events => %d matched, %d unmatched.\n', n_total, n_matched, n_unmatched);

valid_shifts = results.time_shift_samples(~isnan(results.time_shift_samples));
if ~isempty(valid_shifts)
    med_shift = median(valid_shifts);
    mean_shift = mean(valid_shifts);
    std_shift = std(valid_shifts);
    fprintf('Time-shift summary (samples): median=%d, mean=%.1f, std=%.1f => ms median=%.6f\n', med_shift, mean_shift, std_shift, med_shift/fs_sniffer*1e3);
    figure('Position',[200 200 800 300]);
    histogram(valid_shifts / fs_sniffer * 1e3, 40);
    xlabel('Time shift (ms)'); ylabel('Count'); title('Distribution of per-event time shifts (ms)');
end

if n_unmatched>0
    fprintf('Top unmatched (first 20):\n');
    tmp = results(~results.match, {'idx','gnb_timestamp','reason','sniffer_peak_db','time_shift_ms'});
    disp(tmp(1:min(20,height(tmp)),:));
end

fprintf('\nDiagnosis hints:\n');
fprintf('- If median time-shift is large and consistent => clock offset. Use delta_samples to realign.\n');
fprintf('- If shifts vary widely => jitter or variable buffering in capture (investigate OS scheduling / pipeline).\n');
fprintf('- If many "zeros_or_truncated_data" => capture truncated or wrong endianness.\n');
fprintf('- If many "low_snr_or_filtered_out" => sniffer center freq or front-end gain may be wrong. Check spectrograms saved as event_*.png\n');
fprintf('Done.\n');

%% ============================
%% Local helper functions
%% ============================
function dt = try_parse_ts(ts_str)
    % robust timestamp parse with fallbacks
    try
        dt = datetime(ts_str, 'InputFormat','yyyy-MM-dd''T''HH:mm:ss.SSSSSS');
    catch
        try
            dt = datetime(ts_str, 'InputFormat','yyyy-MM-dd''T''HH:mm:ss.SSSS');
        catch
            try
                dt = datetime(ts_str, 'InputFormat','yyyy-MM-dd HH:mm:ss.SSSSSS');
            catch
                try
                    dt = datetime(ts_str, 'InputFormat','yyyy-MM-dd''T''HH:mm:ss');
                catch
                    dt = NaT;
                end
            end
        end
    end
end
