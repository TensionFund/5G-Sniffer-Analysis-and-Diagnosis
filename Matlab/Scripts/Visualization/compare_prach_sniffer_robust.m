% compare_prach_sniffer_robust.m
% Robust, production-ready MATLAB script to compare gNB RX_PRACH events
% with a sniffer .fc32 capture and produce a detailed CSV + diagnostics.
%
% Put this file in your working dir and run. Edit only the PARAMETERS block.

clearvars; close all; clc;

%% ---------- PARAMETERS (Edit these) ----------
gnb_log_file = '/home/uob/work/log/gnb.log';                       % path to gnb log (contains RX_PRACH)
gnb_symbols_file = '/home/uob/work/log/phy_rx_symbols';            % gNB's recorded symbol stream (binary) (optional)
sniffer_fc32_file = '/home/uob/Downloads/sample.fc32';             % sniffer capture (float32 I,Q)
output_csv = 'prach_sniffer_report_robust.csv';

% Sample rates (Hz) -- adjust to match your environment
fs_gnb = 23.04e6;      % sample rate used by gNB's saved symbols
fs_sniffer = 23.04e6;  % sample rate used by sniffer capture

% Sniffer start time if known (wall clock when sniffer file first sample taken).
% If unknown, set to NaT and script will try to align with symbols xcorr.
sniffer_start_datetime = NaT; % e.g. datetime('2025-08-10T17:59:32.450426','InputFormat','yyyy-MM-dd''T''HH:mm:ss.SSSSSS');

% Detection / analysis parameters
energy_window_ms = 0.5;             % moving RMS window (ms)
detection_threshold_std = 5.0;      % peaks above median + this*std considered detections
min_peak_prominence_db = 3;         % for findpeaks on dB curve
max_coarse_align_seconds = 5;       % max expected offset between recordings (for xcorr)
per_event_window_ms = [5 15];       % [pre_ms post_ms] for per-event inspection windows

% Performance / safety
max_log_size_bytes = 2e9;           % if log > this, still parse but print warnings (2GB default)
progress_every = 200000;            % print every N lines when parsing

% Optionally specify a PRACH preamble waveform file for matched correlation (optional)
prach_preamble_file = ''; % e.g. '/path/to/preamble.fc32' (float32 I,Q), leave '' if not available

%% ---------- Helpers ----------
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
                dt = NaT;
            end
        end
    end
end

%% ---------- Parse gNB log (streaming, line-by-line) ----------
fprintf('Parsing gNB log (streaming): %s\n', gnb_log_file);
file_info = dir(gnb_log_file);
if isempty(file_info)
    error('gNB log file not found: %s', gnb_log_file);
end
if file_info.bytes > max_log_size_bytes
    fprintf('Warning: gNB log is large (%.2f MB). Streaming parse in progress...\n', file_info.bytes/1e6);
end

fid = fopen(gnb_log_file, 'r');
if fid==-1
    error('Cannot open gnb log for reading: %s', gnb_log_file);
end

% Pre-allocate arrays for speed (grow as needed)
ev_ts = {};
ev_offset = [];
ev_size = [];
det_ts = {};
det_rssi = [];
det_metric = [];

linecount = 0;
while ~feof(fid)
    tline = fgetl(fid);
    linecount = linecount + 1;
    if ~ischar(tline); break; end

    % Only inspect lines that contain keywords to speed things up
    if contains(tline, 'RX_PRACH')
        % Extract timestamp at start and offset/size
        % try to find ISO timestamp at beginning
        % Example: 2025-08-10T18:00:05.877483 [PHY     ] [I] ... RX_PRACH: sector=0 offset=4406856 size=1668
        try
            tokens = regexp(tline, '^(?<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3,6}).*RX_PRACH:.*offset=(?<offset>\d+)\s+size=(?<size>\d+)', 'names');
            if ~isempty(tokens)
                ev_ts{end+1,1} = try_parse_ts(tokens.ts);
                ev_offset(end+1,1) = str2double(tokens.offset);
                ev_size(end+1,1) = str2double(tokens.size);
            else
                % fallback: try looser regex
                tokens = regexp(tline, '^(?<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+).*RX_PRACH:.*offset=(?<offset>\d+).*size=(?<size>\d+)', 'names');
                if ~isempty(tokens)
                    ev_ts{end+1,1} = try_parse_ts(tokens.ts);
                    ev_offset(end+1,1) = str2double(tokens.offset);
                    ev_size(end+1,1) = str2double(tokens.size);
                end
            end
        catch
            % ignore parse failures for this line
        end
    elseif contains(tline, 'PRACH:')
        % Example: 2025-08-10T18:00:05.937682 [PHY     ] PRACH: rsi=1 rssi=-30.0dB detected_preambles=[{idx=18 ta=7.42us detection_metric=12.4}] t=235.9us
        try
            tokens = regexp(tline, '^(?<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+).*PRACH:.*rssi=(?<rssi>[-\d\.]+)dB.*detection_metric=(?<dmet>[-\d\.]+)', 'names');
            if ~isempty(tokens)
                det_ts{end+1,1} = try_parse_ts(tokens.ts);
                det_rssi(end+1,1) = str2double(tokens.rssi);
                det_metric(end+1,1) = str2double(tokens.dmet);
            end
        catch
            % ignore parse failures
        end
    end

    if mod(linecount, progress_every)==0
        fprintf('  parsed %d lines ... Found %d RX_PRACH events so far.\n', linecount, numel(ev_ts));
    end
end
fclose(fid);
fprintf('Finished parsing gNB log: %d lines read, %d RX_PRACH events found.\n', linecount, numel(ev_ts));

% Build table of gNB events
if isempty(ev_ts)
    error('No RX_PRACH events found in gNB log. Check the log format or path.');
end
gnb_events = table();
gnb_events.timestamp = vertcat(ev_ts{:});
gnb_events.offset = ev_offset(:);
gnb_events.size = ev_size(:);

% Attach detection metrics by proximity in time (if any)
if ~isempty(det_ts)
    det_ts_v = vertcat(det_ts{:});
    gnb_events.detection_metric = NaN(height(gnb_events),1);
    gnb_events.rssi = NaN(height(gnb_events),1);
    tol_sec = 0.01; % 10 ms tolerance to match a PRACH detection line
    for i=1:height(gnb_events)
        [dt, idx] = min(abs( seconds(gnb_events.timestamp(i) - det_ts_v) ));
        if dt <= tol_sec
            gnb_events.detection_metric(i) = det_metric(idx);
            gnb_events.rssi(i) = det_rssi(idx);
        end
    end
else
    gnb_events.detection_metric = NaN(height(gnb_events),1);
    gnb_events.rssi = NaN(height(gnb_events),1);
end

fprintf('gNB events table ready: %d events.\n', height(gnb_events));

%% ---------- Load sniffer fc32 (memory-map if large) ----------
fprintf('Loading sniffer fc32: %s\n', sniffer_fc32_file);
sinfo = dir(sniffer_fc32_file);
if isempty(sinfo)
    error('Sniffer file not found: %s', sniffer_fc32_file);
end

% compute number of complex samples (float32 interleaved I,Q)
n_floats = floor(sinfo.bytes / 4);
if mod(n_floats,2)~=0
    fprintf('Warning: fc32 file has odd number of floats -> trimming last float for safety.\n');
    n_floats = n_floats - 1;
end
Ns = n_floats/2;
fprintf('Sniffer file size %.2f MB -> %d complex samples (%.3f s at %.2f MHz)\n', sinfo.bytes/1e6, Ns, Ns/fs_sniffer, fs_sniffer/1e6);

% For efficiency, if file > 200 MB, use memmapfile and read on demand
use_memmap = sinfo.bytes > 200e6;
if use_memmap
    mm = memmapfile(sniffer_fc32_file, 'Format', {'single', [n_floats, 1], 'data'});
    raw_vec = mm.Data.data; % single vector length n_floats
    raw_vec = reshape(raw_vec, 1, []); % ensure row vector
    sniffer_iq = raw_vec(1:2:end) + 1j*raw_vec(2:2:end);
    % clear raw_vec to free memory (memmap still mapped)
    clear raw_vec;
else
    fid = fopen(sniffer_fc32_file,'rb','l');
    if fid==-1, error('Cannot open sniffer file.'); end
    raw = fread(fid, n_floats, 'float32');
    fclose(fid);
    sniffer_iq = raw(1:2:end) + 1j*raw(2:2:end);
    clear raw;
end

% small safety check
if isempty(sniffer_iq)
    error('Sniffer IQ load failed or returned empty.');
end

%% ---------- Optionally read gNB phy_rx_symbols (if present) ----------
use_gnb_symbols = false;
gnb_symbols_iq = [];
if exist(gnb_symbols_file,'file')
    try
        info_g = dir(gnb_symbols_file);
        n_floats_g = floor(info_g.bytes/4);
        if mod(n_floats_g,2)~=0
            n_floats_g = n_floats_g - 1;
        end
        Ng = n_floats_g/2;
        fprintf('gNB symbols file found: %d complex samples (%.3f s at %.2f MHz)\n', Ng, Ng/fs_gnb, fs_gnb/1e6);
        % use memmap if large
        if info_g.bytes > 200e6
            mmg = memmapfile(gnb_symbols_file, 'Format', {'single', [n_floats_g,1], 'data'});
            rawg = mmg.Data.data;
            gnb_symbols_iq = rawg(1:2:end) + 1j*rawg(2:2:end);
            clear rawg;
        else
            fid = fopen(gnb_symbols_file,'rb','l');
            rawg = fread(fid, n_floats_g, 'float32');
            fclose(fid);
            gnb_symbols_iq = rawg(1:2:end) + 1j*rawg(2:2:end);
            clear rawg;
        end
        use_gnb_symbols = true;
    catch ME
        warning('Could not read gNB symbols file: %s\nProceeding without it.\n', ME.message);
        use_gnb_symbols = false;
    end
else
    fprintf('gNB symbols file not found: %s\n', gnb_symbols_file);
end

%% ---------- Coarse alignment ----------
delta_samples = NaN;
aligned_by = 'none';
% --- PATCH: ensure sniffer_index fields exist and compute safe mappings ---
% (Insert this right after the coarse-alignment block that sets delta_samples)

% Ensure fields exist
if ~ismember('sniffer_index_from_time', gnb_events.Properties.VariableNames)
    gnb_events.sniffer_index_from_time = NaN(height(gnb_events),1);
end
if ~ismember('sniffer_index_from_offset', gnb_events.Properties.VariableNames)
    gnb_events.sniffer_index_from_offset = NaN(height(gnb_events),1);
end
if ~ismember('sniffer_index', gnb_events.Properties.VariableNames)
    gnb_events.sniffer_index = NaN(height(gnb_events),1);
end

% Compute time-based sample index if sniffer_start_datetime was provided
if ~isnat(sniffer_start_datetime)
    gnb_events.sniffer_index_from_time = round( seconds(gnb_events.timestamp - sniffer_start_datetime) * fs_sniffer );
    fprintf('Computed sniffer_index_from_time for %d events using provided sniffer_start_datetime.\n', height(gnb_events));
end

% Compute offset-based sample index if delta_samples available (from symbols xcorr)
if ~isnan(delta_samples) && use_gnb_symbols
    gnb_events.sniffer_index_from_offset = gnb_events.offset + delta_samples;
    fprintf('Computed sniffer_index_from_offset for %d events using delta_samples=%d.\n', height(gnb_events), round(delta_samples));
end

% Choose the primary index: prefer offset-based (if within sniffer range), else time-based (if within range)
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

% Count how many events actually fall within the short sniffer capture
n_in_range = sum(~isnan(gnb_events.sniffer_index));
fprintf('Events with a valid sniffer_index inside the sniffer file: %d of %d total events.\n', n_in_range, height(gnb_events));
% --- end patch ---

if use_gnb_symbols
    fprintf('Attempting coarse alignment using decimated energy cross-correlation (safe limits)...\n');
    % compute energy envelopes (short moving average to be robust)
    wlen_env_sn = max(1, round((energy_window_ms/1000) * fs_sniffer));
    wlen_env_gnb = max(1, round((energy_window_ms/1000) * fs_gnb));
    env_sn = movmean(abs(sniffer_iq).^2, wlen_env_sn);
    env_g = movmean(abs(gnb_symbols_iq).^2, wlen_env_gnb);

    % decimate to speed up xcorr
    decim = max(1, floor(length(env_sn)/1e6)); % target ~1e6 samples or less
    env_sn_d = env_sn(1:decim:end);
    env_g_d = env_g(1:decim:end);
    clear env_sn env_g;

    % Make sure the decimated vectors have the same length for normalized xcorr
    minLen = min(length(env_sn_d), length(env_g_d));
    if minLen < 512
        fprintf('Warning: decimated envelopes have very small overlap (minLen=%d). Skipping coarse xcorr alignment.\n', minLen);
    else
        env_sn_d = env_sn_d(1:minLen);
        env_g_d  = env_g_d(1:minLen);

        % limit max lag according to max_coarse_align_seconds but also <= minLen-1
        maxlag = min( floor(max_coarse_align_seconds * (fs_sniffer/decim)), minLen-1 );
        if maxlag < 1
            fprintf('Warning: computed maxlag < 1 (maxlag=%d). Skipping coarse xcorr alignment.\n', maxlag);
        else
            % Now safe to use 'normalized' because both inputs are same length
            [c, lags] = xcorr(env_sn_d - mean(env_sn_d), env_g_d - mean(env_g_d), maxlag, 'normalized');
            [~, imax] = max(abs(c));
            lag_decim = lags(imax);
            delta_samples = lag_decim * decim; % sniffer_index - gnb_index estimate
            aligned_by = 'symbols_xcorr';
            fprintf('Coarse alignment result: delta_samples = %d (%.6f s)\n', delta_samples, delta_samples/fs_sniffer);
        end
    end
else
    fprintf('gNB symbols not available; skipping symbols xcorr alignment.\n');
end


%% ---------- For each event compute expected sniffer index and analyze ----------
Ns_total = length(sniffer_iq);
gnb_events.sniffer_index = NaN(height(gnb_events),1);
gnb_events.sniffer_index_from_offset = NaN(height(gnb_events),1);

for i=1:height(gnb_events)
    if ~isnan(delta_samples) && use_gnb_symbols
        % from offset: gnb offset is sample index within gnb_symbols -> map to sniffer using delta
        sidx = gnb_events.offset(i) + delta_samples;
        gnb_events.sniffer_index_from_offset(i) = sidx;
    end
    % from timestamp (if computed)
    if ~isnan(gnb_events.sniffer_index_from_time(i))
        sidx_time = gnb_events.sniffer_index_from_time(i);
    else
        sidx_time = NaN;
    end

    % choose a valid index: prefer offset if within sniffer file, else time-based if valid
    chosen = NaN;
    if ~isnan(gnb_events.sniffer_index_from_offset(i)) && gnb_events.sniffer_index_from_offset(i) > 0 && gnb_events.sniffer_index_from_offset(i) <= Ns_total
        chosen = round(gnb_events.sniffer_index_from_offset(i));
    elseif ~isnan(sidx_time) && sidx_time>0 && sidx_time<=Ns_total
        chosen = round(sidx_time);
    else
        chosen = NaN;
    end
    gnb_events.sniffer_index(i) = chosen;
end

% Prepare sniffer power envelope and automatic peaks (coarse)
winlen = max(1, round((energy_window_ms/1000)*fs_sniffer));
sniffer_power = movmean(abs(sniffer_iq).^2, winlen);
sniffer_db = 10*log10(sniffer_power + eps);
minPeakHeight = median(sniffer_db) + detection_threshold_std*std(sniffer_db);
[pk_vals, pk_locs] = findpeaks(sniffer_db, 'MinPeakHeight', minPeakHeight, 'MinPeakProminence', min_peak_prominence_db);

fprintf('Sniffer automatic energy peak detector found %d peaks.\n', length(pk_locs));

%% ---------- Per-event detailed check (local energy + optional xcorr with gNB symbols) ----------
results = table((1:height(gnb_events))', gnb_events.timestamp, gnb_events.offset, gnb_events.size, ...
    gnb_events.rssi, gnb_events.detection_metric, gnb_events.sniffer_index, ...
    'VariableNames', {'idx','gnb_timestamp','gnb_offset','gnb_size','gnb_rssi','gnb_detection_metric','sniffer_index'});

% Add result columns
results.match = false(height(results),1);
results.reason = strings(height(results),1);
results.sniffer_peak_db = NaN(height(results),1);
results.sniffer_peak_idx = NaN(height(results),1);
results.time_shift_samples = NaN(height(results),1); % sniffer_idx - expected_idx
results.time_shift_ms = NaN(height(results),1);

% For accurate per-event lag estimate using cross-correlation to gNB symbol snippet (if available)
for i=1:height(gnb_events)
    sidx = gnb_events.sniffer_index(i);
    if isnan(sidx)
        results.reason(i) = "index_out_of_range_or_unknown";
        continue;
    end

    pre_s = round(per_event_window_ms(1)/1000 * fs_sniffer);
    post_s = round(per_event_window_ms(2)/1000 * fs_sniffer);
    start_s = max(1, sidx - pre_s);
    end_s = min(Ns_total, sidx + post_s);
    local_db = sniffer_db(start_s:end_s);
    [pmax, pidx] = max(local_db);
    peak_global_idx = start_s + (pidx-1);

    results.sniffer_peak_db(i) = pmax;
    results.sniffer_peak_idx(i) = peak_global_idx;

    % Decide coarse match by energy
    if pmax >= minPeakHeight
        % was there a large shift?
        if abs(peak_global_idx - sidx) <= max(pre_s, post_s)
            results.match(i) = true;
            if abs(peak_global_idx - sidx) <= round(0.002*fs_sniffer) % 2 ms tolerance
                results.reason(i) = "matched_by_energy";
            else
                results.reason(i) = "matched_but_shifted_time";
            end
        else
            results.match(i) = true;
            results.reason(i) = "matched_but_far_shifted"; % peak found but not near expectation
        end
    else
        % no significant energy -> check for zeros or truncated area
        raw_chunk = sniffer_iq(start_s:end_s);
        if all(abs(raw_chunk) < 1e-8)
            results.reason(i) = "zeros_or_truncated_data";
            results.match(i) = false;
        else
            results.reason(i) = "low_snr_or_filtered_out";
            results.match(i) = false;
        end
    end

    % If gNB symbols available, compute precise sample lag via xcorr between gNB snippet and sniffer snippet
    if use_gnb_symbols
        % pick a gNB symbol snippet around gnb offset
        g_start = max(1, gnb_events.offset(i) - pre_s);
        g_end = min(length(gnb_symbols_iq), gnb_events.offset(i) + post_s);
        g_chunk = gnb_symbols_iq(g_start:g_end);
        s_chunk = sniffer_iq(start_s:end_s);
        if length(g_chunk) >= 32 && length(s_chunk) >= 32
            % compute cross-correlation (complex), limit lag
            maxlag_local = min( round(0.01*fs_sniffer), floor(length(s_chunk)/2) ); % 10 ms max
            [c_loc, lags_loc] = xcorr(s_chunk - mean(s_chunk), g_chunk - mean(g_chunk), maxlag_local, 'coeff');
            [~, imaxloc] = max(abs(c_loc));
            lag_loc = lags_loc(imaxloc);
            % lag_loc positive => s_chunk leads g_chunk (sniffer index higher than gnb index)
            % convert local lag to global sample difference: (peak_global_idx - sidx) + lag_loc ...
            % Simpler: compute exact sample shift between gnb offset and sniffer peak
            % approximate expected sniffer index from gnb offset using previously computed delta_samples
            if ~isnan(delta_samples)
                expected_sniffer_idx = gnb_events.offset(i) + delta_samples;
            else
                expected_sniffer_idx = sidx; % fallback previously chosen index
            end
            precise_shift = round( (start_s - expected_sniffer_idx) + lag_loc );
            results.time_shift_samples(i) = precise_shift;
            results.time_shift_ms(i) = precise_shift / fs_sniffer * 1e3;
            % If strong correlation but earlier marked unmatched due to energy threshold, mark reason accordingly
            if abs(c_loc(imaxloc)) > 0.25 % threshold for correlation strength (empirical)
                if ~results.match(i)
                    results.match(i) = true;
                    results.reason(i) = "xcorr_matched_low_energy";
                end
            end
        end
    else
        % if no gnb symbols, we can still give a crude shift relative to expected sidx
        if ~isnan(sidx)
            results.time_shift_samples(i) = peak_global_idx - sidx;
            results.time_shift_ms(i) = results.time_shift_samples(i) / fs_sniffer * 1e3;
        end
    end

    % Save small spectrogram image for any unmatched/shifted events for manual inspection
    try
        fname = sprintf('event_%03d_%s.png', i, datestr(gnb_events.timestamp(i),'yyyymmdd_HHMMSS'));
        figure('Visible','off','Position',[100 100 900 500]);
        window = 256; noverlap = round(window*0.8); nfft = 512;
        chunk_iq = sniffer_iq(start_s:end_s);
        [S,F,T] = spectrogram(chunk_iq, window, noverlap, nfft, fs_sniffer);
        imagesc((T - per_event_window_ms(1)/1000), F/1e6, 20*log10(abs(S)+eps));
        axis xy; xlabel('Time (s) relative to expected event'); ylabel('Freq (MHz)');
        title(sprintf('Event %d: %s (reason: %s, shift_ms=%.3f)', i, datestr(gnb_events.timestamp(i),'yyyy-mm-dd HH:MM:SS.FFF'), results.reason(i), results.time_shift_ms(i)));
        colorbar;
        hold on;
        xline((sidx - start_s)/fs_sniffer, 'r--', 'Expected');
        xline((results.sniffer_peak_idx(i) - start_s)/fs_sniffer, 'g-', 'Peak');
        saveas(gcf, fname);
        close(gcf);
    catch
        % ignore image save failures
    end
end

%% ---------- Summary statistics and CSV ----------
% Add fields to results table for CSV export
results.gnb_rssi = gnb_events.rssi;
results.gnb_detection_metric = gnb_events.detection_metric;

writetable(results, output_csv);
fprintf('Saved detailed report to %s\n', output_csv);

% Print summary
n_total = height(results);
n_matched = sum(results.match);
n_unmatched = n_total - n_matched;
fprintf('\nSummary: %d gNB events => %d matched, %d unmatched.\n', n_total, n_matched, n_unmatched);

% Time-shift distribution (use only finite entries)
valid_shifts = results.time_shift_samples(~isnan(results.time_shift_samples));
if ~isempty(valid_shifts)
    med_shift = median(valid_shifts);
    mean_shift = mean(valid_shifts);
    std_shift = std(valid_shifts);
    fprintf('Time-shift summary (samples): median=%d, mean=%.1f, std=%.1f => (ms) median=%.3f ms\n', ...
        med_shift, mean_shift, std_shift, med_shift/fs_sniffer*1e3);
    % Show histogram summary
    figure('Position',[200 200 800 300]);
    histogram(valid_shifts / fs_sniffer * 1e3, 40);
    xlabel('Time shift (ms)'); ylabel('Count'); title('Distribution of measured per-event time shifts (ms)');
end

% Show top unmatched rows
if n_unmatched>0
    fprintf('Top unmatched events (first 20):\n');
    tmp = results(~results.match, {'idx','gnb_timestamp','reason','sniffer_peak_db','time_shift_ms'});
    numShow = min(20, height(tmp));
    disp(tmp(1:numShow, :));
end


fprintf('\nNotes & next actions:\n');
fprintf('- If time shifts are essentially constant (median shift >> 0 and low std): you have a clock offset. Use that delta to realign recordings (script computed delta_samples if symbols available).\n');
fprintf('- If time shifts vary widely (high std): you have jitter / variable buffering / thread scheduling issues in sniffer. Investigate capture pipeline timing and pre-buffering.\n');
fprintf('- If many events are "zeros_or_truncated_data": check capture truncation or wrong file format/endianness.\n');
fprintf('- If many are "low_snr_or_filtered_out": check sniffer center freq, bandwidth, front-end gain, or filtering. Use spectrogram PNGs event_*.png for each problematic event.\n');
fprintf('- If you want, provide a PRACH preamble waveform; script can run matched-correlation per event for very accurate detection.\n');
fprintf('Done.\n');


