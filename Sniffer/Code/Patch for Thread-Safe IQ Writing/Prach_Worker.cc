#include "sniffer/hdr/prach/prach_worker.h"
#include <fstream>
#include <sstream>
#include <thread>
#include <functional>
#include "sniffer/hdr/adt/complex.h"

using namespace sniffer;


bool prach_worker::is_preamble_occasion(uint32_t                          sfn,
                                       uint32_t                          sf_idx,
                                       unsigned                          x,
                                       const static_vector<uint8_t, 2>&  y,
                                       const static_vector<uint8_t, 40>& slots)
{
 for (uint8_t offset : y) {
   if ((sfn % x) == offset) {
     for (uint8_t slot : slots) {
       if (sf_idx == slot) {
         return true;
       }
     }
   }
 }
 return false;
}


prach_worker::prach_worker(srsran_subcarrier_spacing_t scs,
                          double                      srate,
                          bool                        _prach_scan,
                          float                       _freq_offset_ssb_fc,
                          std::function<void()>       on_prach_found) :
 logger(srslog::fetch_basic_logger("PRE_DEC")),
 advance_max_sz(1000),
 delay_max_sz(1000)
{
 is_inited = false;


 memset(&sib1_prach_cfg, 0, sizeof(srsran_prach_cfg_t));
 //   memset(&prach_conf, 0, sizeof(prach_configuration)); // can't do this... fixed value inside
 //   memset(&prach_context, 0, sizeof(prach_buffer_context));


 pusch_scs          = scs;
 srsran_srate_hz    = srate;
 prach_scan         = _prach_scan;
 freq_offset_ssb_fc = _freq_offset_ssb_fc;


 on_preamble_found_callback = on_prach_found;


 // Allocate receive buffer
 slot_sz     = (uint32_t)(srsran_srate_hz / 1000.0f / SRSRAN_NOF_SLOTS_PER_SF_NR(pusch_scs));
 subframe_sz = SRSRAN_NOF_SLOTS_PER_SF_NR(pusch_scs) * slot_sz;


 t_offset = 0;


 // extra space used for time offset(delay or advance)
 input[0] = srsran_vec_cf_malloc(subframe_sz+advance_max_sz+delay_max_sz);
 if (!input[0]) {
   printf("Error malloc input buffer inside sib worker\n");
 }
 srsran_vec_cf_zero(input[0], subframe_sz+advance_max_sz+delay_max_sz);


 create_prach_demodulator();


 // code from include/srsran/phy/upper/channel_processors/channel_processor_factories.h
 dft_factory = create_dft_processor_factory_fftw_slow();


 // Construct prach_dtct
 idft_long_config.size  = idft_long_size;
 idft_long_config.dir   = dft_processor::direction::INVERSE;
 idft_short_config.size = idft_short_size;
 idft_short_config.dir  = dft_processor::direction::INVERSE;


 // Construct prach_generator
 prach_gen  = std::make_unique<prach_generator_impl>();
 prach_dtct = std::make_unique<prach_detector_generic_impl>(dft_factory->create(idft_long_config),
                                                            dft_factory->create(idft_short_config),
                                                            std::move(prach_gen),
                                                            combine_symbols);
 logger.info("Preamble detector worker constructed");
}


prach_worker::~prach_worker()
{
 if (input[0]) {
   free(input[0]);
 }
};


prach_configuration* prach_worker::init(asn1::rrc_nr::sib1_s& sib1)
{
 if (is_inited) {
   return nullptr;
 }


 if (!sib1.serving_cell_cfg_common_present || !sib1.serving_cell_cfg_common.ul_cfg_common_present) {
   logger.error("SIB uncomplete");
   return nullptr;
 }


 srsran::make_phy_rach_cfg(sib1.serving_cell_cfg_common.ul_cfg_common.init_ul_bwp.rach_cfg_common.setup(),
                           srsran_duplex_mode_t::SRSRAN_DUPLEX_MODE_TDD,
                           &sib1_prach_cfg);


 //
 //   prach_conf = prach_configuration_get(frequency_range::FR1, duplex_mode::TDD, (uint8_t) 98);
 prach_conf = prach_configuration_get(frequency_range::FR1, duplex_mode::TDD, (uint8_t)sib1_prach_cfg.config_idx);


 if (prach_conf.format == prach_format_type::invalid) {
   logger.error("Got invalid prachconfiguration");
   return nullptr;
 }
 /**
  * TODO: Add prach_conf verification
  *
  * TODO: prach_conf to prach_context
  */


 prach_context.format              = prach_conf.format;
 prach_context.pusch_scs           = to_subcarrier_spacing(pusch_scs);
 prach_context.start_symbol        = prach_conf.starting_symbol;
 prach_context.nof_td_occasions    = prach_conf.nof_occasions_within_slot;
 prach_context.root_sequence_index = sib1_prach_cfg.root_seq_idx;
 // prach_context.slot
 prach_context.rb_offset             = sib1_prach_cfg.freq_offset;
 prach_context.zero_correlation_zone = sib1_prach_cfg.zero_corr_zone;
 prach_context.nof_preamble_indices  = sib1_prach_cfg.num_ra_preambles;
  prach_context.nof_prb_ul_grid = 51;
 prach_context.nof_fd_occasions = 1;
 prach_context.restricted_set       = restricted_set_config::UNRESTRICTED;
 prach_context.ports                = {0};
 prach_context.sector               = 0;
 prach_context.start_preamble_index = 0;


 // prach_conf.x;
 // prach_conf.y;
 // prach_conf.slots;


 for (uint8_t sf_num : prach_conf.slots) {
   logger.info("Prach subframe occasion: %d", (int)sf_num);
 }


 uint8_t l_0         = prach_conf.starting_symbol;
 uint8_t N_RA_slot_t = prach_conf.nof_occasions_within_slot;


 for (uint8_t n = 0; n < N_RA_slot_t; n++) {
   uint8_t l = l_0 + n * prach_conf.duration + 14 * prach_conf.nof_prach_slots_within_subframe;
   logger.info("Prach symbol occasion: %d", (int)l);
   symbol_occasion_L.push_back(l);


   /** NOTE: Only support FR1 */
   int nof_symbols_per_slot = 14;


   int slot_idx = l / nof_symbols_per_slot; // num of symbol per slot
   slot_indices.insert(slot_idx);
 }


 /**
  * TODO: Verify the configuration is correctly set
  */


 // Calculate the PRACH window size starting at the beginning of the slot.
 window_length =
     get_prach_window_duration(
         prach_context.format, prach_context.pusch_scs, prach_context.start_symbol, prach_context.nof_td_occasions)
         .to_samples(srsran_srate_hz);


 logger.debug("Prach decoding window length: %u", window_length);


 //   // put the prach slot samples into a buffer
 //   // Initialize the output buffer
 //   get_span_buffer();
 unsigned num_ports             = prach_context.ports.size();
 unsigned td_occasions          = prach_context.nof_td_occasions;
 unsigned fd_occasions          = 1;                   /** FIXME: msg1-FDM is unhandled, use one as default */
 unsigned symbols_per_occasion  = prach_conf.duration; // duration only valid for short format
 unsigned prach_sequence_length = is_long_preamble(prach_conf.format) ? 839U : 139U;


 /** FIXME: Check the scs for long preamble format, might leads to error */


 buffer = std::make_unique<prach_buffer>(
     num_ports, td_occasions, fd_occasions, symbols_per_occasion, prach_sequence_length);


 logger.info("PRACH Worker init finished");


 is_inited = true;
 return &prach_conf;
}


void prach_worker::create_prach_demodulator()
{
 frequency_range fr    = frequency_range::FR1;
 sampling_rate   srate = sampling_rate::from_Hz<uint32_t>(srsran_srate_hz);


 static constexpr std::array<prach_subcarrier_spacing, 5> fr1_prach_scs = {prach_subcarrier_spacing::kHz15,
                                                                           prach_subcarrier_spacing::kHz30,
                                                                           prach_subcarrier_spacing::kHz60,
                                                                           prach_subcarrier_spacing::kHz1_25,
                                                                           prach_subcarrier_spacing::kHz5};


 // Select set of valid PRACH subcarrier spacing for the given frequency range. This avoids having extremely large
 // unused DFTs.
 span<const prach_subcarrier_spacing> possible_prach_scs = span<const prach_subcarrier_spacing>(fr1_prach_scs);


 bool               avoid_wisdom    = false;
 const std::string& wisdom_filename = "";


 std::shared_ptr<dft_processor_factory> dft_factory = create_dft_processor_factory_fftw_slow();


 for (prach_subcarrier_spacing ra_scs : possible_prach_scs) {
   // Create DFT for the given PRACH subcarrier spacing.
   dft_processor::configuration   dft_config = {.size = srate.get_dft_size(ra_scs_to_Hz(ra_scs)),
                                                .dir  = dft_processor::direction::DIRECT};
   std::unique_ptr<dft_processor> dft_proc   = dft_factory->create(dft_config);
   srsran_assert(dft_proc,
                 "Invalid DFT processor of size {}, for subcarrier spacing of {} and sampling rate {}.",
                 dft_config.size,
                 to_string(ra_scs),
                 srate);


   // Emplace the DFT into the dictionary.
   dft_processors.emplace(ra_scs, std::move(dft_proc));
 }


 demodulator = std::make_unique<ofdm_prach_demodulator_impl>(srate, std::move(dft_processors));
}




bool prach_worker::update_input(sf_buffer_t* sf_buffer)
{
 if (!is_preamble_occasion(sf_buffer->sfn, sf_buffer->sf_idx, prach_conf.x, prach_conf.y, prach_conf.slots)) {
   sf_buffer->process_end_callback();
   return false;
 }


 logger.info("Processing SFN.Slot: %u.%u", sf_buffer->sfn, sf_buffer->sf_idx);
 this->sf_buffer = sf_buffer;


 // prach_context.slot = ;


 /** TODO: cf_t is not totally same with srsran::cf_t, need to be careful */
 srsran_vec_cf_copy(input[0]+advance_max_sz, sf_buffer->rf_buffer.to_cf_t()[0], subframe_sz);
 return true;
}


/// \brief Returns a PRACH detector slot configuration using the given PRACH buffer context.
static prach_detector::configuration get_prach_dectector_config_from_prach_context(const prach_buffer_context& context)
{
 prach_detector::configuration config;
 config.root_sequence_index   = context.root_sequence_index;
 config.format                = context.format;
 config.restricted_set        = context.restricted_set;
 config.zero_correlation_zone = context.zero_correlation_zone;
 config.start_preamble_index  = context.start_preamble_index;
 config.nof_preamble_indices  = context.nof_preamble_indices;
 config.ra_scs                = to_ra_subcarrier_spacing(context.pusch_scs);
 config.nof_rx_ports          = context.ports.size();
 config.slot                  = context.slot;


 return config;
}


int prach_worker::process()
{
 /**
  * The input contains whole subframe. eg. scs=30khz, 1sf==2slots
  * Input should have extra space for the time offset
  * */
 span<cf_t> input_span_wide =
     span<cf_t>(reinterpret_cast<cf_t*>(input[0]), subframe_sz + advance_max_sz + delay_max_sz);


 // This is imporatant to shift the signal center to carrier center instead of frequency center
 freq_rotate_manual(input_span_wide, input_span_wide, freq_offset_ssb_fc, srsran_srate_hz);


 /** TODO: Currently only support scs 30khz */
 for (auto slt_idx : slot_indices) {


   slot_point slt = slot_point(pusch_scs ,slt_idx);
   prach_context.slot = slt;


   // Move the signal back
  
   if(sf_buffer->subframe_offset < advance_max_sz && sf_buffer->subframe_offset > 0){
     t_offset = sf_buffer->subframe_offset;
   } else {
     logger.debug("sf_buffer->subframe_offset == %lu", sf_buffer->subframe_offset);
   }
   logger.debug("t_offset == %lu", t_offset);


     // Get the span view of offset preamble 'slot'
     if(advance_max_sz + slt_idx * slot_sz - t_offset + slot_sz > subframe_sz + advance_max_sz + delay_max_sz){
       logger.error("The selected slot index exceed maximum index");
       return SRSRAN_ERROR;
     }


     span<cf_t> input_subspan_t_offset =
         input_span_wide.subspan(advance_max_sz + slt_idx * slot_sz - t_offset, slot_sz);


     // Debug
     // export_rx_buffer_to_fc32(reinterpret_cast<const std::complex<float>*>(input_ofc->data()), input_ofc->size(),
     // "/workspaces/nr_sniffer/log/ofc_rach.fc32");


     // OFDM Demodulation
     unsigned nof_ports = 1;
     /** TODO: WARNING uint != int */
     for (unsigned i_port = 0; i_port != nof_ports; ++i_port) {
       // Prepare PRACH demodulator configuration.
       ofdm_prach_demodulator::configuration config;
       config.slot             = prach_context.slot;
       config.format           = prach_context.format;
       config.nof_td_occasions = prach_context.nof_td_occasions;
       config.nof_fd_occasions = prach_context.nof_fd_occasions;
       config.start_symbol     = prach_context.start_symbol;
       config.rb_offset        = prach_context.rb_offset;
       config.nof_prb_ul_grid  = prach_context.nof_prb_ul_grid;
       config.port             = i_port;


       // Demodulate all candidates, and store the result into buffer
       demodulator->demodulate(*buffer, input_subspan_t_offset, config);
       // logger.debug("PRACH Demodulation finished");
     }


     // // Debug use: output the rach symbol
     // string out_path = "/workspaces/nr_sniffer/log/ofdm/rach_symbols_25times.fc32";
     // export_ofdm_symbols(*buffer, 0, 0, 0, 12, out_path.c_str());


     prach_detection_result res =
     prach_dtct->detect(*buffer, get_prach_dectector_config_from_prach_context(prach_context));


     logger.debug("PRACH Detection: RSSI (dB): %f; Time Resolution: %f(us); Max Time Advance: %f(us)", res.rssi_dB, res.time_resolution.to_seconds() * 1e6, res.time_advance_max.to_seconds() * 1e6);


     if (!res.preambles.empty()) {
       logger.info("Preamble detected!!!!!!!!!");
       for (const auto& preamble : res.preambles) {
         logger.info("Preamble Index: %u, Time Advance: %.2f(us), Detection Metric: %lf", preamble.preamble_index, preamble.time_advance.to_seconds() * 1e6, preamble.detection_metric);
       }
       logger.debug("Detected with t_offset: %u", t_offset);
       on_preamble_found_callback();
       return SRSRAN_SUCCESS;
     }
   // }
 }


 return SRSRAN_SUCCESS;
}


// This function will not be called unless it's a prach oaccasion
void prach_worker::work_imp()
{
 logger.info("PRACH detection %u on thread %c, sfn: %u.%u, processing buffer %d", get_id(), std::this_thread::get_id(), sf_buffer->sfn, sf_buffer->sf_idx, sf_buffer->idx);


 struct timeval t0, t1;
 gettimeofday(&t0, NULL);
 // Decode the preamble
 if (process() < SRSRAN_SUCCESS) {
   logger.error("ERROR during preamble worker processing");
 }


 gettimeofday(&t1, NULL);
 logger.debug("PRACH Detection spent: %ld (us)", t1.tv_usec - t0.tv_usec);

 // ------------------ PRACH IQ dump (per-thread, fc32) ------------------
// Writes interleaved float32 [I,Q] for each complex sample.
// File per thread: /home/uob/work/log/sniffer/prach_symbols_thread_<hash>.fc32
{
  // Build per-thread filename (hash of std::thread::id)
  size_t tid_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
  std::ostringstream fname;
  fname << "/home/uob/work/log/sniffer/prach_symbols_thread_" << tid_hash << ".fc32";
  const std::string out_filename = fname.str();

  // Open in binary append mode (so multiple events get appended)
  std::ofstream ofs(out_filename, std::ios::binary | std::ios::app);
  if (!ofs) {
    logger.error("Failed to open PRACH dump file: %s", out_filename.c_str());
  } else {
    // Number of OFDM symbols stored in the buffer (e.g., 12)
    unsigned num_symbols = buffer->get_max_nof_symbols();

    for (unsigned sym = 0; sym < num_symbols; ++sym) {
      // get_symbol(port, td_occ, fd_occ, symbol_idx)
      auto sym_span = buffer->get_symbol(0, 0, 0, sym);

      for (size_t k = 0; k < sym_span.size(); ++k) {
        // Convert the sample to float real/imag.
        // Depending on your sample type, one of these will work.
        // Option A (most std::complex-like types):
        //float re = static_cast<float>(sym_span[k].real());
        //float im = static_cast<float>(sym_span[k].imag()); (Does not work. sym_span[k].real() tried to call something that's not a function)

        // Option B (if project provides helper to convert to cf_t):
        sniffer::cf_t cf = sniffer::to_cf(sym_span[k]);
        float re = cf.real(); 
        float im = cf.imag();

        // Option C (if your type uses .r and .i members instead, uncomment below and comment Option A):
        // float re = static_cast<float>(sym_span[k].r);
        // float im = static_cast<float>(sym_span[k].i);

        // Option D — if members are named .real and .imag (without parentheses)
        //float re = static_cast<float>(sym_span[k].real);
        //float im = static_cast<float>(sym_span[k].imag);

        ofs.write(reinterpret_cast<const char*>(&re), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&im), sizeof(float));
      }
    }

    ofs.close();
    logger.debug("Wrote PRACH symbols to %s", out_filename.c_str());
  }
}
 // ------------------ end PRACH IQ dump ------------------


 logger.info("PRACH Worker %u on thread %c process finished, sfn: %u.%u, processing buffer %d", get_id(), std::this_thread::get_id(), sf_buffer->sfn, sf_buffer->sf_idx, sf_buffer->idx);


 /** IMPORTANT!!!
  *  This must at then end of processing
  */
 sf_buffer->process_end_callback();
}
