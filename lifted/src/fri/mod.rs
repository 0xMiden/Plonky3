
pub mod fold;
pub mod commit;


pub struct Params {
    log_blowup: usize,
    log_folding_factor: usize,
    log_final_degree: usize,
}
