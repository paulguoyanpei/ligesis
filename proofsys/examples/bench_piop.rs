use std::time::{Duration, Instant};

use proofsys::{protocol::EF, prove, prove_protocol, verify_protocol, Config, ModelWeights};
use utils::oracle::RandomOracle;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let export_dir = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("../int_gpt/export");
    let runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    let config = Config::gpt2_31();
    let load_start = Instant::now();
    let weights = ModelWeights::load_export_dir(export_dir, &config).expect("load weights");
    let public =
        ModelWeights::load_exported_public_data(export_dir, &config).expect("load public data");
    let load_time = load_start.elapsed();
    let mut times = Vec::with_capacity(runs);
    let mut protocol_times = Vec::with_capacity(runs);
    let mut protocol_verify_times = Vec::with_capacity(runs);
    let mut proof_sizes = Vec::with_capacity(runs);
    let mut protocol_sizes = Vec::with_capacity(runs);
    let mut lookup_real_queries = Vec::with_capacity(runs);
    let mut lookup_padded_queries = Vec::with_capacity(runs);
    let mut fs_draws = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let proof = prove(&config, &weights, public.x0.clone());
        let elapsed = start.elapsed();
        assert_eq!(proof.witness.q_log, public.logits_int);
        proof_sizes.push(transparent_proof_entries(&proof));
        times.push(elapsed);

        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let start = Instant::now();
        let protocol = prove_protocol(&config, &weights, &proof.witness, &mut oracle);
        protocol_times.push(start.elapsed());
        protocol_sizes.push(protocol.size_bytes());
        lookup_real_queries.push(protocol.lookup.num_real_queries);
        lookup_padded_queries.push(protocol.lookup.num_padded_queries);
        fs_draws.push((protocol.fs_field_challenges, protocol.fs_int_challenges));

        oracle.restart();
        let start = Instant::now();
        assert!(verify_protocol(&config, &protocol, &mut oracle));
        protocol_verify_times.push(start.elapsed());
    }

    let min = times.iter().min().copied().unwrap_or_default();
    let mean = mean_duration(&times);
    let max = times.iter().max().copied().unwrap_or_default();
    let protocol_min = protocol_times.iter().min().copied().unwrap_or_default();
    let protocol_mean = mean_duration(&protocol_times);
    let protocol_max = protocol_times.iter().max().copied().unwrap_or_default();
    let protocol_verify_min = protocol_verify_times
        .iter()
        .min()
        .copied()
        .unwrap_or_default();
    let protocol_verify_mean = mean_duration(&protocol_verify_times);
    let protocol_verify_max = protocol_verify_times
        .iter()
        .max()
        .copied()
        .unwrap_or_default();

    println!("export_dir={export_dir}");
    println!("runs={runs}");
    println!("load_time_ms={:.3}", millis(load_time));
    println!("prover_min_ms={:.3}", millis(min));
    println!("prover_mean_ms={:.3}", millis(mean));
    println!("prover_max_ms={:.3}", millis(max));
    println!("protocol_min_ms={:.3}", millis(protocol_min));
    println!("protocol_mean_ms={:.3}", millis(protocol_mean));
    println!("protocol_max_ms={:.3}", millis(protocol_max));
    println!("protocol_verify_min_ms={:.3}", millis(protocol_verify_min));
    println!(
        "protocol_verify_mean_ms={:.3}",
        millis(protocol_verify_mean)
    );
    println!("protocol_verify_max_ms={:.3}", millis(protocol_verify_max));
    println!("transparent_proof_i64_entries={}", proof_sizes[0]);
    println!("lookup_real_queries={}", lookup_real_queries[0]);
    println!("lookup_padded_queries={}", lookup_padded_queries[0]);
    println!("protocol_size_bytes={}", protocol_sizes[0]);
    println!("fiat_shamir_field_challenges={}", fs_draws[0].0);
    println!("fiat_shamir_int_challenges={}", fs_draws[0].1);
}

fn transparent_proof_entries(proof: &proofsys::InferencePiopProof) -> usize {
    let p = &proof.merged_polys;
    p.std.len()
        + p.q_qkv.len()
        + p.q_sc.len()
        + p.x_max.len()
        + p.exp.len()
        + p.q_prob.len()
        + p.q_ao.len()
        + p.q_fc.len()
        + p.act.len()
}

fn mean_duration(values: &[Duration]) -> Duration {
    let nanos: u128 = values.iter().map(Duration::as_nanos).sum();
    Duration::from_nanos((nanos / values.len() as u128) as u64)
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
