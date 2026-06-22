//! Faithful PIOP for one integer GPT-2 inference: commit the witness once, reduce every sumcheck
//! and the unified LogUp lookup to opening claims against those commitments, and discharge them
//! with a single PCS batch-open (transparent placeholder today; Basefold-ready). See
//! [`faithful`] for the assembly and `examples/bench_faithful.rs` for the end-to-end benchmark.

pub mod canonical;
pub mod commit;
pub mod faithful;
pub mod model;
pub mod protocol;
pub mod reduce;
pub mod tensor;
pub mod witness;

pub use model::{BlockWeights, ExportedPublicData, ModelWeights};
pub use protocol::EF;
pub use tensor::{Matrix, Tensor3};
pub use witness::{Config, Witness};
