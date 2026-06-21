//! PIOP proof layer for one integer GPT-2 inference.
//!
//! The arithmetic protocol is transparent: proof messages carry the multilinear evaluations
//! required by the verifier's sumcheck and LogUp checks.

pub mod model;
pub mod piop;
pub mod protocol;
pub mod tensor;
pub mod witness;

pub use model::{BlockWeights, ExportedPublicData, ModelWeights};
pub use piop::{prove, verify, InferencePiopProof, PublicInput};
pub use protocol::{prove_protocol, verify_protocol, ProtocolProof};
pub use tensor::{Matrix, Tensor3};
pub use witness::{Config, Witness};
