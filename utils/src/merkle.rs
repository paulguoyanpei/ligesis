use p3_field::{Field, RawDataSerializable};
use rs_merkle::{Hasher, MerkleProof, MerkleTree};

#[derive(Debug, Clone)]
pub struct Blake32 {}

impl Hasher for Blake32 {
    type Hash = [u8; 32];

    fn hash(data: &[u8]) -> [u8; 32] {
        blake3::hash(data).as_bytes().clone()
    }
}

#[derive(Clone)]
pub struct MerkleTreeProver {
    pub merkle_tree: MerkleTree<Blake32>,
    leave_num: usize,
}

pub struct Serialize;

impl Serialize {
    pub fn serialize_fields<F: Field>(v: &[F]) -> Vec<u8> {
        <F as RawDataSerializable>::into_byte_stream(v.iter().copied())
            .into_iter()
            .collect()
    }

    /// Append the serialized bytes of `v` to `out`, reusing its allocation. Lets a caller
    /// build many leaves into one reusable buffer instead of allocating a `Vec` per leaf.
    pub fn serialize_fields_into<F: Field>(v: &[F], out: &mut Vec<u8>) {
        out.extend(<F as RawDataSerializable>::into_byte_stream(v.iter().copied()));
    }
}

/// Hash one leaf's bytes with the Merkle tree's hash function.
pub fn hash_leaf(bytes: &[u8]) -> <Blake32 as Hasher>::Hash {
    Blake32::hash(bytes)
}

#[derive(Debug, Clone)]
pub struct MerkleTreeVerifier {
    pub merkle_root: <Blake32 as Hasher>::Hash,
    pub leave_number: usize,
}

impl MerkleTreeProver {
    pub fn new(leaf_values: &Vec<Vec<u8>>) -> Self {
        let leaves = leaf_values
            .iter()
            .map(|x| Blake32::hash(x))
            .collect::<Vec<_>>();
        Self::from_leaf_hashes(leaves)
    }

    /// Build the tree from precomputed leaf hashes (i.e. each `Blake32::hash(leaf_bytes)`).
    /// Lets callers fuse leaf serialization and hashing without materializing the leaf bytes.
    pub fn from_leaf_hashes(leaf_hashes: Vec<<Blake32 as Hasher>::Hash>) -> Self {
        let leave_num = leaf_hashes.len();
        let merkle_tree = MerkleTree::<Blake32>::from_leaves(&leaf_hashes);
        Self {
            merkle_tree,
            leave_num,
        }
    }

    pub fn leave_num(&self) -> usize {
        self.leave_num
    }

    pub fn commit(&self) -> <Blake32 as Hasher>::Hash {
        self.merkle_tree.root().unwrap()
    }

    pub fn open(&self, leaf_indices: &[usize]) -> Vec<u8> {
        self.merkle_tree.proof(leaf_indices).to_bytes()
    }
}

impl MerkleTreeVerifier {
    pub fn new(leave_number: usize, merkle_root: &<Blake32 as Hasher>::Hash) -> Self {
        Self {
            leave_number,
            merkle_root: merkle_root.clone(),
        }
    }

    pub fn verify(
        &self,
        proof_bytes: Vec<u8>,
        indices: &Vec<usize>,
        leaves: &Vec<Vec<u8>>,
    ) -> bool {
        let proof = MerkleProof::<Blake32>::try_from(proof_bytes).unwrap();
        let leaves_to_prove = leaves.iter().map(|x| Blake32::hash(x)).collect::<Vec<_>>();
        proof.verify(
            self.merkle_root,
            indices,
            &leaves_to_prove,
            self.leave_number,
        )
    }

    pub fn get_root(
        proof_bytes: Vec<u8>,
        index: usize,
        leaf: Vec<u8>,
        leave_number: usize,
    ) -> <Blake32 as Hasher>::Hash {
        let proof = MerkleProof::<Blake32>::try_from(proof_bytes).unwrap();
        let leaf_hashes = vec![Blake32::hash(&leaf)];
        proof
            .root(&vec![index], &leaf_hashes, leave_number)
            .unwrap()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use p3_field::PrimeField64;
    use p3_goldilocks::Goldilocks;

    #[test]
    fn commit_and_open() {
        let leaf_values = (0..8)
            .map(|x| {
                Serialize::serialize_fields(&[Goldilocks::new(x * 2), Goldilocks::new(x * 2 + 1)])
            })
            .collect::<Vec<_>>();
        let leave_number = leaf_values.len();
        let prover = MerkleTreeProver::new(&leaf_values);
        let root = prover.commit();
        let verifier = MerkleTreeVerifier::new(leave_number, &root);
        let leaf_indices = vec![2, 3];
        let proof_bytes = prover.open(&leaf_indices);
        let open_values = vec![
            Serialize::serialize_fields(&[Goldilocks::new(2 * 2), Goldilocks::new(2 * 2 + 1)]),
            Serialize::serialize_fields(&[Goldilocks::new(3 * 2), Goldilocks::new(3 * 2 + 1)]),
        ];
        assert!(verifier.verify(proof_bytes, &leaf_indices, &open_values));
    }

    #[test]
    fn serialize() {
        let values = [Goldilocks::new(42), Goldilocks::new(u64::MAX)];
        let bytes = Serialize::serialize_fields(&values);

        assert_eq!(bytes.len(), values.len() * Goldilocks::NUM_BYTES);
        assert_eq!(
            &bytes[..Goldilocks::NUM_BYTES],
            &values[0].as_canonical_u64().to_le_bytes()
        );
    }
}
