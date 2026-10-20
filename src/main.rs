use anyhow::{bail, Result};
use hf_hub::HFClientSync;
use ort::{
    ep,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
    value::Tensor
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const MODEL_USER: &str = "onnx-community";
const REPO: &str = "d1-3B-ONNX";
// Text-only stack: embed + decoder (+ external data files resolve from the same dir)
const FILES: &[&str] = &[
    "onnx/embed_tokens_q4.onnx", 
    "onnx/embed_tokens_q4.onnx_data",
    "onnx/decoder_model_merged_q4.onnx", 
    "onnx/decoder_model_merged_q4.onnx_data",
    "onnx/decoder_model_merged_q4.onnx_data_1", 
    "tokenizer.json",
];

fn fetch_all() -> anyhow::Result<PathBuf> {
    let client = HFClientSync::new()?;
    let repo = client.model(MODEL_USER, REPO);
    let dir = PathBuf::from("model");
    let dir = dir.join("onnx");

    std::fs::create_dir_all(&dir)?;

    for f in FILES {
        let local = dir.join(f);
        if local.exists() {
            continue;
        } else {
            println!("downloading {f} to {local:?}");
        }
        let downloaded = repo
            .download_file()
            .filename(format!("{f}"))
            .local_dir(&dir)
            .send()?;
        if downloaded != local {
            std::fs::rename(&downloaded, &local)?;
        }
    }

    Ok(dir)
}

fn fetch_env() -> ort::Result<ort::environment::Environment> {
    let env: ort::environment::Environment = ort::init().build()?;
    Ok(env)
}

/// GPU first, CPU fallback: try each EP list in order, first that commits wins.
fn make_session(env: &ort::environment::Environment, model: &PathBuf) -> ort::Result<Session> {
    let session = Session::builder(env)?
        .with_execution_providers([
            #[cfg(feature = "cuda")]
            ep::CUDA::default().build(),
            ep::CPU::default().build(),
        ])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model)?;
    
    Ok(session)
}

pub struct D1 {
    tok: Tokenizer,
    embed: Session,
    decoder: Session,
    // single-token forms, resolved at load time (prompt.py's YES_FORMS / NO_FORMS)
    yes_ids: Vec<i64>,
    no_ids: Vec<i64>,
}

impl D1 {
    pub fn new() -> Result<Self> {
        let dir = fetch_all()?;
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let env = fetch_env()?;

        let mut yes_ids = Vec::new();
        let mut no_ids = Vec::new();
        for (forms, out) in [(
            ["yes", "Yes", "YES"],
            &mut yes_ids,
        ), (["no", "No", "NO"], &mut no_ids)] {
            for w in forms {
                let enc = tok.encode(&*w, false).unwrap();
                if enc.get_ids().len() == 1 {
                    out.push(enc.get_ids()[0] as i64);
                }
            }
        }
        if yes_ids.is_empty() || no_ids.is_empty() { bail!("no single-token yes/no forms"); }

        Ok(Self {
            tok,
            embed: make_session(&env, &dir.join("onnx/embed_tokens_q4.onnx"))?,
            decoder: make_session(&env, &dir.join("onnx/decoder_model_merged_q4.onnx"))?,
            yes_ids, no_ids,
        })
    }

    /// d1 prompt (prompt.py, noul): state block + QUESTION: + instructions + assistant header.
    /// Verify byte-parity against the README's JS example on your first run.
    fn prompt(state: &str, instructions: &str) -> String {
        format!(
            "<|startoftext|><|im_start|>user\n{state}\n\nQUESTION:\n{instructions}\
             \n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n"
        )
    }

    pub fn yes_no(&mut self, state: &str, instructions: &str) -> Result<f32> {
        let enc = self.tok.encode(Self::prompt(state, instructions), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let seq = enc.get_ids().len() as i64;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();

        // 1) embed_tokens: input_ids [1,seq] -> inputs_embeds [1,seq,2048]
        let embeds = self.embed.run(ort::inputs![
            "input_ids" => Value::from_array((vec![1i64, seq], ids.clone()))?,
        ])?;
        let (shape, data) = embeds["inputs_embeds"].try_extract_tensor::<f32>()?;
        let embeds = Tensor::from_array((shape.to_vec(), data.to_vec()))?;

        // 2) decoder: one prefill. Cache inputs come back from introspection (see notes):
        //    16 kv tensors [1,8,0,64] (empty past) + 22 conv tensors [1,2048,3] (zero state).
        let mut inputs = ort::inputs![
            "inputs_embeds" => embeds,
            "attention_mask" => Value::from_array((vec![1i64, seq], vec![1i64; seq as usize]))?,
            // "position_ids" => Value::from_array((vec![1i64, seq], (0..seq).collect::<Vec<i64>>()))?,
        ];

        let cache_meta: Vec<(String, usize)> = self.decoder
            .inputs()
            .iter()
            .filter_map(|inp| {
                if !inp.name().starts_with("past") {
                    return None;
                }
                let rank = match inp.dtype() {
                    ort::value::ValueType::Tensor { shape, .. } => shape.len(),
                    _ => return None, // or handle the error outside
                };
                Some((inp.name().to_string(), rank))
            })
            .collect();

            for (n, rank) in &cache_meta {
                let dims: Vec<i64> = match rank {
                    3 => vec![1, 2048, 3],
                    4 => vec![1, 8, 0, 64],
                    _ => bail!("unexpected cache rank on {n}"),
                };
                let len: usize = dims.iter().product::<i64>() as usize;
                inputs.push((n.as_str().into(), Value::from_array((dims, vec![0f32; len]))?.into()));
            }

        let out = self.decoder.run(inputs)?;                    // logits: [1, 1, vocab]
        let logits = out["logits"].try_extract_tensor::<f32>()?;

        // 3) readout: max logit per option group, softmax over options (prompt.py::readout)
        let max_of = |g: &[i64]| g.iter().map(|&i| logits.1[i as usize]).fold(f32::NEG_INFINITY, f32::max);
        let (y, n) = (max_of(&self.yes_ids), max_of(&self.no_ids));
        let m = y.max(n);
        let (ey, en) = ((y - m).exp(), (n - m).exp());
        Ok(ey / (ey + en))
    }
}

fn main() -> Result<()> {
    let mut d1 = D1::new()?;
    let p = d1.yes_no(
        "The experiment was a complete success and the results were published. Unfortunately, the experiment failed to replicate.",
        "Did the experiment succeed?",
    )?;
    println!("P(claim) = {p}"); // tune your flag threshold against this, recall-first
    Ok(())
}