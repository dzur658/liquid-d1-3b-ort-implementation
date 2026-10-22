use anyhow::{bail, Result, Context};
use hf_hub::HFClientSync;
use ort::{
    ep,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
    value::Tensor
};
use std::path::{PathBuf};
use tokenizers::Tokenizer;

const MODEL_USER: &str = "onnx-community";
const MODEL_REPO: &str = "d1-3B-ONNX";
const MODEL_REVISION: &str = "03619d8ef1abc297947473bcedc99838a59bab39"; // pin to October 7th 2026
// Text-only stack: embed + decoder (+ external data files resolve from the same dir)
const FILES: &[&str] = &[
    "onnx/embed_tokens_q4.onnx", 
    "onnx/embed_tokens_q4.onnx_data",
    "onnx/decoder_model_merged_q4.onnx", 
    "onnx/decoder_model_merged_q4.onnx_data",
    "onnx/decoder_model_merged_q4.onnx_data_1", 
    "tokenizer.json",
];

pub struct ModelPaths {
    pub tokenizer: PathBuf,
    pub embed: PathBuf,
    pub decoder: PathBuf,
}

pub fn fetch_weights() -> Result<ModelPaths> {
    let client = HFClientSync::new()?;
    let repo = client
        .model(MODEL_USER, MODEL_REPO); // still keep the pinned SHA

    let (mut tok, mut encoder, mut decoder) = (None, None, None);

    for f in FILES {
        // skips and dedups automatically if already cached
        let p = repo.download_file().filename(*f).revision(MODEL_REVISION).send()?;
        
        match *f {
            "tokenizer.json" => tok = Some(p),
            "onnx/embed_tokens_q4.onnx" => encoder = Some(p),
            "onnx/decoder_model_merged_q4.onnx" => decoder = Some(p),
            _ => {}
        }
    }

    Ok(ModelPaths {
        tokenizer: tok.context("tokenizer.json")?,
        embed: encoder.context("embed model")?,
        decoder: decoder.context("decoder model")?,
    })
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

fn tok_err(e: Box<dyn std::error::Error + Send + Sync>) -> anyhow::Error {
    anyhow::Error::from_boxed(e)
}

fn single_token_ids(tok: &Tokenizer, forms: &[&str], label: &str) -> Result<Vec<i64>> {
    let mut ids = Vec::new();
    for w in forms {
        let enc = tok
            .encode(*w, false)
            .map_err(tok_err)
            .with_context(|| format!("tokenizing {label} form {w:?}"))?;
        if enc.get_ids().len() == 1 {
            ids.push(enc.get_ids()[0] as i64);
        }
    }
    if ids.is_empty() {
        bail!("{label}: none of {forms:?} tokenize to a single token — wrong tokenizer/model pairing?");
    }
    Ok(ids)
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
        let model_paths = fetch_weights()?;
        let env = fetch_env()?;

        let tok = Tokenizer::from_file(&model_paths.tokenizer)
            .map_err(tok_err)
            .with_context(|| format!(
                "loading tokenizer from {}", model_paths.tokenizer.display()
            ))?;

        let yes_ids = single_token_ids(&tok, &["yes", "Yes", "YES"], "yes")?;
        let no_ids = single_token_ids(&tok, &["no", "No", "NO"], "no")?;

        Ok(Self {
            tok,
            embed: make_session(&env, &model_paths.embed)?,
            decoder: make_session(&env, &model_paths.decoder)?,
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
        let prompt = Self::prompt(state, instructions);

        let enc = self.tok.encode(&*prompt, false)
            .map_err(tok_err)
            .with_context(|| format!("encoding prompt ({} chars)", prompt.len()))?;
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