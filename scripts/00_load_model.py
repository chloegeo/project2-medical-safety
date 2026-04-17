"""
00_load_model.py -- Smoke test for model loading and activation extraction.
 
Purpose:
    Verify that we can (a) load a HuggingFace instruction-tuned model,
    (b) apply its chat template correctly, (c) generate a response, and
    (d) extract MLP activations at a chosen layer via forward hooks.
    Validates the entire mechanistic pipeline on a tiny scale before
    running expensive experiments.
 
Usage:
    # Validation with 1B model (no quantization needed, works on Colab T4):
    python scripts/00_load_model.py --model meta-llama/Llama-3.2-1B-Instruct
 
    # Production with 8B model on Colab Pro A100 (use 4-bit quantization):
    python scripts/00_load_model.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --quantize
 
    # Test a different layer:
    python scripts/00_load_model.py --model meta-llama/Llama-3.2-1B-Instruct --layer 12
 
If this script fails, everything downstream will fail. Fix this first.
"""
import argparse
import sys
 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 
# A medical dosage question -- non-graphic, tests that safety-tuned models
# handle factual medical queries gracefully. We're testing the pipeline here,
# not the safety behavior itself.
TEST_PROMPT = "What is the maximum safe dose of acetaminophen for an adult?"
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test: load model, generate, extract MLP activation."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model identifier (must be an *-Instruct variant).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Transformer block index to hook for activation extraction.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable bitsandbytes 4-bit quantization. Use for 8B+ models on limited VRAM.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max tokens to generate in the smoke-test response.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()
 
 
def set_seeds(seed: int) -> None:
    """Set torch CPU and CUDA seeds."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
 
def build_model_kwargs(quantize: bool) -> dict:
    """Construct kwargs for AutoModelForCausalLM.from_pretrained."""
    kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if quantize:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "bitsandbytes is required for --quantize. "
                "Install with: pip install bitsandbytes"
            ) from e
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return kwargs
 
 
def register_activation_hook(model, layer_idx: int):
    """
    Register a forward hook on the MLP output of a specific transformer block.
 
    Returns (hook_handle, captured_dict). captured_dict["activation"] will be
    populated after a forward pass.
 
    The hook grabs the hidden state AFTER the MLP but BEFORE the residual add,
    which is the standard target for safety-neuron style analysis
    (Chen et al. 2024, "Finding Safety Neurons in LLMs").
    For LLaMA-family models, this is accessible via model.model.layers[i].mlp.
    """
    captured: dict = {}
 
    def hook(module, inputs, output):
        # output shape: [batch, seq_len, hidden_dim]
        # Detach and move to CPU immediately to avoid holding GPU memory.
        captured["activation"] = output.detach().to("cpu")
 
    try:
        target_module = model.model.layers[layer_idx].mlp
    except AttributeError as e:
        raise RuntimeError(
            f"Could not find model.model.layers[{layer_idx}].mlp. "
            f"This script assumes LLaMA-family architecture. "
            f"Model type: {type(model).__name__}"
        ) from e
    except IndexError as e:
        n_layers = len(model.model.layers)
        raise RuntimeError(
            f"Layer index {layer_idx} out of range. Model has {n_layers} layers."
        ) from e
 
    handle = target_module.register_forward_hook(hook)
    return handle, captured
 
 
def main() -> int:
    args = parse_args()
    set_seeds(args.seed)
 
    print("=" * 70)
    print(f"SMOKE TEST: {args.model}")
    print("=" * 70)
    print(f"  Quantize:       {args.quantize}")
    print(f"  Target layer:   {args.layer}")
    print(f"  Seed:           {args.seed}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device:    {torch.cuda.get_device_name(0)}")
        print(
            f"  Total VRAM:     "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    print()
 
    # --- Load tokenizer ---
    print("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"      Vocab size:           {tokenizer.vocab_size}")
    print(f"      Chat template present: {tokenizer.chat_template is not None}")
    assert tokenizer.chat_template is not None, (
        "This script assumes an instruction-tuned model with a chat template. "
        f"{args.model} has no chat_template -- use an *-Instruct variant."
    )
 
    # --- Load model ---
    print("[2/5] Loading model (this may take 30-120s)...")
    model_kwargs = build_model_kwargs(args.quantize)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
 
    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    print(f"      Num transformer layers: {n_layers}")
    print(f"      Hidden dim:             {hidden_dim}")
    print(f"      Model dtype:            {next(model.parameters()).dtype}")
    print(f"      Model device:           {next(model.parameters()).device}")
    assert args.layer < n_layers, f"Layer {args.layer} >= {n_layers} layers."
 
    # --- Apply chat template ---
    print("[3/5] Applying chat template to test prompt...")
    messages = [{"role": "user", "content": TEST_PROMPT}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    print(f"      Prompt tokens: {inputs['input_ids'].shape[1]}")
 
    # --- Register hook and run forward pass ---
    print(f"[4/5] Registering hook on layer {args.layer} MLP and generating...")
    handle, captured = register_activation_hook(model, args.layer)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        handle.remove()
 
    response = tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
 
    # --- Validate capture ---
    print("[5/5] Validating captured activation...")
    assert "activation" in captured, "Hook did not fire -- check layer path."
    act = captured["activation"]
    print(f"      Activation shape: {tuple(act.shape)}")
    print(f"      Activation dtype: {act.dtype}")
    assert act.shape[-1] == hidden_dim, (
        f"Expected hidden dim {hidden_dim}, got {act.shape[-1]}."
    )
    assert not torch.isnan(act).any(), "Activation contains NaN."
    assert not torch.isinf(act).any(), "Activation contains Inf."
 
    # --- Report ---
    print()
    print("-" * 70)
    print("PROMPT:")
    print(f"  {TEST_PROMPT}")
    print()
    print("RESPONSE:")
    print(f"  {response.strip()}")
    print()
    print(f"ACTIVATION STATS (layer {args.layer} MLP output, last token):")
    last_token_act = act[0, -1, :].float()
    print(f"  mean = {last_token_act.mean().item():.4f}")
    print(f"  std  = {last_token_act.std().item():.4f}")
    print(f"  min  = {last_token_act.min().item():.4f}")
    print(f"  max  = {last_token_act.max().item():.4f}")
    print("-" * 70)
    print()
    print("SMOKE TEST PASSED")
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
