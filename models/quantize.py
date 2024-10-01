import click
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
from transformers import AutoTokenizer


@click.command()
@click.option(
    "-i",
    "--input",
    "inputID",
    help="HuggingFace model ID for casual language modeling",
    required=False,
    default="meta-llama/Llama-2-7b-chat-hf",
    show_default=True,
    type=str,
)
@click.option(
    "-a",
    "--api-key",
    "hfApiKey",
    help="HuggingFace API key",
    required=True,
    type=str,
)
@click.option(
    "-o",
    "--output",
    "outputID",
    help="Path to save quantized model to",
    required=False,
    default="ov_model.int8",
    show_default=True,
    type=str,
)
def main(inputID: str, hfApiKey: str, outputID: str) -> None:
    quantization_config: OVWeightQuantizationConfig = (
        OVWeightQuantizationConfig(  # noqa: E501
            bits=8,
        )
    )

    model = OVModelForCausalLM.from_pretrained(
        model_id=inputID,
        export=True,
        use_auth_token=hfApiKey,
        force_download=False,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=inputID,
    )

    model.save_pretrained(save_directory=outputID)
    tokenizer.save_pretrained(save_directory=outputID)


if __name__ == "__main__":
    main()
