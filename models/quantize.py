import click
from optimum.intel import OVModelForCausalLM


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
    model = OVModelForCausalLM.from_pretrained(
        inputID,
        export=True,
        use_auth_token=hfApiKey,
        force_download=False,
    )

    model.save_pretrained(outputID)


if __name__ == "__main__":
    main()
