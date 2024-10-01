from pathlib import Path

import click

# from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

# import openvino as ov
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFacePipeline


def setupModel(modelPath: Path, tokenizer: str, device: str = "CPU") -> None:
    # ov_config: dict[str, str] = {
    #     "PERFORMANCE_HINT": "LATENCY",
    #     "NUM_STREAMS": "1",
    #     "CACHE_DIR": "",
    #     "KV_CACHE_PRECISION": "u8",
    #     "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
    # }

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # ov_llm = HuggingFacePipeline.from_model_id(
    #     model_id=modelPath,
    #     task="text-generation",
    #     backend="openvino",
    #     model_kwargs={"device": "CPU", "ov_config": ov_config},
    #     pipeline_kwargs={"max_new_tokens": 10},
    # )


@click.command()
@click.option(
    "-m",
    "--model",
    "modelPath",
    help="OpenVino quantized model directory",
    type=click.Path(
        exists=True,
        dir_okay=True,
        readable=True,
        path_type=Path,
        resolve_path=True,
    ),
    required=False,
    default="../models/ov_model.int8/openvino_model.xml",
    show_default=True,
)
@click.option(
    "-t",
    "--tokenizer",
    "tokenizerModel",
    help="Tokenizer to use with the model",
    type=str,
    required=False,
    default="meta-llama/Llama-2-7b-chat-hf",
    show_default=True,
)
def main(modelPath: Path, tokenizerModel: str) -> None:
    setupModel(modelPath=modelPath, tokenizer=tokenizerModel)


if __name__ == "__main__":
    main()
