from pathlib import Path

import click
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline


def setupModel(modelPath: Path, device: str = "CPU") -> HuggingFacePipeline:
    ov_config: dict[str, str] = {
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": "1",
        "CACHE_DIR": "",
        "KV_CACHE_PRECISION": "u8",
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
    }

    return HuggingFacePipeline.from_model_id(
        model_id=modelPath.__str__(),
        task="text-generation",
        backend="openvino",
        model_kwargs={"device": device, "ov_config": ov_config},
    )


def prompt(hfp: HuggingFacePipeline) -> None:
    template = """Question: {question}

    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)

    generation_config = {
        "skip_prompt": True,
        "pipeline_kwargs": {"max_new_tokens": 100},
    }
    chain = prompt | hfp.bind(**generation_config)

    question = "What is electroencephalography?"

    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)


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
    default="../models/ov_model.int8",
    show_default=True,
)
def main(modelPath: Path) -> None:
    hfp: HuggingFacePipeline = setupModel(modelPath=modelPath)

    prompt(hfp=hfp)


if __name__ == "__main__":
    main()
