"""OK-VQA loading script — reads from local HuggingFace-format parquet files."""

import glob
import os

import datasets
import pyarrow.parquet as pq


_CITATION = """\
@InProceedings{OKVQA,
  author    = {Kenneth Marino and Mohammad Rastegari and Ali Farhadi and Roozbeh Mottaghi},
  title     = {OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge},
  booktitle = {CVPR},
  year      = {2019},
}
"""

_DESCRIPTION = """\
OK-VQA is a visual question answering dataset that requires methods to draw upon
outside knowledge to answer questions about images.
"""

_HOMEPAGE = "https://okvqa.allenai.org/"

_LICENSE = "CC BY 4.0"

_SPLIT_FILE_PATTERNS = {
    "validation": "val2014-*-of-*.parquet",
    "test": "test-*-of-*.parquet",
}


class OKVQADataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {
                "question_type": datasets.Value("string"),
                "answers": datasets.Sequence(datasets.Value("string")),
                "answers_original": [
                    {
                        "answer": datasets.Value("string"),
                        "raw_answer": datasets.Value("string"),
                        "answer_confidence": datasets.Value("string"),
                        "answer_id": datasets.Value("int64"),
                    }
                ],
                "image_id": datasets.Value("int64"),
                "answer_type": datasets.Value("string"),
                "question_id": datasets.Value("int64"),
                "question": datasets.Value("string"),
                "image": datasets.Image(),
                "confidence": datasets.Value("int32"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = self.config.data_dir
        if not data_dir:
            raise ValueError(
                "This script works with local OK-VQA parquet files. "
                "The argument `data_dir` in `load_dataset()` is required."
            )

        parquet_dir = os.path.join(data_dir, "data")

        splits = []
        for split_name, pattern in _SPLIT_FILE_PATTERNS.items():
            files = sorted(glob.glob(os.path.join(parquet_dir, pattern)))
            if not files:
                continue
            hf_split = (datasets.Split.VALIDATION if split_name == "validation"
                        else datasets.Split.TEST)
            splits.append(
                datasets.SplitGenerator(
                    name=hf_split,
                    gen_kwargs={"parquet_files": files},
                )
            )
        return splits

    def _generate_examples(self, parquet_files):
        for path in parquet_files:
            table = pq.read_table(path)
            for i in range(len(table)):
                row = {col: table[col][i].as_py() for col in table.column_names}

                image_bytes = row.get("image", {})
                if isinstance(image_bytes, dict) and "bytes" in image_bytes:
                    image_value = {"bytes": image_bytes["bytes"],
                                   "path": image_bytes.get("path")}
                else:
                    image_value = image_bytes

                answers_orig = row.get("answers_original") or []
                answers_original = [
                    {
                        "answer": a.get("answer", ""),
                        "raw_answer": a.get("raw_answer", ""),
                        "answer_confidence": a.get("answer_confidence", ""),
                        "answer_id": a.get("answer_id", 0),
                    }
                    for a in answers_orig
                ]

                record = {
                    "question_type": row.get("question_type", ""),
                    "answers": row.get("answers", []),
                    "answers_original": answers_original,
                    "image_id": row.get("id_image", row.get("image_id", -1)),
                    "answer_type": row.get("answer_type", ""),
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "image": image_value,
                    "confidence": row.get("confidence", 0),
                }

                yield row["question_id"], record
