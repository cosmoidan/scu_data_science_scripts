#!/usr/bin/env python3

"""
- Script name: display_annotations.
- Author: Dan Bright, cosmoid@tuta.io.
- Description: A script to display NER tags from 
  JSON formatted annotated data files.
- Version: 1.3
"""

import os, json, re
import numpy as np
import pandas as pd
from spacy import displacy

JUPYTER: bool = False
WRITE_JSON_FILE: bool = False
WRITE_EXCEL_FILE: bool = True
ANNO_FILE_PATH: str = "../../data/sample/train/json"
OUTPUT_JSON_PATH: str = "../../data/annotations.json"
OUTPUT_EXCEL_PATH: str = "../../data/annotations.xlsx"


class DisplayAnnotations:
    """
    Class that displays NER tags from annotated data.

    Consumes:
        - dir_path: str = path directory of JSON annotation files
        - jupyter: bool = whether script is being run as a Jupyter notebook
        - write_json_file: bool = whether to write a JSON file containing entity classes
          and their labelled tokens
        - write_excel_file: bool = whether to write an EXCEL file containing entity classes
          and their labelled tokens
        - output_json_url: str = the URL of the output JSON file (optional)
        - output_excel_url: str = the URL of the output EXCEL file (optional)
    Produces:
        - Textual data from JSON formatted annotation files, tagged with
          named entities
        - JSON formatted file containing tokens (strings) that were annotated
          for each entity class (optional)
        - EXCEL formatted file containing tokens (strings) that were annotated
          for each entity class (optional)
    Notes:
        - Record number is derived from the filenames of the consumed text files, which
          MUST be named according to the convention of `record_n.txt`, where n is record number.
    """

    def __init__(
        self,
        dir_path: str,
        jupyter: bool,
        write_json_file: bool = False,
        write_excel_file: bool = False,
        output_json_url: str = "",
        output_excel_url: str = "",
    ) -> None:
        # define variables
        self._dir_path: str = dir_path
        self._jupyter: bool = jupyter
        self._output_json_url = output_json_url
        self._output_excel_url = output_excel_url
        self._annotations: list[dict] = []
        self._output: list[dict(list)] = []
        self._label_colors: dict = dict()
        # run methods
        self._read_json_files()
        # self._visualise_annotations()
        self._format_output()
        self._write_json_output() if write_json_file else None
        self._write_excel_output() if write_excel_file else None

    def _read_json_files(self) -> None:
        """Read all JSON files in the given directory and return their contents
        as a list of python dictionaries."""
        for filename in os.listdir(self._dir_path):
            if filename.endswith(".json"):
                with open(os.path.join(self._dir_path, filename), "r") as file:
                    annotation = json.load(file)
                    for paragraph in annotation["annotations"]:
                        ents: list[dict] = []
                        for e in paragraph[1]["entities"]:
                            ents.append({"start": e[0], "end": e[1], "label": e[2]})
                        self._annotations.append(
                            {
                                "labels": annotation["classes"],
                                "text": paragraph[0],
                                "ents": ents,
                                "title": filename,
                                "rec_num": int(re.findall(r"\d+", filename)[0]),
                            }
                        )
        self._annotations = sorted(self._annotations, key=lambda e: e["rec_num"])

    @staticmethod
    def _gen_colors(annotations) -> dict:
        # generates random colors & assigns them to named entity classes
        _label_count: int = len(annotations[0]["labels"])
        _min_color_diff: int = 10

        colors = np.array([np.random.randint(128, 255, 3, dtype=int)])
        for _ in range(_label_count - 1):
            val = np.array([np.random.randint(128, 255, 3, dtype=int)])
            created = False
            while not created:
                if (
                    [
                        val[:, 0] - _min_color_diff,
                        val[:, 0],
                        val[:, 0] + _min_color_diff,
                    ]
                    not in colors[:, 0]
                    or [
                        val[:, 1] - _min_color_diff,
                        val[:, 1],
                        val[:, 1] + _min_color_diff,
                    ]
                    not in colors[:, 1]
                    or [
                        val[:, 2] - _min_color_diff,
                        val[:, 2],
                        val[:, 2] + _min_color_diff,
                    ]
                    not in colors[:, 2]
                ):
                    colors = np.append(colors, val, axis=0)
                    created = True
                else:
                    val = np.array([np.random.randint(128, 255, 3, dtype=int)])

        colors_lst: list[str] = [f"rgb({c[0]},{c[1]},{c[2]})" for c in colors]
        return dict(zip(annotations[0]["labels"], colors_lst))

    def _visualise_annotations(self) -> None:
        # displays tagged annotations in Jupyter notebook or serves as web page
        if self._jupyter:
            for a in self._annotations:
                print(f"Annotations for file: {a['title']}")
                displacy.render(
                    a,
                    manual=True,
                    style="ent",
                    jupyter=True,
                    options={"colors": self._gen_colors(self._annotations)},
                )
                print("\n----------------\n")
        else:
            displacy.serve(
                self._annotations,
                manual=True,
                style="ent",
                options={"colors": self._gen_colors(self._annotations)},
                host="127.0.0.1",
                page=True,
                minify=True,
                port=8753,
            )

    def _format_output(self) -> None:
        # creates formatted output, for JSON/EXCEL
        for record in self._annotations:
            rec_output: dict(list) = dict()
            rec_output["RECORD_ID"] = record["rec_num"]
            for ent in record["ents"]:
                rec_output.setdefault(ent["label"], []).append(
                    record["text"][ent["start"] : ent["end"]]
                )
            self._output.append(rec_output)

    def _write_json_output(self) -> None:
        # writes formatted output to JSON file
        with open(self._output_json_url, "w") as fp:
            json.dump(self._output, fp)

    def _write_excel_output(self) -> None:
        # writes formatted output to EXCEL file
        df = pd.read_json(json.dumps(self._output))
        df.to_excel(self._output_excel_url)


if __name__ == "__main__":
    DisplayAnnotations(
        dir_path=ANNO_FILE_PATH,  # path to directory containing input JSON files (string)
        jupyter=JUPYTER,  # Running on Jupyter notebook? True|False
        write_json_file=WRITE_JSON_FILE,  # write JSON formatted output file? True|False
        write_excel_file=WRITE_EXCEL_FILE,  # write EXCEL formatted output file? True|False
        output_json_url=OUTPUT_JSON_PATH,  # path to JSON formatted output file (if any) (string)
        output_excel_url=OUTPUT_EXCEL_PATH,  # path to EXCEL formatted output file (if any) (string)
    )