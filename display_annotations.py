#!/usr/bin/env python3

"""
- Script name: display_annotations.
- Author: Dan Bright, cosmoid@tuta.io.
- Description: A script to display NER tags from 
  JSON formatted annotated data files.
- Version: 1.2
"""

import os, json, random, re
import numpy as np
from spacy import displacy

ANNO_FILE_PATH: str = "../../data/sample/train/json"


class DisplayAnnotations:
    """
    Class that displays NER tags from annotated data.

    Consumes:
        dir_path: str = path directory of JSON annotation files
        jupyter: bool = whether script is being run as a Jupyter notebook
    Produces:
        Textual data from JSON formatted annotation files, tagged with
        named entities
    """

    def __init__(self, dir_path: str, jupyter: bool):
        # define variables
        self._dir_path: str = dir_path
        self._jupyter: bool = jupyter
        self._annotations: list[dict] = []
        self._label_colors: dict = dict()
        # run methods
        self._read_json_files()
        self._visualise_annotations()

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


if __name__ == "__main__":
    DisplayAnnotations(dir_path=ANNO_FILE_PATH, jupyter=False)