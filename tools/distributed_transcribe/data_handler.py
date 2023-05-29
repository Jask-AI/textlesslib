# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torchaudio

import pathlib

import logging

logger = logging.getLogger(__name__)


class ManifestDataset:
    def __init__(self, manifest):
        with open(manifest, "r") as fin:
            # self.root = pathlib.Path(fin.readline().strip())
            self.files = [x.strip().split('|')[0] for x in fin.readlines()]
        # print(self.root)
        logger.info(
            f"Init dataset with root in , containing {len(self.files)} files"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, k):
        path = self.files[k]
        data, sr = torchaudio.load(str(path))
        # data = encoder.maybe_resample(data, sr)
        data = torchaudio.functional.resample(
            data, sr, 16_000
        )
        # assert sr == 16_000
        return data.squeeze(0), path
