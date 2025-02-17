# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import tempfile

from arctic_agentic_rag.utils import get_data_processor_from_filename, stream_from_file


def test_stream_json_file():
    """
    Tests streaming data from a JSON file.
    """
    test_data = [{"id": 1}, {"id": 2}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
        json.dump(test_data, f)
        f.seek(0)
        result = list(stream_from_file(f.name))
        assert len(result) == 2, "Should stream two JSON objects from the file"


def test_stream_jsonl_file():
    """
    Tests streaming data from a JSONL file.
    """
    test_data = [{"id": 1}, {"id": 2}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=True) as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
        f.seek(0)
        result = list(stream_from_file(f.name))
        assert len(result) == 2, "Should stream two JSON objects from the JSONL file"


def test_crag_data_processor():
    """
    Tests the CRAG data processor functionality.
    """
    processor = get_data_processor_from_filename("crag-subset.json")
    test_example = {
        "question": "test",
        "search_results": [{"excerpt": "test context"}],
        "answer": "test answer",
    }
    result = processor(test_example)
    assert "contexts" in result, "Processed result should contain 'contexts' key"
