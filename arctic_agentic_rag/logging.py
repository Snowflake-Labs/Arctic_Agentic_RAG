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


import os

from loguru import logger


class Logger:
    def __init__(
        self,
        base_path: str,
        log_file: str = "run.log",
        out_file: str = "result.jsonl",
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
        outputs_format="jsonl",
    ):
        """
        Initialize the RAGLogger class with default configurations.

        Args:
            base_path (str): Path to the log directory
            log_file (str): Filename of the log file
            out_file (str): Filename of the output file, that will contain the final results.
            level (str): The minimum log level.
            rotation (str): Rotation policy (e.g., '10 MB', '1 day').
            retention (str): Retention policy (e.g., '10 days').
        """
        self.logger = logger
        self.log_file = log_file
        self.out_file = out_file
        self.level = level
        self.rotation = rotation
        self.retention = retention

        if outputs_format == "jsonl":
            self.logger.add(
                os.path.join(base_path, out_file),
                format="{message}",
                level="INFO",
                filter=self.is_final,
            )
        else:
            raise NotImplementedError

        self.logger.add(
            os.path.join(base_path, log_file),
            format="{time} - {level} - {message}",
            level=level,
        )

    @staticmethod
    def is_final(record):
        return record["extra"].get("is_final_result", False)

    def log_final_results(self, result):
        self.logger.bind(is_final_result=True).info(result)

    def log_info(self, message, **kwargs):
        """Log an info message."""
        self.logger.info(message, **kwargs)

    def log_debug(self, message, **kwargs):
        """Log a debug message."""
        self.logger.debug(message, **kwargs)

    def log_warning(self, message, **kwargs):
        """Log a warning message."""
        self.logger.warning(message, **kwargs)

    def log_error(self, message, **kwargs):
        """Log an error message."""
        self.logger.error(message, **kwargs)

    def log_exception(self, message, **kwargs):
        """Log an exception with traceback."""
        self.logger.exception(message, **kwargs)
