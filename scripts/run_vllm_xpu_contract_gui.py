#!/usr/bin/env python3
"""Run the loopback-only Intel vLLM Engine Contract Console."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from typing import NamedTuple, Protocol, cast

from exo.diagnostics.vllm_xpu_contract_app import create_app


class ServerArguments(NamedTuple):
    host: str
    port: int
    allow_remote: bool


class _UvicornModule(Protocol):
    def run(
        self,
        app: object,
        *,
        host: str,
        port: int,
        access_log: bool,
    ) -> None: ...


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=52416)
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Explicitly allow a non-loopback HTTP listener",
    )
    return parser


def parse_arguments(
    parser: argparse.ArgumentParser,
    argument_values: Sequence[str] | None = None,
) -> ServerArguments:
    namespace = parser.parse_args(argument_values)
    return ServerArguments(
        host=cast(str, namespace.host),
        port=cast(int, namespace.port),
        allow_remote=cast(bool, namespace.allow_remote),
    )


def validate_arguments(
    arguments: ServerArguments, parser: argparse.ArgumentParser
) -> None:
    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if arguments.host not in {"127.0.0.1", "::1", "localhost"} and not (
        arguments.allow_remote
    ):
        parser.error("a non-loopback --host requires --allow-remote")


def run(arguments: ServerArguments) -> None:
    uvicorn = cast(
        _UvicornModule,
        cast(object, importlib.import_module("uvicorn")),
    )
    uvicorn.run(
        create_app(),
        host=arguments.host,
        port=arguments.port,
        access_log=False,
    )


def main(argument_values: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parse_arguments(parser, argument_values)
    validate_arguments(arguments, parser)
    run(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
